from __future__ import annotations

import csv
import json
import os
import pickle
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Lyapunov.direct_lyapunov_mpc import (
    build_direct_lyapunov_run_bundle,
    design_direct_lyapunov_mpc_solver,
    make_direct_lyapunov_comparison_record,
    save_direct_lyapunov_debug_artifacts,
)
from Lyapunov.gart_lmpc import solve_gart_lmpc_step
from Lyapunov.gart_target import GARTTargetState, jsonable, select_gart_target
from Simulation.mpc import compute_observer_gain
from Simulation.run_mpc_lyapunov import _reset_system_on_entry, _set_system_input_phys, _system_io_phys
from Simulation.system_functions import PolymerCSTR
from TD3Agent.reward_functions import make_reward_fn_relative_QR
from utils.direct_lyapunov_study import (
    DIRECT_TWO_SETPOINT_Y_PHYS,
    direct_disturbance_test_cycle,
)
from utils.gart_defaults import (
    discover_gart_case_values,
    gart_rl_observation,
    make_gart_mpc_config,
    make_gart_target_config,
)
from utils.gart_runtime import ResourceGuard
from utils.helpers import generate_setpoints_training_rl_gradually
from utils.polymer_td3_defaults import DEFAULT_U_MAX_PHYS, DEFAULT_U_MIN_PHYS
from utils.scaling_helpers import apply_min_max, reverse_min_max
from utils.td3_helpers import load_and_prepare_system_data
from utils.lyapunov_utils import get_y_sp_step


PREDICT_H = 9
CONT_H = 3
RHO_LYAP = 0.98
LYAP_EPS = 1.0e-3
SLACK_PENALTY = 1.0e6
QY_DIAG = np.array([5.0, 1.0], dtype=float)
SU_DIAG = np.array([1.0, 1.0], dtype=float)
RDU_DIAG = np.array([1.0, 1.0], dtype=float)


FINAL_GART_CASE_NAME = "gartlmpc"
FINAL_GART_MPC_OBJECTIVE = "raw"
FINAL_GART_LYAPUNOV_MODE = "hard"
FINAL_GART_TARGET_OVERRIDES: dict[str, Any] = {
    "disable_u_mid_tiebreak": True,
    "disable_x_smoothing": True,
    "disable_y_smoothing": True,
    "input_headroom_frac": 0.01,
    "dx_s_max_abs": 0.05,
    "dy_s_max_abs": 1.0,
    "d_rate_scale": 1.0,
    "adaptive_rate_enabled": False,
}


def _jsonable(value: Any) -> Any:
    return jsonable(value)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2), encoding="utf-8")


def _is_scalar(value: Any) -> bool:
    return value is None or isinstance(value, (str, bool, int, float, np.bool_, np.integer, np.floating))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    scalar_rows = [{key: _jsonable(value) for key, value in row.items() if _is_scalar(value)} for row in rows]
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in scalar_rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in scalar_rows:
            writer.writerow(row)


def _build_polymer_setup() -> dict[str, Any]:
    Ad = 2.142e17
    Ed = 14897.0
    Ap = 3.816e10
    Ep = 3557.0
    At = 4.50e12
    Et = 843.0
    fi = 0.6
    m_delta_H_r = -6.99e4
    hA = 1.05e6
    rhocp = 1506.0
    rhoccpc = 4043.0
    Mm = 104.14
    system_params = np.array([Ad, Ed, Ap, Ep, At, Et, fi, m_delta_H_r, hA, rhocp, rhoccpc, Mm], dtype=float)

    CIf = 0.5888
    CMf = 8.6981
    Qi = 108.0
    Qs = 459.0
    Tf = 330.0
    Tcf = 295.0
    V = 3000.0
    Vc = 3312.4
    system_design_params = np.array([CIf, CMf, Qi, Qs, Tf, Tcf, V, Vc], dtype=float)

    system_steady_state_inputs = np.array([471.6, 378.0], dtype=float)
    delta_t = 0.5
    cstr_ss = PolymerCSTR(system_params, system_design_params, system_steady_state_inputs, delta_t, deviation_form=False)
    steady_states = {"ss_inputs": system_steady_state_inputs.copy(), "y_ss": cstr_ss.y_ss.copy()}
    return {
        "system_params": system_params,
        "system_design_params": system_design_params,
        "system_steady_state_inputs": system_steady_state_inputs,
        "delta_t": delta_t,
        "steady_states": steady_states,
        "nominal_qi": 108.0,
        "nominal_qs": 459.0,
        "nominal_ha": 1.05e6,
        "qi_change": 0.95,
        "qs_change": 1.05,
        "ha_change": 0.92,
    }


def _make_system(setup: dict[str, Any]) -> PolymerCSTR:
    return PolymerCSTR(
        setup["system_params"],
        setup["system_design_params"],
        setup["system_steady_state_inputs"],
        setup["delta_t"],
        deviation_form=False,
    )


def _build_context(
    *,
    results_roots: list[str] | None = None,
    max_result_files: int = 3,
    max_npz_bytes: int = 100_000_000,
    max_rows_per_array: int = 2000,
) -> dict[str, Any]:
    setup = _build_polymer_setup()
    steady_states = setup["steady_states"]
    system_data = load_and_prepare_system_data(
        steady_states=steady_states,
        setpoint_y=DIRECT_TWO_SETPOINT_Y_PHYS.copy(),
        u_min=DEFAULT_U_MIN_PHYS,
        u_max=DEFAULT_U_MAX_PHYS,
        system_dict_path=os.path.join("Data", "system_dict"),
        augmentation_style="rawlings",
        augmentation_mode="output_disturbance",
    )
    A_aug = system_data["A_aug"]
    B_aug = system_data["B_aug"]
    C_aug = system_data["C_aug"]
    data_min = system_data["data_min"]
    data_max = system_data["data_max"]
    n_inputs = int(B_aug.shape[1])

    y_sp_scenario = apply_min_max(DIRECT_TWO_SETPOINT_Y_PHYS, data_min[n_inputs:], data_max[n_inputs:]) - apply_min_max(
        steady_states["y_ss"], data_min[n_inputs:], data_max[n_inputs:]
    )
    poles = np.array([0.44619852, 0.33547649, 0.36380595, 0.70467118, 0.3562966, 0.42900673, 0.4228262, 0.96916776, 0.91230187])
    observer_gain = compute_observer_gain(A_aug, C_aug, poles)
    u_dev_min = np.asarray(system_data["b_min"], dtype=float).reshape(-1)
    u_dev_max = np.asarray(system_data["b_max"], dtype=float).reshape(-1)
    bnds = tuple((float(lo), float(hi)) for _ in range(CONT_H) for lo, hi in zip(u_dev_min, u_dev_max))
    ic_opt = np.zeros(n_inputs * CONT_H, dtype=float)
    lmpc_obj = design_direct_lyapunov_mpc_solver(
        A_aug=A_aug,
        B_aug=B_aug,
        C_aug=C_aug,
        Qy_diag=QY_DIAG,
        NP=PREDICT_H,
        NC=CONT_H,
        Su_diag=SU_DIAG,
        u_min=u_dev_min,
        u_max=u_dev_max,
        Rdu_diag=RDU_DIAG,
        terminal_set_on=True,
        terminal_alpha_scale=1.0,
    )
    reward_config, reward_fn = make_reward_fn_relative_QR(
        data_min=data_min,
        data_max=data_max,
        n_inputs=n_inputs,
        k_rel=np.array([0.003, 0.0003], dtype=float),
        band_floor_phys=np.array([0.006, 0.07], dtype=float),
        Q_diag=QY_DIAG,
        R_diag=RDU_DIAG,
        tau_frac=0.7,
        gamma_out=0.5,
        gamma_in=0.5,
        beta=7.0,
        gate="geom",
        lam_in=1.0,
        bonus_kind="exp",
        bonus_k=12.0,
        bonus_p=0.6,
        bonus_c=20.0,
    )
    discovered = discover_gart_case_values(
        system_data,
        setup,
        results_roots=results_roots,
        max_result_files=max_result_files,
        max_npz_bytes=max_npz_bytes,
        max_rows_per_array=max_rows_per_array,
    )
    return {
        "setup": setup,
        "system_data": system_data,
        "lmpc_obj": lmpc_obj,
        "observer_gain": observer_gain,
        "y_sp_scenario": y_sp_scenario,
        "u_dev_min": u_dev_min,
        "u_dev_max": u_dev_max,
        "bnds": bnds,
        "ic_opt": ic_opt,
        "reward_config": reward_config,
        "reward_fn": reward_fn,
        "discovered": discovered,
    }


def _setpoint_schedule(ctx: dict[str, Any], *, n_tests: int, set_points_len: int, force_final_test: bool = True) -> tuple[Any, ...]:
    setup = ctx["setup"]
    test_cycle = direct_disturbance_test_cycle(int(n_tests))
    return generate_setpoints_training_rl_gradually(
        ctx["y_sp_scenario"],
        int(n_tests),
        int(set_points_len),
        0,
        test_cycle,
        setup["nominal_qi"],
        setup["nominal_qs"],
        setup["nominal_ha"],
        setup["qi_change"],
        setup["qs_change"],
        setup["ha_change"],
        force_final_test=force_final_test,
    )


def _target_classification(error_inf: float | None, target_config: Any) -> dict[str, bool]:
    if error_inf is None:
        return {
            "target_exact": False,
            "target_good": False,
            "target_acceptable": False,
            "target_unreachable": False,
            "classified_unreachable": False,
        }
    value = float(error_inf)
    exact = value <= float(getattr(target_config, "target_exact_tol", 1.0e-6))
    good = value <= float(getattr(target_config, "target_good_tol", 0.1))
    acceptable = value <= float(getattr(target_config, "target_acceptable_tol", 0.5))
    unreachable = value > float(getattr(target_config, "target_acceptable_tol", 0.5))
    return {
        "target_exact": bool(exact),
        "target_good": bool(good),
        "target_acceptable": bool(acceptable),
        "target_unreachable": bool(unreachable),
        "classified_unreachable": bool(unreachable),
    }


def _disturbance_diag_scalars(diag: dict[str, Any]) -> dict[str, Any]:
    disturbance = diag.get("disturbance", {}) if isinstance(diag, dict) else {}
    return {
        "d_raw_gap_inf": disturbance.get("d_raw_gap_inf"),
        "d_rate_max_base_inf": disturbance.get("d_rate_max_base_inf"),
        "d_rate_max_effective_inf": disturbance.get("d_rate_max_effective_inf"),
    }


def run_gart_closed_loop_case(
    ctx: dict[str, Any],
    *,
    case_name: str,
    mpc_objective: str,
    lyapunov_mode: str,
    mode: str,
    n_tests: int,
    set_points_len: int,
    target_overrides: dict[str, Any] | None = None,
    mpc_overrides: dict[str, Any] | None = None,
    guard: ResourceGuard | None = None,
) -> dict[str, Any]:
    setup = ctx["setup"]
    system_data = ctx["system_data"]
    lmpc_obj = ctx["lmpc_obj"]
    data_min = system_data["data_min"]
    data_max = system_data["data_max"]
    n_inputs = int(lmpc_obj.B.shape[1])
    n_outputs = int(lmpc_obj.C.shape[0])
    n_aug = int(lmpc_obj.A.shape[0])
    n_x = int(n_aug - n_outputs)

    system = _make_system(setup)
    system.Qi = setup["nominal_qi"]
    system.Qs = setup["nominal_qs"]
    system.hA = setup["nominal_ha"]
    _reset_system_on_entry(system)
    system.Qi = setup["nominal_qi"]
    system.Qs = setup["nominal_qs"]
    system.hA = setup["nominal_ha"]

    y_sp, nFE, sub_changes, time_in_sub_episodes, _, _, qi, qs, ha = _setpoint_schedule(
        ctx, n_tests=n_tests, set_points_len=set_points_len
    )
    ss_scaled_inputs = apply_min_max(setup["steady_states"]["ss_inputs"], data_min[:n_inputs], data_max[:n_inputs])
    y_ss_scaled = apply_min_max(setup["steady_states"]["y_ss"], data_min[n_inputs:], data_max[n_inputs:])

    y_mpc = np.zeros((int(nFE) + 1, n_outputs), dtype=float)
    y_mpc[0, :] = _system_io_phys(system, setup["steady_states"])[1]
    u_applied_phys = np.zeros((int(nFE), n_inputs), dtype=float)
    yhat = np.zeros((n_outputs, int(nFE)), dtype=float)
    xhatdhat = np.zeros((n_aug, int(nFE) + 1), dtype=float)
    rewards = np.zeros(int(nFE), dtype=float)
    direct_info_storage: list[dict[str, Any]] = []
    target_info_storage: list[dict[str, Any]] = []
    delta_y_storage: list[np.ndarray] = []
    delta_u_storage: list[np.ndarray] = []
    avg_rewards: list[float] = []
    target_state: GARTTargetState | None = None
    IC_opt = ctx["ic_opt"].copy()
    target_config = make_gart_target_config(ctx["discovered"], **(target_overrides or {}))
    mpc_config = make_gart_mpc_config(ctx["discovered"], objective=mpc_objective, lyapunov_mode=lyapunov_mode, **(mpc_overrides or {}))

    for step_idx in range(int(nFE)):
        if guard is not None:
            guard.tick_closed_loop()
        x0_aug = xhatdhat[:, step_idx].copy()
        scaled_current_input = apply_min_max(system.current_input, data_min[:n_inputs], data_max[:n_inputs])
        u_prev_dev = scaled_current_input - ss_scaled_inputs
        y_sp_k = get_y_sp_step(y_sp, step_idx, n_outputs)
        y_prev_scaled = apply_min_max(y_mpc[step_idx, :], data_min[n_inputs:], data_max[n_inputs:]) - y_ss_scaled
        yhat_now = np.asarray(lmpc_obj.C @ x0_aug, dtype=float).reshape(-1)
        innovation = y_prev_scaled - yhat_now
        if guard is not None:
            guard.tick_target()
        target_result, target_state = select_gart_target(
            lmpc_obj.A,
            lmpc_obj.B,
            lmpc_obj.C,
            x0_aug,
            y_sp_k,
            ctx["u_dev_min"],
            ctx["u_dev_max"],
            state=target_state,
            config=target_config,
            P_x=lmpc_obj.P_x,
            K_x=lmpc_obj.K_x,
            innovation=innovation,
            u_smooth_ref=u_prev_dev,
        )
        r_cmd = None if target_result.r_cmd is None else np.asarray(target_result.r_cmd, dtype=float).reshape(n_outputs)
        y_s = None if target_result.y_s is None else np.asarray(target_result.y_s, dtype=float).reshape(n_outputs)
        d_s = None if target_result.d_cert is None else np.asarray(target_result.d_cert, dtype=float).reshape(n_outputs)
        target_diag = target_result.diagnostics if isinstance(target_result.diagnostics, dict) else {}
        dx_s_max = None if target_config.dx_s_max is None else np.asarray(target_config.dx_s_max, dtype=float).reshape(-1)
        dc_rate_inf = target_diag.get("disturbance", {}).get("d_cert_delta_inf")
        disturbance_scalars = _disturbance_diag_scalars(target_diag)
        target_info = target_result.to_dict()
        target_info.update(
            {
                "step": step_idx,
                "target_mode": "gart",
                "solve_stage": target_result.stage,
                "d_s": None if d_s is None else d_s.copy(),
                "r_cmd_minus_y_sp": None if r_cmd is None else r_cmd - y_sp_k,
                "y_s_minus_r_cmd": None if y_s is None or r_cmd is None else y_s - r_cmd,
                "governor_probe_available": target_result.contraction_probe_success is not None,
                "governor_probe_success": target_result.contraction_probe_success,
                "governor_probe_margin_good": target_result.contraction_probe_margin_good,
                "governor_probe_margin": target_result.contraction_probe_margin,
                "governor_probe_min_value": target_result.contraction_probe_min_value,
                "governor_probe_bound": target_result.contraction_probe_bound,
                "governor_probe_status": target_result.status,
                "target_solve_success": target_result.solve_success,
                "target_accepted": target_result.accepted,
                "target_usable_for_lmpc": target_result.usable_for_lmpc,
                "target_rejection_reason": target_result.rejection_reason,
                "target_rate_inf": target_result.target_rate_y_inf,
                "target_rate_x_inf": target_result.target_rate_x_inf,
                "dx_s_inf": target_result.target_rate_x_inf,
                "dc_rate_inf": dc_rate_inf,
                "d_cert_delta_inf": dc_rate_inf,
                "dx_s_max_active": dx_s_max is not None,
                "dx_s_max_inf": None if dx_s_max is None else float(np.max(np.abs(dx_s_max))),
                "command_move_inf": target_result.target_rate_y_inf,
                "input_headroom_frac": target_config.input_headroom_frac,
                "stage2_u_smooth_source": target_diag.get("stage2_u_smooth_source"),
                "residual_total_norm": target_result.target_error_inf,
                **disturbance_scalars,
                **_target_classification(target_result.target_error_inf, target_config),
            }
        )
        target_info_storage.append(target_info)

        u_dev_apply, IC_opt, step_info = solve_gart_lmpc_step(
            lmpc_obj,
            x0_aug,
            y_sp_k,
            target_result,
            u_prev_dev,
            IC_opt,
            ctx["bnds"],
            ctx["u_dev_min"],
            ctx["u_dev_max"],
            mpc_config,
        )
        if guard is not None:
            guard.tick_solver()

        u_scaled = u_dev_apply + ss_scaled_inputs
        u_phys = reverse_min_max(u_scaled, data_min[:n_inputs], data_max[:n_inputs])
        u_applied_phys[step_idx, :] = u_phys.copy()
        delta_u = u_scaled - scaled_current_input

        if mode == "disturb":
            system.hA = ha[step_idx]
            system.Qs = qs[step_idx]
            system.Qi = qi[step_idx]

        _set_system_input_phys(system, setup["steady_states"], u_phys)
        system.step()

        y_phys = _system_io_phys(system, setup["steady_states"])[1]
        y_mpc[step_idx + 1, :] = y_phys
        y_current_scaled = apply_min_max(y_mpc[step_idx + 1, :], data_min[n_inputs:], data_max[n_inputs:]) - y_ss_scaled
        delta_y = y_current_scaled - y_sp_k
        y_target_step = step_info.get("y_target")
        delta_y_target = None if y_target_step is None else y_current_scaled - np.asarray(y_target_step, dtype=float).reshape(n_outputs)

        yhat[:, step_idx] = yhat_now
        xhat_next_openloop = lmpc_obj.A @ x0_aug + lmpc_obj.B @ u_dev_apply
        observer_correction = ctx["observer_gain"] @ innovation
        xhatdhat[:, step_idx + 1] = xhat_next_openloop + observer_correction

        y_sp_phys = reverse_min_max(y_sp_k + y_ss_scaled, data_min[n_inputs:], data_max[n_inputs:])
        reward = ctx["reward_fn"](delta_y, delta_u, y_sp_phys)
        rewards[step_idx] = reward
        delta_y_storage.append(delta_y.copy())
        delta_u_storage.append(delta_u.copy())
        obs = gart_rl_observation(system_data["min_max_dict"], x0_aug, target_result.d_cert, y_sp_k, u_prev_dev, target_result)
        step_info.update(
            {
                "step": step_idx,
                "case_name": case_name,
                "target_mode": "gart",
                "plant_mode": mode,
                "disturbance_after_step": False,
                "target_stage": target_result.stage,
                "target_residual_total_norm": target_result.target_error_inf,
                "target_quality_enabled": True,
                "target_quality_ok": bool(target_result.accepted),
                "target_quality_reason": target_result.status,
                "target_quality_mismatch_inf": target_result.target_error_inf,
                "target_quality_residual_norm": target_result.target_error_inf,
                "target_solve_success": target_result.solve_success,
                "target_accepted": target_result.accepted,
                "target_usable_for_lmpc": target_result.usable_for_lmpc,
                "target_rejection_reason": target_result.rejection_reason,
                "d_s": None if d_s is None else d_s.copy(),
                "r_cmd": None if r_cmd is None else r_cmd.copy(),
                "r_cmd_minus_y_sp": None if r_cmd is None else r_cmd - y_sp_k,
                "y_s_minus_r_cmd": None if y_s is None or r_cmd is None else y_s - r_cmd,
                "target_rate_inf": target_result.target_rate_y_inf,
                "target_rate_x_inf": target_result.target_rate_x_inf,
                "dx_s_inf": target_result.target_rate_x_inf,
                "dc_rate_inf": dc_rate_inf,
                "d_cert_delta_inf": dc_rate_inf,
                "dx_s_max_active": dx_s_max is not None,
                "dx_s_max_inf": None if dx_s_max is None else float(np.max(np.abs(dx_s_max))),
                "stage2_u_smooth_source": target_diag.get("stage2_u_smooth_source"),
                "governor_probe_available": target_result.contraction_probe_success is not None,
                "governor_probe_success": target_result.contraction_probe_success,
                "governor_probe_margin_good": target_result.contraction_probe_margin_good,
                "governor_probe_margin": target_result.contraction_probe_margin,
                "governor_probe_min_value": target_result.contraction_probe_min_value,
                "governor_probe_bound": target_result.contraction_probe_bound,
                "governor_probe_status": target_result.status,
                **_target_classification(target_result.target_error_inf, target_config),
                "command_move_inf": target_result.target_rate_y_inf,
                "input_headroom_frac": target_config.input_headroom_frac,
                "y_current_scaled": y_current_scaled.copy(),
                **disturbance_scalars,
                "xhat_next_openloop": xhat_next_openloop.copy(),
                "observer_correction": observer_correction.copy(),
                "xhat_next": xhatdhat[:, step_idx + 1].copy(),
                "reward": float(reward),
                "reward_base": float(reward),
                "reward_no_penalty": float(reward),
                "reward_augmented": float(reward),
                "delta_y": delta_y.copy(),
                "y_minus_y_sp": delta_y.copy(),
                "y_minus_y_s": None if target_result.y_s is None else y_current_scaled - target_result.y_s,
                "y_minus_y_target": None if delta_y_target is None else delta_y_target.copy(),
                "delta_u": delta_u.copy(),
                "target_info": target_info,
                "gart_rl_observation": obs.copy(),
                "slack_lyap": float(step_info.get("slack_lyap", 0.0) or 0.0),
            }
        )
        direct_info_storage.append(step_info)
        if step_idx in sub_changes:
            start = max(0, step_idx - int(time_in_sub_episodes) + 1)
            avg_rewards.append(float(np.mean(rewards[start : step_idx + 1])))
            last = direct_info_storage[-1]
            print(
                "Sub_Episode:", sub_changes[step_idx],
                "| avg. reward:", avg_rewards[-1],
                "| target_mode:", "gart",
                "| lyapunov_mode:", lyapunov_mode,
                "| plant_mode:", mode,
                "| success:", last.get("success"),
                "| target_stage:", last.get("target_stage"),
                "| contraction_margin:", last.get("contraction_margin"),
                "| slack_lyap:", last.get("slack_lyap"),
                "| nit:", last.get("solver_nit"),
            )

    return {
        "case_name": case_name,
        "y_system": y_mpc,
        "u_applied_phys": u_applied_phys,
        "avg_rewards": avg_rewards,
        "rewards": rewards,
        "xhatdhat": xhatdhat,
        "nFE": int(nFE),
        "time_in_sub_episodes": int(time_in_sub_episodes),
        "y_sp": np.asarray(y_sp, float).copy(),
        "yhat": yhat,
        "delta_y_storage": delta_y_storage,
        "delta_u_storage": delta_u_storage,
        "direct_info_storage": direct_info_storage,
        "target_info_storage": target_info_storage,
        "qi": np.asarray(qi, float).copy(),
        "qs": np.asarray(qs, float).copy(),
        "ha": np.asarray(ha, float).copy(),
        "target_mode": "gart",
        "lyapunov_mode": lyapunov_mode,
        "plant_mode": mode,
        "disturbance_after_step": False,
        "use_target_output_for_tracking": False,
        "nominal_qi": float(setup["nominal_qi"]),
        "nominal_qs": float(setup["nominal_qs"]),
        "nominal_ha": float(setup["nominal_ha"]),
        "final_qi": float(qi[-1]) if len(qi) else float(setup["nominal_qi"]),
        "final_qs": float(qs[-1]) if len(qs) else float(setup["nominal_qs"]),
        "final_ha": float(ha[-1]) if len(ha) else float(setup["nominal_ha"]),
        "rho_lyap": RHO_LYAP,
        "lyap_eps": LYAP_EPS,
        "slack_penalty": SLACK_PENALTY,
        "first_step_contraction_on": True,
        "u_dev_min": ctx["u_dev_min"].copy(),
        "u_dev_max": ctx["u_dev_max"].copy(),
        "delta_t": float(setup["delta_t"]),
        "target_config": asdict(target_config),
        "mpc_config": asdict(mpc_config),
    }


def _controller_metrics(payload: dict[str, Any], ctx: dict[str, Any], *, case_name: str) -> dict[str, Any]:
    data_min = ctx["system_data"]["data_min"]
    data_max = ctx["system_data"]["data_max"]
    n_inputs = int(ctx["lmpc_obj"].B.shape[1])
    y_ss_scaled = apply_min_max(ctx["setup"]["steady_states"]["y_ss"], data_min[n_inputs:], data_max[n_inputs:])
    nFE = int(payload["nFE"])
    y_scaled = apply_min_max(payload["y_system"][1 : nFE + 1, :], data_min[n_inputs:], data_max[n_inputs:]) - y_ss_scaled
    y_sp = np.asarray(payload["y_sp"], dtype=float)[:nFE, :]
    raw_err = y_scaled - y_sp
    y_s_rows = []
    target_err = []
    y_minus_ys = []
    governor = []
    holds = []
    target_success = []
    target_solve_success = []
    target_accepted = []
    target_usable = []
    contraction_probe = []
    slack = []
    solver_success = []
    dx_s_inf = []
    dc_rate_inf = []
    d_raw_gap_inf = []
    d_rate_max_effective_inf = []
    dx_s_max_active = []
    for row in payload.get("direct_info_storage", []):
        y_s = row.get("y_s")
        if y_s is not None:
            y_s_arr = np.asarray(y_s, dtype=float).reshape(y_sp.shape[1])
            y_s_rows.append(y_s_arr)
            target_err.append(float(np.max(np.abs(y_s_arr - np.asarray(row.get("y_sp", np.zeros_like(y_s_arr)), dtype=float).reshape(-1)))))
            y_current = row.get("y_current_scaled")
            if y_current is not None:
                y_minus_ys.append(float(np.max(np.abs(np.asarray(y_current, dtype=float).reshape(-1) - y_s_arr))))
        governor.append(1.0 if bool(row.get("governor_active", False)) else 0.0)
        holds.append(1.0 if bool(row.get("hold_previous", False)) else 0.0)
        target_success.append(1.0 if bool(row.get("target_success", False)) else 0.0)
        target_solve_success.append(1.0 if bool(row.get("target_solve_success", row.get("target_success", False))) else 0.0)
        target_accepted.append(1.0 if bool(row.get("target_accepted", row.get("target_success", False))) else 0.0)
        target_usable.append(1.0 if bool(row.get("target_usable_for_lmpc", row.get("target_success", False))) else 0.0)
        contraction_probe.append(1.0 if bool(row.get("contraction_probe_success", row.get("governor_probe_success", False))) else 0.0)
        slack.append(float(row.get("slack_lyap", 0.0) or 0.0))
        solver_success.append(1.0 if bool(row.get("success", False)) else 0.0)
        if row.get("dx_s_inf") is not None:
            dx_s_inf.append(float(row.get("dx_s_inf")))
        if row.get("dc_rate_inf") is not None:
            dc_rate_inf.append(float(row.get("dc_rate_inf")))
        if row.get("d_raw_gap_inf") is not None:
            d_raw_gap_inf.append(float(row.get("d_raw_gap_inf")))
        if row.get("d_rate_max_effective_inf") is not None:
            d_rate_max_effective_inf.append(float(row.get("d_rate_max_effective_inf")))
        dx_s_max_active.append(1.0 if bool(row.get("dx_s_max_active", False)) else 0.0)
    du = np.asarray(payload.get("delta_u_storage", []), dtype=float)
    return {
        "case_name": case_name,
        "plant_mode": payload.get("plant_mode"),
        "n_steps": nFE,
        "reward_mean": float(np.mean(payload["rewards"])) if nFE else None,
        "reward_no_penalty_mean": float(np.mean(payload["rewards"])) if nFE else None,
        "output_rmse_raw_ysp": float(np.sqrt(np.mean(raw_err**2))) if raw_err.size else None,
        "output_rmse_to_ys": None if not y_minus_ys else float(np.sqrt(np.mean(np.asarray(y_minus_ys) ** 2))),
        "mean_target_error_inf": None if not target_err else float(np.mean(target_err)),
        "p95_target_error_inf": None if not target_err else float(np.quantile(target_err, 0.95)),
        "max_target_error_inf": None if not target_err else float(np.max(target_err)),
        "solver_success_rate": float(np.mean(solver_success)) if solver_success else None,
        "target_success_rate": float(np.mean(target_success)) if target_success else None,
        "target_solve_success_rate": float(np.mean(target_solve_success)) if target_solve_success else None,
        "target_accepted_rate": float(np.mean(target_accepted)) if target_accepted else None,
        "target_usable_rate": float(np.mean(target_usable)) if target_usable else None,
        "contraction_satisfied_rate": float(np.mean(contraction_probe)) if contraction_probe else None,
        "mean_slack_lyap": float(np.mean(slack)) if slack else None,
        "p95_slack_lyap": float(np.quantile(slack, 0.95)) if slack else None,
        "mean_abs_delta_u": None if du.size == 0 else float(np.mean(np.abs(du))),
        "governor_active_rate": float(np.mean(governor)) if governor else None,
        "hold_previous_rate": float(np.mean(holds)) if holds else None,
        "unreachable_rate": None if not target_err else float(np.mean(np.asarray(target_err) > 0.5)),
        "mean_dx_s_inf": None if not dx_s_inf else float(np.mean(dx_s_inf)),
        "max_dx_s_inf": None if not dx_s_inf else float(np.max(dx_s_inf)),
        "mean_dc_rate_inf": None if not dc_rate_inf else float(np.mean(dc_rate_inf)),
        "max_dc_rate_inf": None if not dc_rate_inf else float(np.max(dc_rate_inf)),
        "mean_d_raw_gap_inf": None if not d_raw_gap_inf else float(np.mean(d_raw_gap_inf)),
        "max_d_raw_gap_inf": None if not d_raw_gap_inf else float(np.max(d_raw_gap_inf)),
        "mean_d_rate_max_effective_inf": None if not d_rate_max_effective_inf else float(np.mean(d_rate_max_effective_inf)),
        "dx_s_max_active_rate": float(np.mean(dx_s_max_active)) if dx_s_max_active else None,
    }


def _save_case_payload(case_dir: Path, payload: dict[str, Any]) -> None:
    case_dir.mkdir(parents=True, exist_ok=True)
    with (case_dir / "payload.pickle").open("wb") as f:
        pickle.dump(payload, f)
    _write_csv(case_dir / "steps.csv", payload.get("direct_info_storage", []))
    _write_json(case_dir / "config.json", {key: value for key, value in payload.items() if key.endswith("config") or key in {"case_name", "plant_mode", "lyapunov_mode", "target_mode"}})
    np.savez_compressed(
        case_dir / "arrays.npz",
        y_system=np.asarray(payload["y_system"], dtype=float),
        u_applied_phys=np.asarray(payload["u_applied_phys"], dtype=float),
        y_sp=np.asarray(payload["y_sp"], dtype=float),
        rewards=np.asarray(payload["rewards"], dtype=float),
        xhatdhat=np.asarray(payload["xhatdhat"], dtype=float),
    )


def _build_direct_style_bundle(case_name: str, payload: dict[str, Any], ctx: dict[str, Any]) -> dict[str, Any]:
    config = {
        "case_name": case_name,
        "controller_mode": payload.get("method", "gart_lmpc" if payload.get("target_mode") == "gart" else "direct_lyapunov_mpc"),
        "target_mode": payload.get("target_mode"),
        "lyapunov_mode": payload.get("lyapunov_mode"),
        "plant_mode": payload.get("plant_mode"),
        "disturbance_after_step": payload.get("disturbance_after_step"),
        "use_target_output_for_tracking": payload.get("use_target_output_for_tracking", False),
        "predict_h": PREDICT_H,
        "cont_h": CONT_H,
        "rho_lyap": RHO_LYAP,
        "lyap_eps": LYAP_EPS,
        "slack_penalty": SLACK_PENALTY,
        "setpoint_y_phys": DIRECT_TWO_SETPOINT_Y_PHYS.tolist(),
    }
    if payload.get("target_config") is not None:
        config["target_config"] = payload.get("target_config")
    if payload.get("mpc_config") is not None:
        config["mpc_config"] = payload.get("mpc_config")
    return build_direct_lyapunov_run_bundle(
        source=case_name,
        results=payload,
        steady_states=ctx["setup"]["steady_states"],
        config=config,
        data_min=ctx["system_data"]["data_min"],
        data_max=ctx["system_data"]["data_max"],
        extra={
            "reward_config": ctx["reward_config"],
            "min_max_dict": ctx["system_data"].get("min_max_dict", {}),
            "gart_discovered": ctx.get("discovered", {}),
        },
    )


def _save_case_direct_artifacts(case_dir: Path, case_name: str, payload: dict[str, Any], ctx: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
    try:
        bundle = _build_direct_style_bundle(case_name, payload, ctx)
        with (case_dir / "direct_style_bundle.pickle").open("wb") as f:
            pickle.dump(bundle, f)
        debug_dir = save_direct_lyapunov_debug_artifacts(
            bundle,
            directory=case_dir,
            prefix_name="direct_style",
            save_plots=True,
            timestamp_subdir=False,
        )
        return bundle, debug_dir
    except Exception as exc:
        _write_json(case_dir / "direct_style_artifact_error.json", {"error": repr(exc)})
        return None, None


def run_closed_loop(
    ctx: dict[str, Any],
    output_dir: Path,
    *,
    mode: str,
    n_tests: int,
    set_points_len: int,
    guard: ResourceGuard | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    case_name = FINAL_GART_CASE_NAME
    print(f"[GART] running {case_name} ({mode}, n_tests={n_tests}, set_points_len={set_points_len})")
    payload = run_gart_closed_loop_case(
        ctx,
        case_name=case_name,
        mpc_objective=FINAL_GART_MPC_OBJECTIVE,
        lyapunov_mode=FINAL_GART_LYAPUNOV_MODE,
        mode=mode,
        n_tests=n_tests,
        set_points_len=set_points_len,
        target_overrides=FINAL_GART_TARGET_OVERRIDES,
        mpc_overrides=None,
        guard=guard,
    )
    case_dir = output_dir / case_name
    _save_case_payload(case_dir, payload)
    bundle, debug_dir = _save_case_direct_artifacts(case_dir, case_name, payload, ctx)
    gart_metrics = _controller_metrics(payload, ctx, case_name=case_name)
    if bundle is None:
        record = gart_metrics
    else:
        record = make_direct_lyapunov_comparison_record(case_name, bundle, debug_dir)
        for key, value in gart_metrics.items():
            record.setdefault(key, value)
    records: list[dict[str, Any]] = [record]
    artifacts: dict[str, Any] = {
        case_name: {
            "case_dir": str(case_dir.relative_to(REPO_ROOT)),
            "direct_style_debug_dir": None if debug_dir is None else str(Path(debug_dir).relative_to(REPO_ROOT)),
            "tracking_plot_dir": None if debug_dir is None else str((Path(debug_dir) / "plots").relative_to(REPO_ROOT)),
        }
    }
    _write_csv(output_dir / "comparison.csv", records)
    summary = {
        "status": "completed",
        "plant_mode": mode,
        "n_tests": int(n_tests),
        "set_points_len": int(set_points_len),
        "disturbance_after_step": False,
        "case_name": case_name,
        "objective": FINAL_GART_MPC_OBJECTIVE,
        "lyapunov_mode": FINAL_GART_LYAPUNOV_MODE,
        "target_overrides": FINAL_GART_TARGET_OVERRIDES,
        "records": records,
        "artifacts": artifacts,
    }
    _write_json(output_dir / "summary.json", summary)
    _make_closed_loop_plots(output_dir / "plots", output_dir, records)
    return summary


def _make_closed_loop_plots(plot_dir: Path, run_dir: Path, records: list[dict[str, Any]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    plot_dir.mkdir(parents=True, exist_ok=True)
    names = [row["case_name"] for row in records]
    rmse = [row.get("output_rmse_mean", row.get("output_rmse_raw_ysp", np.nan)) for row in records]
    target = [
        row.get("target_reference_error_inf_mean", row.get("mean_target_error_inf", np.nan))
        for row in records
    ]
    x = np.arange(len(names))
    plt.figure(figsize=(9, 4))
    plt.bar(x - 0.18, rmse, width=0.36, label="physical output RMSE")
    plt.bar(x + 0.18, target, width=0.36, label="mean |y_s-y_sp|_inf")
    plt.xticks(x, names, rotation=20, ha="right")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_dir / "comparison_tracking_target_error.png", dpi=180)
    plt.close()
