from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Lyapunov.direct_lyapunov_mpc import (
    design_direct_lyapunov_mpc_solver,
    run_direct_output_disturbance_lyapunov_mpc,
)
from Lyapunov.gart_lmpc import solve_gart_lmpc_step
from Lyapunov.gart_target import GARTTargetState, jsonable, select_gart_target
from Simulation.mpc import MpcSolver, compute_observer_gain
from Simulation.run_mpc_lyapunov import _reset_system_on_entry, _set_system_input_phys, _system_io_phys
from Simulation.system_functions import PolymerCSTR
from TD3Agent.reward_functions import make_reward_fn_relative_QR
from utils.direct_lyapunov_study import governed_reference_case_spec
from utils.gart_defaults import (
    discover_gart_case_values,
    gart_rl_observation,
    make_gart_mpc_config,
    make_gart_target_config,
)
from utils.helpers import generate_setpoints_training_rl_gradually
from utils.path_helpers import repo_path
from utils.polymer_td3_defaults import DEFAULT_DIRECT_SETPOINT_Y_PHYS, DEFAULT_U_MAX_PHYS, DEFAULT_U_MIN_PHYS
from utils.scaling_helpers import apply_min_max, reverse_min_max
from utils.td3_helpers import load_and_prepare_system_data
from utils.lyapunov_utils import get_y_sp_step


PREDICT_H = 9
CONT_H = 3
RHO_LYAP = 0.99
LYAP_EPS = 1.0e-3
SLACK_PENALTY = 1.0e6
QY_DIAG = np.array([5.0, 1.0], dtype=float)
SU_DIAG = np.array([1.0, 1.0], dtype=float)
RDU_DIAG = np.array([1.0, 1.0], dtype=float)


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


def _build_context() -> dict[str, Any]:
    setup = _build_polymer_setup()
    steady_states = setup["steady_states"]
    system_data = load_and_prepare_system_data(
        steady_states=steady_states,
        setpoint_y=DEFAULT_DIRECT_SETPOINT_Y_PHYS.copy(),
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

    y_sp_scenario = apply_min_max(DEFAULT_DIRECT_SETPOINT_Y_PHYS, data_min[n_inputs:], data_max[n_inputs:]) - apply_min_max(
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
    of_mpc_obj = MpcSolver(A_aug, B_aug, C_aug, Q_out=QY_DIAG, R_in=RDU_DIAG, NP=PREDICT_H, NC=CONT_H)

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
    discovered = discover_gart_case_values(system_data, setup)
    return {
        "setup": setup,
        "system_data": system_data,
        "lmpc_obj": lmpc_obj,
        "of_mpc_obj": of_mpc_obj,
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
    test_cycle = [False] * int(n_tests)
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


def run_target_only(ctx: dict[str, Any], output_dir: Path, *, n_tests: int, set_points_len: int) -> dict[str, Any]:
    y_sp, nFE, *_ = _setpoint_schedule(ctx, n_tests=n_tests, set_points_len=set_points_len)
    system_data = ctx["system_data"]
    lmpc_obj = ctx["lmpc_obj"]
    target_config = make_gart_target_config(ctx["discovered"])
    target_state: GARTTargetState | None = None
    xhat_aug = np.zeros(lmpc_obj.A.shape[0], dtype=float)
    records: list[dict[str, Any]] = []
    y_s_store = []
    r_cmd_store = []
    d_cert_store = []
    margin_store = []

    for step_idx in range(int(nFE)):
        y_sp_k = get_y_sp_step(y_sp, step_idx, lmpc_obj.C.shape[0])
        result, target_state = select_gart_target(
            lmpc_obj.A,
            lmpc_obj.B,
            lmpc_obj.C,
            xhat_aug,
            y_sp_k,
            ctx["u_dev_min"],
            ctx["u_dev_max"],
            state=target_state,
            config=target_config,
            P_x=lmpc_obj.P_x,
            K_x=lmpc_obj.K_x,
            innovation=None,
        )
        if result.success and result.x_s is not None:
            xhat_aug[: result.x_s.size] = result.x_s
            if result.d_cert is not None:
                xhat_aug[result.x_s.size :] = result.d_cert
        row = {
            "step": step_idx,
            "target_success": result.success,
            "target_status": result.status,
            "target_stage": result.stage,
            "target_error_inf": result.target_error_inf,
            "target_rate_y_inf": result.target_rate_y_inf,
            "target_rate_u_inf": result.target_rate_u_inf,
            "target_rate_x_inf": result.target_rate_x_inf,
            "d_cert_delta_inf": result.diagnostics.get("disturbance", {}).get("d_cert_delta_inf"),
            "input_headroom_min": result.input_headroom_min,
            "contraction_probe_success": result.contraction_probe_success,
            "contraction_probe_margin": result.contraction_probe_margin,
            "governor_alpha": result.governor_alpha,
            "governor_active": result.governor_active,
            "hold_previous": result.hold_previous,
            "classified_unreachable": bool(result.target_error_inf is not None and result.target_error_inf > 1.0e-6),
        }
        records.append(row)
        y_s_store.append(np.full(lmpc_obj.C.shape[0], np.nan) if result.y_s is None else result.y_s)
        r_cmd_store.append(np.full(lmpc_obj.C.shape[0], np.nan) if result.r_cmd is None else result.r_cmd)
        d_cert_store.append(np.full(lmpc_obj.C.shape[0], np.nan) if result.d_cert is None else result.d_cert)
        margin_store.append(np.nan if result.contraction_probe_margin is None else result.contraction_probe_margin)

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "target_only_steps.csv", records)
    np.savez_compressed(
        output_dir / "target_only_arrays.npz",
        y_sp=np.asarray(y_sp, dtype=float),
        y_s=np.asarray(y_s_store, dtype=float),
        r_cmd=np.asarray(r_cmd_store, dtype=float),
        d_cert=np.asarray(d_cert_store, dtype=float),
        contraction_probe_margin=np.asarray(margin_store, dtype=float),
    )
    summary = _summarize_step_records(records, case_name="target_only")
    _write_json(output_dir / "target_only_summary.json", summary)
    _write_json(output_dir / "target_only_config.json", {"target_config": asdict(target_config), "discovered": ctx["discovered"]})
    _make_target_plots(output_dir / "plots", np.asarray(y_sp, dtype=float), np.asarray(y_s_store, dtype=float), np.asarray(d_cert_store, dtype=float), np.asarray(margin_store, dtype=float), records)
    return summary


def run_gart_closed_loop_case(
    ctx: dict[str, Any],
    *,
    case_name: str,
    mpc_objective: str,
    lyapunov_mode: str,
    mode: str,
    n_tests: int,
    set_points_len: int,
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
    target_config = make_gart_target_config(ctx["discovered"])
    mpc_config = make_gart_mpc_config(ctx["discovered"], objective=mpc_objective, lyapunov_mode=lyapunov_mode)

    for step_idx in range(int(nFE)):
        x0_aug = xhatdhat[:, step_idx].copy()
        scaled_current_input = apply_min_max(system.current_input, data_min[:n_inputs], data_max[:n_inputs])
        u_prev_dev = scaled_current_input - ss_scaled_inputs
        y_sp_k = get_y_sp_step(y_sp, step_idx, n_outputs)
        y_prev_scaled = apply_min_max(y_mpc[step_idx, :], data_min[n_inputs:], data_max[n_inputs:]) - y_ss_scaled
        yhat_now = np.asarray(lmpc_obj.C @ x0_aug, dtype=float).reshape(-1)
        innovation = y_prev_scaled - yhat_now
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
        )
        target_info = target_result.to_dict()
        target_info.update({"step": step_idx, "target_mode": "gart"})
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
                "y_current_scaled": y_current_scaled.copy(),
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


def _run_old_governed_reference(ctx: dict[str, Any], *, mode: str, n_tests: int, set_points_len: int) -> dict[str, Any]:
    cfg = governed_reference_case_spec(QY_DIAG, case_name="old_governed_reference", lyapunov_mode="hard")
    setup = ctx["setup"]
    return run_direct_output_disturbance_lyapunov_mpc(
        system=_make_system(setup),
        LMPC_obj=ctx["lmpc_obj"],
        y_sp_scenario=ctx["y_sp_scenario"],
        n_tests=n_tests,
        set_points_len=set_points_len,
        steady_states=setup["steady_states"],
        IC_opt=ctx["ic_opt"].copy(),
        bnds=ctx["bnds"],
        L=ctx["observer_gain"],
        data_min=ctx["system_data"]["data_min"],
        data_max=ctx["system_data"]["data_max"],
        test_cycle=[False] * int(n_tests),
        reward_fn=ctx["reward_fn"],
        nominal_qi=setup["nominal_qi"],
        nominal_qs=setup["nominal_qs"],
        nominal_ha=setup["nominal_ha"],
        qi_change=setup["qi_change"],
        qs_change=setup["qs_change"],
        ha_change=setup["ha_change"],
        target_mode=cfg["target_mode"],
        lyapunov_mode=cfg["lyapunov_mode"],
        target_config=cfg["target_config"],
        mode=mode,
        disturbance_after_step=False,
        use_target_output_for_tracking=False,
        skip_terminal_if_alpha_small=True,
        alpha_terminal_min=1.0e-8,
        use_target_on_solver_fail=False,
        rho_lyap=RHO_LYAP,
        lyap_eps=LYAP_EPS,
        slack_penalty=SLACK_PENALTY,
        first_step_contraction_on=True,
        force_final_test=True,
    )


def _summarize_step_records(records: list[dict[str, Any]], *, case_name: str) -> dict[str, Any]:
    if not records:
        return {"case_name": case_name, "n_steps": 0}
    def mean_bool(key: str) -> float:
        vals = [1.0 if bool(row.get(key, False)) else 0.0 for row in records]
        return float(np.mean(vals)) if vals else float("nan")
    def nanmean(key: str) -> float | None:
        vals = np.array([np.nan if row.get(key) is None else float(row.get(key)) for row in records], dtype=float)
        return None if np.all(~np.isfinite(vals)) else float(np.nanmean(vals))
    def nanp95(key: str) -> float | None:
        vals = np.array([np.nan if row.get(key) is None else float(row.get(key)) for row in records], dtype=float)
        vals = vals[np.isfinite(vals)]
        return None if vals.size == 0 else float(np.quantile(vals, 0.95))
    return {
        "case_name": case_name,
        "n_steps": len(records),
        "target_success_rate": mean_bool("target_success"),
        "solver_success_rate": mean_bool("success"),
        "mean_target_error_inf": nanmean("target_error_inf"),
        "p95_target_error_inf": nanp95("target_error_inf"),
        "mean_contraction_probe_margin": nanmean("contraction_probe_margin"),
        "governor_active_rate": mean_bool("governor_active"),
        "hold_previous_rate": mean_bool("hold_previous"),
        "unreachable_rate": mean_bool("classified_unreachable"),
        "mean_slack_lyap": nanmean("slack_lyap"),
        "p95_slack_lyap": nanp95("slack_lyap"),
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
    contraction_probe = []
    slack = []
    solver_success = []
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
        contraction_probe.append(1.0 if bool(row.get("contraction_probe_success", row.get("governor_probe_success", False))) else 0.0)
        slack.append(float(row.get("slack_lyap", 0.0) or 0.0))
        solver_success.append(1.0 if bool(row.get("success", False)) else 0.0)
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
        "contraction_satisfied_rate": float(np.mean(contraction_probe)) if contraction_probe else None,
        "mean_slack_lyap": float(np.mean(slack)) if slack else None,
        "p95_slack_lyap": float(np.quantile(slack, 0.95)) if slack else None,
        "mean_abs_delta_u": None if du.size == 0 else float(np.mean(np.abs(du))),
        "governor_active_rate": float(np.mean(governor)) if governor else None,
        "hold_previous_rate": float(np.mean(holds)) if holds else None,
        "unreachable_rate": None if not target_err else float(np.mean(np.asarray(target_err) > 1.0e-6)),
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


def _stack_step_vectors(rows: list[dict[str, Any]], key: str, size: int) -> np.ndarray | None:
    values = []
    found = False
    for row in rows:
        value = row.get(key)
        if value is None:
            values.append(np.full(size, np.nan, dtype=float))
        else:
            found = True
            values.append(np.asarray(value, dtype=float).reshape(size))
    if not values or not found:
        return None
    return np.asarray(values, dtype=float)


def _case_target_plot_arrays(payload: dict[str, Any], ctx: dict[str, Any]) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    rows = list(payload.get("direct_info_storage", []))
    n_outputs = int(ctx["lmpc_obj"].C.shape[0])
    n_inputs = int(ctx["lmpc_obj"].B.shape[1])
    y_target = _stack_step_vectors(rows, "y_s", n_outputs)
    y_tracking = _stack_step_vectors(rows, "y_target", n_outputs)
    u_s_dev = _stack_step_vectors(rows, "u_s", n_inputs)
    if u_s_dev is None:
        return y_target, y_tracking, None
    data_min = ctx["system_data"]["data_min"]
    data_max = ctx["system_data"]["data_max"]
    ss_scaled_inputs = apply_min_max(ctx["setup"]["steady_states"]["ss_inputs"], data_min[:n_inputs], data_max[:n_inputs])
    u_target_phys = reverse_min_max(u_s_dev + ss_scaled_inputs.reshape(1, -1), data_min[:n_inputs], data_max[:n_inputs])
    return y_target, y_tracking, u_target_phys


def _save_case_tracking_plots(case_dir: Path, payload: dict[str, Any], ctx: dict[str, Any]) -> str | None:
    y_target, y_tracking, u_target_phys = _case_target_plot_arrays(payload, ctx)
    plot_dir = case_dir / "tracking_plots"
    try:
        from Plotting_fns.mpc_plot_fns import plot_mpc_results_cstr

        return plot_mpc_results_cstr(
            y_sp=payload["y_sp"],
            steady_states=ctx["setup"]["steady_states"],
            nFE=int(payload["nFE"]),
            delta_t=float(payload.get("delta_t", ctx["setup"]["delta_t"])),
            time_in_sub_episodes=int(payload["time_in_sub_episodes"]),
            y_mpc=payload["y_system"],
            u_mpc=payload["u_applied_phys"],
            data_min=ctx["system_data"]["data_min"],
            data_max=ctx["system_data"]["data_max"],
            directory=plot_dir,
            prefix_name="",
            y_target=y_target,
            y_tracking_target=y_tracking,
            u_target=u_target_phys,
            u_bounds=(DEFAULT_U_MIN_PHYS, DEFAULT_U_MAX_PHYS),
            timestamp_subdir=False,
            paper_style=True,
            output_labels=("eta", "T"),
            input_labels=("Qc", "Qm"),
        )
    except Exception as exc:
        _write_json(case_dir / "tracking_plot_error.json", {"error": repr(exc)})
        return None


def run_closed_loop(ctx: dict[str, Any], output_dir: Path, *, mode: str, n_tests: int, set_points_len: int) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    cases = [
        ("old_governed_reference", None, None),
        ("gart_target_raw_objective", "raw", "hard"),
        ("gart_target_mixed_objective", "mixed", "hard"),
        ("gart_target_mixed_soft", "mixed", "soft"),
    ]
    records: list[dict[str, Any]] = []
    artifacts: dict[str, Any] = {}
    for case_name, objective, lyap_mode in cases:
        print(f"[GART] running {case_name} ({mode}, n_tests={n_tests}, set_points_len={set_points_len})")
        if case_name == "old_governed_reference":
            payload = _run_old_governed_reference(ctx, mode=mode, n_tests=n_tests, set_points_len=set_points_len)
            payload["case_name"] = case_name
        else:
            payload = run_gart_closed_loop_case(
                ctx,
                case_name=case_name,
                mpc_objective=str(objective),
                lyapunov_mode=str(lyap_mode),
                mode=mode,
                n_tests=n_tests,
                set_points_len=set_points_len,
            )
        case_dir = output_dir / case_name
        _save_case_payload(case_dir, payload)
        tracking_plot_dir = _save_case_tracking_plots(case_dir, payload, ctx)
        records.append(_controller_metrics(payload, ctx, case_name=case_name))
        artifacts[case_name] = {
            "case_dir": str(case_dir.relative_to(REPO_ROOT)),
            "tracking_plot_dir": None if tracking_plot_dir is None else str(Path(tracking_plot_dir).relative_to(REPO_ROOT)),
        }
    _write_csv(output_dir / "comparison.csv", records)
    summary = {
        "status": "completed",
        "plant_mode": mode,
        "n_tests": int(n_tests),
        "set_points_len": int(set_points_len),
        "disturbance_after_step": False,
        "records": records,
        "artifacts": artifacts,
    }
    _write_json(output_dir / "summary.json", summary)
    _make_closed_loop_plots(output_dir / "plots", output_dir, records)
    return summary


def _make_target_plots(plot_dir: Path, y_sp: np.ndarray, y_s: np.ndarray, d_cert: np.ndarray, margins: np.ndarray, records: list[dict[str, Any]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    plot_dir.mkdir(parents=True, exist_ok=True)
    steps = np.arange(y_s.shape[0])
    plt.figure(figsize=(8, 4))
    for idx in range(y_sp.shape[1]):
        plt.plot(steps, y_sp[: y_s.shape[0], idx], "--", label=f"y_sp[{idx}]")
        plt.plot(steps, y_s[:, idx], label=f"y_s[{idx}]")
    plt.legend()
    plt.xlabel("step")
    plt.ylabel("scaled deviation")
    plt.tight_layout()
    plt.savefig(plot_dir / "target_vs_setpoint.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8, 3))
    plt.plot(steps, margins)
    plt.axhline(0.0, color="k", linewidth=0.8)
    plt.xlabel("step")
    plt.ylabel("probe margin")
    plt.tight_layout()
    plt.savefig(plot_dir / "contraction_probe_margin.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8, 3))
    plt.plot(steps, [row.get("governor_alpha", np.nan) for row in records])
    plt.xlabel("step")
    plt.ylabel("governor alpha")
    plt.tight_layout()
    plt.savefig(plot_dir / "governor_alpha.png", dpi=180)
    plt.close()


def _make_closed_loop_plots(plot_dir: Path, run_dir: Path, records: list[dict[str, Any]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    plot_dir.mkdir(parents=True, exist_ok=True)
    names = [row["case_name"] for row in records]
    rmse = [row.get("output_rmse_raw_ysp", np.nan) for row in records]
    target = [np.nan if row.get("mean_target_error_inf") is None else row.get("mean_target_error_inf") for row in records]
    x = np.arange(len(names))
    plt.figure(figsize=(9, 4))
    plt.bar(x - 0.18, rmse, width=0.36, label="RMSE to y_sp")
    plt.bar(x + 0.18, target, width=0.36, label="mean |y_s-y_sp|_inf")
    plt.xticks(x, names, rotation=20, ha="right")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_dir / "comparison_tracking_target_error.png", dpi=180)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run GART target-selector and GART-LMPC smoke studies.")
    parser.add_argument("--mode", choices=["nominal", "disturb"], default="nominal")
    parser.add_argument("--n-tests", type=int, default=5)
    parser.add_argument("--set-points-len", type=int, default=20)
    parser.add_argument("--target-only", action="store_true")
    parser.add_argument("--closed-loop", action="store_true")
    parser.add_argument("--timestamp", default=None)
    return parser.parse_args()


def main() -> dict[str, Any]:
    args = parse_args()
    if not args.target_only and not args.closed_loop:
        args.target_only = True
        args.closed_loop = True
    timestamp = args.timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    ctx = _build_context()
    root = Path(repo_path())
    summaries: dict[str, Any] = {
        "timestamp": timestamp,
        "mode": args.mode,
        "n_tests": int(args.n_tests),
        "set_points_len": int(args.set_points_len),
        "disturbance_after_step": False,
    }
    if args.target_only:
        target_dir = root / "results" / "GARTTargetSelectorStudy" / timestamp
        summaries["target_only"] = run_target_only(ctx, target_dir, n_tests=args.n_tests, set_points_len=args.set_points_len)
        summaries["target_only_dir"] = str(target_dir.relative_to(root))
    if args.closed_loop:
        lmpc_dir = root / "results" / "GARTLMPC" / timestamp
        summaries["closed_loop"] = run_closed_loop(ctx, lmpc_dir, mode=args.mode, n_tests=args.n_tests, set_points_len=args.set_points_len)
        summaries["closed_loop_dir"] = str(lmpc_dir.relative_to(root))
    print(json.dumps(_jsonable(summaries), indent=2))
    return summaries


if __name__ == "__main__":
    main()
