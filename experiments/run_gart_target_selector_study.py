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
    build_direct_lyapunov_run_bundle,
    design_direct_lyapunov_mpc_solver,
    make_direct_lyapunov_comparison_record,
    run_direct_output_disturbance_lyapunov_mpc,
    save_direct_lyapunov_debug_artifacts,
)
from Lyapunov.gart_lmpc import solve_gart_lmpc_step
from Lyapunov.gart_target import GARTTargetState, jsonable, select_gart_target
from Simulation.mpc import MpcSolver, compute_observer_gain
from Simulation.run_mpc_lyapunov import _reset_system_on_entry, _set_system_input_phys, _system_io_phys
from Simulation.system_functions import PolymerCSTR
from TD3Agent.reward_functions import make_reward_fn_relative_QR
from utils.direct_lyapunov_study import (
    DIRECT_DISTURBANCE_SETPOINT_LEN,
    DIRECT_TWO_SETPOINT_Y_PHYS,
    direct_disturbance_test_cycle,
    governed_reference_case_spec,
)
from utils.gart_defaults import (
    discover_gart_case_values,
    gart_rl_observation,
    make_gart_mpc_config,
    make_gart_target_config,
)
from utils.gart_runtime import GARTStudyLimits, ResourceGuard, set_single_thread_env
from utils.helpers import generate_setpoints_training_rl_gradually
from utils.path_helpers import repo_path
from utils.polymer_td3_defaults import DEFAULT_U_MAX_PHYS, DEFAULT_U_MIN_PHYS
from utils.scaling_helpers import apply_min_max, reverse_min_max
from utils.td3_helpers import load_and_prepare_system_data
from utils.lyapunov_utils import get_y_sp_step


PREDICT_H = 9
CONT_H = 3
RHO_LYAP = 0.99
LYAP_EPS = 1.0e-4
SLACK_PENALTY = 1.0e6
QY_DIAG = np.array([5.0, 1.0], dtype=float)
SU_DIAG = np.array([1.0, 1.0], dtype=float)
RDU_DIAG = np.array([1.0, 1.0], dtype=float)


GART_RELAXED_TARGET_OVERRIDES: dict[str, Any] = {
    "disable_dx_rate": True,
    "disable_u_mid_tiebreak": True,
    "disable_x_smoothing": True,
    "disable_y_smoothing": True,
    "input_headroom_frac": 0.01,
}
GART_RELAXED_DY2_OVERRIDES: dict[str, Any] = {**GART_RELAXED_TARGET_OVERRIDES, "dy_rate_scale": 2.0}
GART_MIXED_MPC_OVERRIDES: dict[str, Any] = {
    "eta_y": 0.1,
    "eta_u": 0.1,
    "target_term_gate_enabled": False,
}


TARGET_ABLATION_CASES: list[dict[str, Any]] = [
    {"name": "T0_current", "overrides": {}},
    {"name": "T1_no_dx_rate", "overrides": {"disable_dx_rate": True}},
    {"name": "T2_no_dx_rate_headroom_0p01", "overrides": {"disable_dx_rate": True, "input_headroom_frac": 0.01}},
    {
        "name": "T3_no_dx_rate_headroom_0p01_dy2",
        "overrides": {"disable_dx_rate": True, "input_headroom_frac": 0.01, "dy_rate_scale": 2.0},
    },
    {
        "name": "T5_no_dx_rate_headroom_0p01_dy2_no_xy_smooth_no_umid",
        "overrides": GART_RELAXED_DY2_OVERRIDES,
    },
    {
        "name": "T7_no_dx_rate_headroom_0p01_dy2_no_du",
        "overrides": {
            "disable_dx_rate": True,
            "input_headroom_frac": 0.01,
            "dy_rate_scale": 2.0,
            "disable_du_rate": True,
        },
    },
    {
        "name": "T8_no_umid_probe_log_only",
        "overrides": {
            **GART_RELAXED_DY2_OVERRIDES,
            "disable_du_rate": True,
            "contraction_probe_log_only": True,
        },
    },
]


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


def _target_step_row(step_idx: int, result: Any, target_config: Any, *, mode_name: str) -> dict[str, Any]:
    diag = result.diagnostics if isinstance(result.diagnostics, dict) else {}
    classification = _target_classification(result.target_error_inf, target_config)
    return {
        "step": step_idx,
        "target_only_mode": mode_name,
        "target_success": result.success,
        "target_solve_success": result.solve_success,
        "target_accepted": result.accepted,
        "target_usable_for_lmpc": result.usable_for_lmpc,
        "target_rejection_reason": result.rejection_reason,
        "target_status": result.status,
        "target_stage": result.stage,
        "target_error_inf": result.target_error_inf,
        "target_rate_y_inf": result.target_rate_y_inf,
        "target_rate_u_inf": result.target_rate_u_inf,
        "target_rate_x_inf": result.target_rate_x_inf,
        "d_cert_delta_inf": diag.get("disturbance", {}).get("d_cert_delta_inf"),
        "input_headroom_min": result.input_headroom_min,
        "contraction_probe_success": result.contraction_probe_success,
        "contraction_probe_margin_good": result.contraction_probe_margin_good,
        "contraction_probe_margin": result.contraction_probe_margin,
        "stage1_probe_margin_good": diag.get("stage1_probe_margin_good"),
        "stage2_probe_margin_good": diag.get("stage2_probe_margin_good"),
        "stage2_minus_stage1_probe_margin_good": diag.get("stage2_minus_stage1_probe_margin_good"),
        "stage1_primary_cost": diag.get("stage1_primary_cost"),
        "stage2_primary_cost": diag.get("stage2_primary_cost"),
        "stage2_tiebreak_cost": diag.get("stage2_tiebreak_cost"),
        "stage2_u_smooth_source": diag.get("stage2_u_smooth_source"),
        "governor_alpha": result.governor_alpha,
        "governor_active": result.governor_active,
        "hold_previous": result.hold_previous,
        **classification,
    }


def _save_target_only_outputs(
    output_dir: Path,
    *,
    records: list[dict[str, Any]],
    y_sp: np.ndarray,
    y_s_store: list[Any],
    r_cmd_store: list[Any],
    d_cert_store: list[Any],
    margin_store: list[Any],
    target_config: Any,
    discovered: dict[str, Any],
    extra_arrays: dict[str, Any] | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "target_only_steps.csv", records)
    arrays = {
        "y_sp": np.asarray(y_sp, dtype=float),
        "y_s": np.asarray(y_s_store, dtype=float),
        "r_cmd": np.asarray(r_cmd_store, dtype=float),
        "d_cert": np.asarray(d_cert_store, dtype=float),
        "contraction_probe_margin": np.asarray(margin_store, dtype=float),
        "contraction_probe_margin_good": np.asarray(margin_store, dtype=float),
    }
    if extra_arrays:
        arrays.update(extra_arrays)
    np.savez_compressed(output_dir / "target_only_arrays.npz", **arrays)
    summary = _summarize_step_records(records, case_name=records[0].get("target_only_mode", "target_only") if records else "target_only")
    _write_json(output_dir / "target_only_summary.json", summary)
    _write_json(output_dir / "target_only_config.json", {"target_config": asdict(target_config), "discovered": discovered})
    _make_target_plots(
        output_dir / "plots",
        np.asarray(y_sp, dtype=float),
        np.asarray(y_s_store, dtype=float),
        np.asarray(d_cert_store, dtype=float),
        np.asarray(margin_store, dtype=float),
        records,
    )
    return summary


def run_synthetic_target_only(
    ctx: dict[str, Any],
    output_dir: Path,
    *,
    n_tests: int,
    set_points_len: int,
    target_overrides: dict[str, Any] | None = None,
    guard: ResourceGuard | None = None,
) -> dict[str, Any]:
    """Selector self-consistency mode: overwrites xhat with accepted x_s."""
    y_sp, nFE, *_ = _setpoint_schedule(ctx, n_tests=n_tests, set_points_len=set_points_len)
    system_data = ctx["system_data"]
    lmpc_obj = ctx["lmpc_obj"]
    target_config = make_gart_target_config(ctx["discovered"], **(target_overrides or {}))
    target_state: GARTTargetState | None = None
    xhat_aug = np.zeros(lmpc_obj.A.shape[0], dtype=float)
    records: list[dict[str, Any]] = []
    y_s_store = []
    r_cmd_store = []
    d_cert_store = []
    margin_store = []

    for step_idx in range(int(nFE)):
        if guard is not None:
            guard.tick_target()
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
        if result.accepted and result.x_s is not None:
            xhat_aug[: result.x_s.size] = result.x_s
            if result.d_cert is not None:
                xhat_aug[result.x_s.size :] = result.d_cert
        records.append(_target_step_row(step_idx, result, target_config, mode_name="synthetic_target_only"))
        y_s_store.append(np.full(lmpc_obj.C.shape[0], np.nan) if result.y_s is None else result.y_s)
        r_cmd_store.append(np.full(lmpc_obj.C.shape[0], np.nan) if result.r_cmd is None else result.r_cmd)
        d_cert_store.append(np.full(lmpc_obj.C.shape[0], np.nan) if result.d_cert is None else result.d_cert)
        margin_store.append(np.nan if result.contraction_probe_margin_good is None else result.contraction_probe_margin_good)

    return _save_target_only_outputs(
        output_dir,
        records=records,
        y_sp=np.asarray(y_sp, dtype=float),
        y_s_store=y_s_store,
        r_cmd_store=r_cmd_store,
        d_cert_store=d_cert_store,
        margin_store=margin_store,
        target_config=target_config,
        discovered=ctx["discovered"],
    )


def run_target_only(ctx: dict[str, Any], output_dir: Path, *, n_tests: int, set_points_len: int) -> dict[str, Any]:
    return run_synthetic_target_only(ctx, output_dir, n_tests=n_tests, set_points_len=set_points_len)


def _payload_from_npz(path: str | Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=True) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def _observer_replay_payload(
    ctx: dict[str, Any],
    *,
    replay_source: str,
    replay_path: str | None,
    mode: str,
    n_tests: int,
    set_points_len: int,
    replay_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    source = str(replay_source).strip().lower()
    if replay_payload is not None:
        return replay_payload
    if source == "explicit_npz":
        if not replay_path:
            raise ValueError("replay_path is required when replay_source='explicit_npz'.")
        return _payload_from_npz(replay_path)
    if source == "old_governed_reference":
        return _run_old_governed_reference(ctx, mode=mode, n_tests=n_tests, set_points_len=set_points_len)
    if source in {"gart_raw_objective", "gart_target_raw_objective"}:
        return run_gart_closed_loop_case(
            ctx,
            case_name="gart_target_raw_no_dx_headroom_0p01_dy2_no_umid_replay_source",
            mpc_objective="raw",
            lyapunov_mode="hard",
            mode=mode,
            n_tests=n_tests,
            set_points_len=set_points_len,
            target_overrides=GART_RELAXED_DY2_OVERRIDES,
        )
    raise ValueError("replay_source must be old_governed_reference, gart_raw_objective, or explicit_npz.")


def run_observer_replay_target_only(
    ctx: dict[str, Any],
    output_dir: Path,
    *,
    replay_source: str,
    n_tests: int,
    set_points_len: int,
    replay_path: str | None = None,
    mode: str = "nominal",
    replay_payload: dict[str, Any] | None = None,
    target_overrides: dict[str, Any] | None = None,
    guard: ResourceGuard | None = None,
) -> dict[str, Any]:
    payload = _observer_replay_payload(
        ctx,
        replay_source=replay_source,
        replay_path=replay_path,
        mode=mode,
        n_tests=n_tests,
        set_points_len=set_points_len,
        replay_payload=replay_payload,
    )
    lmpc_obj = ctx["lmpc_obj"]
    n_outputs = int(lmpc_obj.C.shape[0])
    xhatdhat = np.asarray(payload["xhatdhat"], dtype=float)
    if xhatdhat.shape[0] != int(lmpc_obj.A.shape[0]) and xhatdhat.shape[1] == int(lmpc_obj.A.shape[0]):
        xhatdhat = xhatdhat.T
    y_sp = np.asarray(payload.get("y_sp"), dtype=float)
    if y_sp.ndim != 2:
        raise ValueError("Replay payload must contain 2D y_sp.")
    n_steps = min(int(xhatdhat.shape[1] - 1), int(y_sp.shape[0]))
    target_config = make_gart_target_config(ctx["discovered"], **(target_overrides or {}))
    target_state: GARTTargetState | None = None
    records: list[dict[str, Any]] = []
    y_s_store = []
    r_cmd_store = []
    d_cert_store = []
    margin_store = []
    xhat_used = []
    data_min = ctx["system_data"]["data_min"]
    data_max = ctx["system_data"]["data_max"]
    n_inputs = int(lmpc_obj.B.shape[1])
    ss_scaled_inputs = apply_min_max(ctx["setup"]["steady_states"]["ss_inputs"], data_min[:n_inputs], data_max[:n_inputs])
    y_ss_scaled = apply_min_max(ctx["setup"]["steady_states"]["y_ss"], data_min[n_inputs:], data_max[n_inputs:])
    y_system = payload.get("y_system")
    u_applied_phys = payload.get("u_applied_phys")

    for step_idx in range(n_steps):
        if guard is not None:
            guard.tick_target()
        xhat_aug = xhatdhat[:, step_idx].copy()
        xhat_used.append(xhat_aug.copy())
        y_sp_k = get_y_sp_step(y_sp, step_idx, n_outputs)
        innovation = None
        if y_system is not None:
            y_arr = np.asarray(y_system, dtype=float)
            if y_arr.ndim == 2 and step_idx < y_arr.shape[0]:
                y_prev_scaled = apply_min_max(y_arr[step_idx, :], data_min[n_inputs:], data_max[n_inputs:]) - y_ss_scaled
                innovation = y_prev_scaled - np.asarray(lmpc_obj.C @ xhat_aug, dtype=float).reshape(-1)
        if u_applied_phys is not None and step_idx > 0:
            u_arr = np.asarray(u_applied_phys, dtype=float)
            if u_arr.ndim == 2 and step_idx - 1 < u_arr.shape[0]:
                u_smooth_ref = apply_min_max(u_arr[step_idx - 1, :], data_min[:n_inputs], data_max[:n_inputs]) - ss_scaled_inputs
            else:
                u_smooth_ref = np.zeros(n_inputs, dtype=float)
        else:
            u_smooth_ref = np.zeros(n_inputs, dtype=float)
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
            innovation=innovation,
            u_smooth_ref=u_smooth_ref,
        )
        records.append(_target_step_row(step_idx, result, target_config, mode_name="observer_replay_target_only"))
        y_s_store.append(np.full(n_outputs, np.nan) if result.y_s is None else result.y_s)
        r_cmd_store.append(np.full(n_outputs, np.nan) if result.r_cmd is None else result.r_cmd)
        d_cert_store.append(np.full(n_outputs, np.nan) if result.d_cert is None else result.d_cert)
        margin_store.append(np.nan if result.contraction_probe_margin_good is None else result.contraction_probe_margin_good)

    return _save_target_only_outputs(
        output_dir,
        records=records,
        y_sp=y_sp[:n_steps, :],
        y_s_store=y_s_store,
        r_cmd_store=r_cmd_store,
        d_cert_store=d_cert_store,
        margin_store=margin_store,
        target_config=target_config,
        discovered=ctx["discovered"],
        extra_arrays={
            "replay_xhatdhat_used": np.asarray(xhat_used, dtype=float),
            "replay_xhatdhat_original": xhatdhat[:, :n_steps].T.copy(),
        },
    )


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
                "command_move_inf": target_result.target_rate_y_inf,
                "input_headroom_frac": target_config.input_headroom_frac,
                "stage2_u_smooth_source": target_diag.get("stage2_u_smooth_source"),
                "residual_total_norm": target_result.target_error_inf,
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
        test_cycle=direct_disturbance_test_cycle(int(n_tests)),
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
        "target_solve_success_rate": mean_bool("target_solve_success"),
        "target_accepted_rate": mean_bool("target_accepted"),
        "target_usable_rate": mean_bool("target_usable_for_lmpc"),
        "solver_success_rate": mean_bool("success"),
        "mean_target_error_inf": nanmean("target_error_inf"),
        "p95_target_error_inf": nanp95("target_error_inf"),
        "mean_contraction_probe_margin_good": nanmean("contraction_probe_margin_good"),
        "mean_contraction_probe_margin": nanmean("contraction_probe_margin"),
        "contraction_probe_success_rate": mean_bool("contraction_probe_success"),
        "governor_active_rate": mean_bool("governor_active"),
        "hold_previous_rate": mean_bool("hold_previous"),
        "target_exact_rate": mean_bool("target_exact"),
        "target_good_rate": mean_bool("target_good"),
        "target_acceptable_rate": mean_bool("target_acceptable"),
        "unreachable_rate": mean_bool("target_unreachable"),
        "mean_governor_alpha": nanmean("governor_alpha"),
        "p05_input_headroom": None
        if np.all(~np.isfinite(np.array([np.nan if row.get("input_headroom_min") is None else float(row.get("input_headroom_min")) for row in records], dtype=float)))
        else float(np.nanquantile(np.array([np.nan if row.get("input_headroom_min") is None else float(row.get("input_headroom_min")) for row in records], dtype=float), 0.05)),
        "mean_terminal_alpha": nanmean("alpha_terminal"),
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
    target_solve_success = []
    target_accepted = []
    target_usable = []
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
        target_solve_success.append(1.0 if bool(row.get("target_solve_success", row.get("target_success", False))) else 0.0)
        target_accepted.append(1.0 if bool(row.get("target_accepted", row.get("target_success", False))) else 0.0)
        target_usable.append(1.0 if bool(row.get("target_usable_for_lmpc", row.get("target_success", False))) else 0.0)
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


def _normalize_case_spec(case: Any) -> dict[str, Any]:
    if isinstance(case, dict):
        return {
            "case_name": str(case["case_name"]),
            "objective": case.get("objective"),
            "lyapunov_mode": case.get("lyapunov_mode"),
            "target_overrides": case.get("target_overrides"),
            "mpc_overrides": case.get("mpc_overrides"),
        }
    if isinstance(case, (tuple, list)):
        if len(case) < 3:
            raise ValueError("Closed-loop tuple case specs must contain case_name, objective, and lyapunov_mode.")
        return {
            "case_name": str(case[0]),
            "objective": case[1],
            "lyapunov_mode": case[2],
            "target_overrides": case[3] if len(case) > 3 else None,
            "mpc_overrides": case[4] if len(case) > 4 else None,
        }
    raise TypeError(f"Unsupported closed-loop case spec type: {type(case)!r}")


def run_closed_loop(
    ctx: dict[str, Any],
    output_dir: Path,
    *,
    mode: str,
    n_tests: int,
    set_points_len: int,
    case_specs: list[Any] | None = None,
    guard: ResourceGuard | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    cases = case_specs or [
        {
            "case_name": "gart_target_raw_no_dx_headroom_0p01_dy2_no_umid",
            "objective": "raw",
            "lyapunov_mode": "hard",
            "target_overrides": GART_RELAXED_DY2_OVERRIDES,
        },
    ]
    records: list[dict[str, Any]] = []
    artifacts: dict[str, Any] = {}
    for raw_case in cases:
        case = _normalize_case_spec(raw_case)
        case_name = str(case["case_name"])
        objective = case.get("objective")
        lyap_mode = case.get("lyapunov_mode")
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
                target_overrides=case.get("target_overrides"),
                mpc_overrides=case.get("mpc_overrides"),
                guard=guard,
            )
        case_dir = output_dir / case_name
        _save_case_payload(case_dir, payload)
        bundle, debug_dir = _save_case_direct_artifacts(case_dir, case_name, payload, ctx)
        if bundle is None:
            records.append(_controller_metrics(payload, ctx, case_name=case_name))
        else:
            records.append(make_direct_lyapunov_comparison_record(case_name, bundle, debug_dir))
        artifacts[case_name] = {
            "case_dir": str(case_dir.relative_to(REPO_ROOT)),
            "direct_style_debug_dir": None if debug_dir is None else str(Path(debug_dir).relative_to(REPO_ROOT)),
            "tracking_plot_dir": None if debug_dir is None else str((Path(debug_dir) / "plots").relative_to(REPO_ROOT)),
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


def run_target_ablation_study(
    ctx: dict[str, Any],
    output_dir: Path,
    *,
    mode: str,
    n_tests: int,
    set_points_len: int,
    replay_source: str = "gart_raw_objective",
    replay_path: str | None = None,
    guard: ResourceGuard | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    artifacts: dict[str, Any] = {}
    replay_payload = _observer_replay_payload(
        ctx,
        replay_source=replay_source,
        replay_path=replay_path,
        mode=mode,
        n_tests=n_tests,
        set_points_len=set_points_len,
    )
    for case in TARGET_ABLATION_CASES:
        name = str(case["name"])
        case_dir = output_dir / name
        summary = run_observer_replay_target_only(
            ctx,
            case_dir,
            replay_source=replay_source,
            replay_path=replay_path,
            mode=mode,
            n_tests=n_tests,
            set_points_len=set_points_len,
            replay_payload=replay_payload,
            target_overrides=dict(case.get("overrides", {})),
            guard=guard,
        )
        row = dict(summary)
        row["case_name"] = name
        row["overrides"] = json.dumps(_jsonable(case.get("overrides", {})))
        records.append(row)
        artifacts[name] = {"case_dir": str(case_dir.relative_to(REPO_ROOT))}
    _write_csv(output_dir / "target_ablation_comparison.csv", records)
    summary = {
        "status": "completed",
        "mode": mode,
        "n_tests": int(n_tests),
        "set_points_len": int(set_points_len),
        "replay_source": replay_source,
        "records": records,
        "artifacts": artifacts,
    }
    _write_json(output_dir / "summary.json", summary)
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run GART target-selector and GART-LMPC smoke studies.")
    parser.add_argument("--mode", choices=["nominal", "disturb"], default="nominal")
    parser.add_argument("--n-tests", type=int, default=1)
    parser.add_argument("--set-points-len", type=int, default=20)
    parser.add_argument("--target-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--closed-loop", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--target-only-mode", choices=["synthetic", "observer-replay"], default="synthetic")
    parser.add_argument("--replay-source", choices=["old_governed_reference", "gart_raw_objective", "explicit_npz"], default="gart_raw_objective")
    parser.add_argument("--replay-path", default=None)
    parser.add_argument("--target-ablation", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--confirm-full", action="store_true")
    parser.add_argument("--max-target-evals", type=int, default=None)
    parser.add_argument("--max-closed-loop-steps", type=int, default=None)
    parser.add_argument("--max-solver-calls", type=int, default=None)
    parser.add_argument("--max-wall-clock-seconds", type=float, default=300.0)
    parser.add_argument("--max-memory-mb", type=float, default=4096.0)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--timestamp", default=None)
    return parser.parse_args()


def _estimated_steps(n_tests: int, set_points_len: int) -> int:
    return max(int(n_tests), 1) * max(int(set_points_len), 1) * 2


def _resource_guard_from_args(args: argparse.Namespace) -> ResourceGuard:
    estimated = _estimated_steps(args.n_tests, args.set_points_len)
    max_target = args.max_target_evals
    max_closed = args.max_closed_loop_steps
    max_solver = args.max_solver_calls
    if max_target is None:
        target_multiplier = (1 if args.target_only else 0) + (len(TARGET_ABLATION_CASES) if args.target_ablation else 0)
        max_target = max(100, estimated * max(target_multiplier, 1))
    if max_closed is None:
        max_closed = max(20, estimated if args.closed_loop else 20)
    if max_solver is None:
        max_solver = max(500, 2 * max_target + 2 * max_closed)
    return ResourceGuard(
        GARTStudyLimits(
            max_target_evals=max_target,
            max_closed_loop_steps=max_closed,
            max_solver_calls=max_solver,
            max_wall_clock_seconds=float(args.max_wall_clock_seconds),
            max_memory_mb=None if args.max_memory_mb is None or args.max_memory_mb <= 0 else float(args.max_memory_mb),
        )
    )


def _apply_runtime_safety(args: argparse.Namespace) -> None:
    if args.smoke:
        args.mode = "nominal"
        args.n_tests = 1
        args.set_points_len = 20
        args.target_only = True
        args.closed_loop = False
    if args.full and not args.confirm_full:
        raise RuntimeError("Full GART runs require both --full and --confirm-full.")
    if not args.full:
        full_like = args.mode == "disturb" or int(args.n_tests) > 1 or int(args.set_points_len) > 20
        if full_like and not args.confirm_full:
            raise RuntimeError(
                "Non-smoke GART runs require --full --confirm-full. "
                "Use --smoke or keep nominal n_tests=1 set_points_len=20 for default checks."
            )
    if not args.target_only and not args.closed_loop and not args.target_ablation:
        args.target_only = True


def main() -> dict[str, Any]:
    args = parse_args()
    set_single_thread_env(args.threads)
    _apply_runtime_safety(args)
    guard = _resource_guard_from_args(args)
    timestamp = args.timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    ctx = _build_context()
    root = Path(repo_path())
    summaries: dict[str, Any] = {
        "timestamp": timestamp,
        "mode": args.mode,
        "n_tests": int(args.n_tests),
        "set_points_len": int(args.set_points_len),
        "disturbance_after_step": False,
        "target_only_mode": args.target_only_mode,
        "target_only_overrides": GART_RELAXED_DY2_OVERRIDES,
        "resource_limits": asdict(guard.limits),
    }
    if args.target_only:
        target_dir = root / "results" / "GARTTargetSelectorStudy" / timestamp
        if args.target_only_mode == "observer-replay":
            summaries["target_only"] = run_observer_replay_target_only(
                ctx,
                target_dir,
                replay_source=args.replay_source,
                replay_path=args.replay_path,
                mode=args.mode,
                n_tests=args.n_tests,
                set_points_len=args.set_points_len,
                target_overrides=GART_RELAXED_DY2_OVERRIDES,
                guard=guard,
            )
        else:
            summaries["target_only"] = run_synthetic_target_only(
                ctx,
                target_dir,
                n_tests=args.n_tests,
                set_points_len=args.set_points_len,
                target_overrides=GART_RELAXED_DY2_OVERRIDES,
                guard=guard,
            )
        summaries["target_only_dir"] = str(target_dir.relative_to(root))
    if args.target_ablation:
        ablation_dir = root / "results" / "GARTTargetAblation" / timestamp
        summaries["target_ablation"] = run_target_ablation_study(
            ctx,
            ablation_dir,
            mode=args.mode,
            n_tests=args.n_tests,
            set_points_len=args.set_points_len,
            replay_source=args.replay_source,
            replay_path=args.replay_path,
            guard=guard,
        )
        summaries["target_ablation_dir"] = str(ablation_dir.relative_to(root))
    if args.closed_loop:
        lmpc_dir = root / "results" / "GARTLMPC" / timestamp
        summaries["closed_loop"] = run_closed_loop(
            ctx,
            lmpc_dir,
            mode=args.mode,
            n_tests=args.n_tests,
            set_points_len=args.set_points_len,
            guard=guard,
        )
        summaries["closed_loop_dir"] = str(lmpc_dir.relative_to(root))
    print(json.dumps(_jsonable(summaries), indent=2))
    return summaries


if __name__ == "__main__":
    main()
