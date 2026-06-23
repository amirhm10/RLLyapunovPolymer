from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from pprint import pprint

import numpy as np

from utils.online_disturbance_runner import build_disturbance_context, run_offset_free_mpc_disturbance
from utils.two_phase_profiles import (
    TwoPhaseExperimentSpec,
    build_two_phase_profiles,
    jsonable_two_phase_profile,
)


# Offset-free MPC feasibility/setpoint probe for the online profile family.
# The setpoint-search mode is a fast screen: it uses OF-MPC execution with the
# fixed current Lyapunov diagnostic configuration to rank candidate setpoints
# before spending time on full GART/TD3 runs.

RUN_SETPOINT_SEARCH = True
PHASE1_SETPOINT_HOLD_STEPS = 400
REPORTING_WINDOW_STEPS = 800
PHASE2_EPISODES = 10
PHASE2_STEPS = PHASE2_EPISODES * REPORTING_WINDOW_STEPS
SEED = 123
SAVE_PLOTS = True

OUTPUT_ROOT = Path.home() / "Desktop" / "Lyapunov_polymer_results"
STUDY_NAME = "OffsetFreeMPC_Phase2Feasibility"
CASE_NAME = "offset_free_mpc_phase2_online_setpoint_cycle"
TIMESTAMP = None

SEARCH_OUTPUT_ROOT = Path("results")
SEARCH_STUDY_NAME = "OffsetFreeMPC_SetpointCycleSearch"
SEARCH_CASE_PREFIX = "cycle"
SEARCH_PROFILE_MODE = "cycle"  # "cycle" or "held"
SEARCH_EPISODES = 3
SEARCH_SAVE_PLOTS = False
SEARCH_TAIL_STEPS = 400
SETTLING_TAIL_STEPS = 100
SETTLING_BAND_PHYS = np.array([0.05, 0.30], dtype=float)
SEARCH_CYCLES_Y_PHYS = (
    ((4.5, 324.0), (3.4, 321.0)),
    ((4.5, 324.0), (3.35, 323.5)),
    ((4.5, 324.0), (3.3, 324.5)),
    ((4.6, 321.0), (3.35, 323.5)),
    ((4.4, 321.5), (3.3, 324.5)),
    ((4.0, 320.5), (3.35, 323.5)),
    ((4.25, 322.5), (3.1, 323.0)),
    ((4.6, 321.0), (3.2, 324.5)),
    ((4.4, 321.5), (3.1, 323.0)),
    ((4.0, 320.5), (3.3, 324.5)),
)
SEARCH_SETPOINTS_Y_PHYS = (
    (4.5, 324.0),
    (3.4, 321.0),
    (4.25, 322.5),
    (3.35, 323.5),
    (4.4, 321.5),
    (3.3, 324.5),
    (4.6, 321.0),
    (3.2, 324.5),
    (4.0, 320.5),
    (3.1, 323.0),
)

PHASE2_SETPOINTS_Y_PHYS = (
    (4.5, 324.0),
    (3.4, 321.0),
)

NOMINAL_QI = 108.0
NOMINAL_QS = 459.0
NOMINAL_HA = 1.05e6

PHASE1_QI_MULTIPLIER = 0.95
PHASE1_QS_MULTIPLIER = 1.05
PHASE1_HA_MULTIPLIER = 0.92

PHASE2_QI_MULTIPLIER = 1.02
PHASE2_QS_MULTIPLIER = 0.97
PHASE2_HA_MULTIPLIER = 0.9


def _phase2_probe_profiles(
    *,
    phase2_setpoints_y_phys=PHASE2_SETPOINTS_Y_PHYS,
    phase2_episodes: int = PHASE2_EPISODES,
) -> tuple[np.ndarray, dict[str, np.ndarray], dict]:
    context = build_disturbance_context()
    spec = TwoPhaseExperimentSpec(
        phase1_episodes=1,
        phase2_episodes=int(phase2_episodes),
        phase2_steps=None,
        set_points_len=int(PHASE1_SETPOINT_HOLD_STEPS),
        reporting_window_steps=int(REPORTING_WINDOW_STEPS),
        phase2_setpoints_y_phys=np.asarray(phase2_setpoints_y_phys, dtype=float),
        nominal_qi=float(NOMINAL_QI),
        nominal_qs=float(NOMINAL_QS),
        nominal_ha=float(NOMINAL_HA),
        phase1_qi_multiplier=float(PHASE1_QI_MULTIPLIER),
        phase1_qs_multiplier=float(PHASE1_QS_MULTIPLIER),
        phase1_ha_multiplier=float(PHASE1_HA_MULTIPLIER),
        phase2_qi_multiplier=float(PHASE2_QI_MULTIPLIER),
        phase2_qs_multiplier=float(PHASE2_QS_MULTIPLIER),
        phase2_ha_multiplier=float(PHASE2_HA_MULTIPLIER),
    )
    profile = build_two_phase_profiles(
        spec=spec,
        data_min=context.system_data["data_min"],
        data_max=context.system_data["data_max"],
        steady_outputs=context.setup.steady_states["y_ss"],
        n_inputs=context.dimensions.inputs_number,
    )
    start = int(profile["phase1_steps"])
    stop = int(profile["total_steps"])
    setpoint_profile = np.asarray(profile["setpoint_profile_scaled_dev"], dtype=float)[start:stop].copy()
    disturbance_profile = {
        name: np.asarray(values, dtype=float)[start:stop].copy()
        for name, values in profile["disturbance_profile"].items()
    }
    scenario_len = int(np.asarray(context.y_sp_scenario).shape[0])
    if REPORTING_WINDOW_STEPS % scenario_len != 0:
        raise ValueError(
            "REPORTING_WINDOW_STEPS must be divisible by the rollout setpoint-scenario count; "
            f"got {REPORTING_WINDOW_STEPS} and {scenario_len}."
        )
    rollout_n_tests = int((stop - start) // REPORTING_WINDOW_STEPS)
    rollout_set_points_len = int(REPORTING_WINDOW_STEPS // scenario_len)
    metadata = {
        "probe": "phase2_only_feasibility",
        "phase2_episodes": int(phase2_episodes),
        "phase2_steps": int(stop - start),
        "reporting_window_steps": int(REPORTING_WINDOW_STEPS),
        "rollout_n_tests": int(rollout_n_tests),
        "rollout_set_points_len": int(rollout_set_points_len),
        "setpoints_y_phys": np.asarray(phase2_setpoints_y_phys, dtype=float).tolist(),
        "n_profile_steps": int(stop - start),
        "disturbance_start": {name: float(values[0]) for name, values in disturbance_profile.items()},
        "disturbance_end": {name: float(values[-1]) for name, values in disturbance_profile.items()},
        "source_two_phase_profile": jsonable_two_phase_profile(profile),
    }
    return setpoint_profile, disturbance_profile, metadata


def _slug_number(value: float) -> str:
    return f"{float(value):.3f}".replace("-", "m").replace(".", "p").rstrip("0").rstrip("p")


def _setpoint_case_name(index: int, y_phys: np.ndarray) -> str:
    return (
        f"{SEARCH_CASE_PREFIX}_{int(index):02d}"
        f"_eta{_slug_number(float(y_phys[0]))}"
        f"_T{_slug_number(float(y_phys[1]))}"
    )


def _cycle_case_name(index: int, y_cycle_phys: np.ndarray) -> str:
    y_cycle_phys = np.asarray(y_cycle_phys, dtype=float)
    labels = [
        f"eta{_slug_number(float(row[0]))}_T{_slug_number(float(row[1]))}"
        for row in y_cycle_phys
    ]
    return f"{SEARCH_CASE_PREFIX}_{int(index):02d}_" + "_to_".join(labels)


def _safe_nanmean(values) -> float:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0 or not np.any(np.isfinite(arr)):
        return float("nan")
    return float(np.nanmean(arr))


def _safe_nanmax(values) -> float:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0 or not np.any(np.isfinite(arr)):
        return float("nan")
    return float(np.nanmax(arr))


def _row_inf_norm(values) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 2 or arr.shape[0] == 0:
        return np.array([], dtype=float)
    return np.nanmax(np.abs(arr), axis=1)


def _tail_window(values, tail_steps: int = SEARCH_TAIL_STEPS) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.shape[0] == 0:
        return arr
    start = max(0, int(arr.shape[0]) - int(tail_steps))
    return arr[start:]


def _tail_step_motion(values) -> tuple[float, float]:
    tail = _tail_window(values)
    if tail.ndim != 2 or tail.shape[0] <= 1:
        return float("nan"), float("nan")
    motion = _row_inf_norm(np.diff(tail, axis=0))
    return _safe_nanmean(motion), _safe_nanmax(motion)


def _sign_change_count(values, *, eps: float = 1.0e-8) -> int:
    tail = _tail_window(values)
    return _sign_change_count_matrix(tail, eps=eps)


def _sign_change_count_matrix(values, *, eps: float = 1.0e-8) -> int:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 2 or arr.shape[0] <= 2:
        return 0
    du = np.diff(arr, axis=0)
    total = 0
    for col in range(du.shape[1]):
        signs = np.sign(du[:, col])
        signs[np.abs(du[:, col]) <= eps] = 0.0
        nonzero = signs[signs != 0.0]
        if nonzero.size > 1:
            total += int(np.sum(nonzero[1:] * nonzero[:-1] < 0.0))
    return total


def _cycle_settling_metrics(
    *,
    y_err_phys: np.ndarray,
    y_output_phys: np.ndarray,
    u_apply_dev: np.ndarray,
    block_steps: int,
    tail_steps: int,
    settling_band_phys: np.ndarray,
) -> dict[str, float | int]:
    y_err_phys = np.asarray(y_err_phys, dtype=float)
    y_output_phys = np.asarray(y_output_phys, dtype=float)
    u_apply_dev = np.asarray(u_apply_dev, dtype=float)
    block_steps = int(block_steps)
    tail_steps = int(tail_steps)
    if y_err_phys.ndim != 2 or y_err_phys.shape[0] == 0 or block_steps <= 0:
        return {
            "cycle_blocks": 0,
            "cycle_nonsettled_blocks": 0,
            "cycle_tail_error_norm_mean": float("nan"),
            "cycle_tail_error_norm_max": float("nan"),
            "cycle_final_error_norm_mean": float("nan"),
            "cycle_final_error_norm_max": float("nan"),
            "cycle_output_sign_changes": 0,
            "cycle_input_sign_changes": 0,
            "cycle_tail_output_motion_mean": float("nan"),
            "cycle_tail_output_motion_max": float("nan"),
        }

    band = np.asarray(settling_band_phys, dtype=float).reshape(1, -1)
    band = np.maximum(band, 1.0e-12)
    n_blocks = int(y_err_phys.shape[0] // block_steps)
    tail_norms = []
    final_norms = []
    output_motion = []
    output_sign_changes = 0
    input_sign_changes = 0

    for block_idx in range(n_blocks):
        start = int(block_idx * block_steps)
        stop = int(min((block_idx + 1) * block_steps, y_err_phys.shape[0]))
        tail_start = max(start, stop - tail_steps)
        err_tail = np.abs(y_err_phys[tail_start:stop, :]) / band
        if err_tail.size:
            tail_norms.append(float(np.nanmean(np.nanmax(err_tail, axis=1))))
            final_norms.append(float(np.nanmax(err_tail[-1, :])))
        y_tail = y_output_phys[tail_start:stop, :] if y_output_phys.ndim == 2 else np.empty((0, 2))
        u_tail = u_apply_dev[tail_start:stop, :] if u_apply_dev.ndim == 2 else np.empty((0, 2))
        output_sign_changes += _sign_change_count_matrix(y_tail)
        input_sign_changes += _sign_change_count_matrix(u_tail)
        if y_tail.shape[0] > 1:
            output_motion.extend(_row_inf_norm(np.diff(y_tail, axis=0)).tolist())

    tail_norms_arr = np.asarray(tail_norms, dtype=float)
    final_norms_arr = np.asarray(final_norms, dtype=float)
    output_motion_arr = np.asarray(output_motion, dtype=float)
    return {
        "cycle_blocks": int(n_blocks),
        "cycle_nonsettled_blocks": int(np.nansum(tail_norms_arr > 1.0)),
        "cycle_tail_error_norm_mean": _safe_nanmean(tail_norms_arr),
        "cycle_tail_error_norm_max": _safe_nanmax(tail_norms_arr),
        "cycle_final_error_norm_mean": _safe_nanmean(final_norms_arr),
        "cycle_final_error_norm_max": _safe_nanmax(final_norms_arr),
        "cycle_output_sign_changes": int(output_sign_changes),
        "cycle_input_sign_changes": int(input_sign_changes),
        "cycle_tail_output_motion_mean": _safe_nanmean(output_motion_arr),
        "cycle_tail_output_motion_max": _safe_nanmax(output_motion_arr),
    }


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _search_record_from_debug_dir(*, debug_dir: str | Path, case_name: str, y_phys: np.ndarray) -> dict:
    debug_path = Path(debug_dir)
    summary = _load_json(debug_path / "summary.json")
    with np.load(debug_path / "arrays.npz") as data:
        arrays = {name: data[name].copy() for name in data.files}

    y_phys = np.asarray(y_phys, dtype=float)
    u_apply = arrays.get("u_apply_dev_store", np.empty((0, 2)))
    u_target = arrays.get("u_target_dev_store", np.empty((0, 2)))
    y_target = arrays.get("y_target_store", np.empty((0, 2)))
    y_err_phys = arrays.get("y_minus_y_sp_phys_store", np.empty((0, 2)))
    ys_err_phys = arrays.get("y_s_minus_y_sp_phys_store", np.empty((0, 2)))
    y_system = arrays.get("y_system", np.empty((0, 2)))
    y_output_phys = y_system[1:, :] if y_system.ndim == 2 and y_system.shape[0] > 1 else np.empty((0, 2))
    diagnostic_unsafe = arrays.get("diagnostic_unsafe_flags", np.array([], dtype=float))
    diagnostic_unstable = arrays.get("diagnostic_unstable_flags", np.array([], dtype=float))
    contraction_margin = arrays.get("contraction_margin", np.array([], dtype=float))

    du_apply_mean, du_apply_max = _tail_step_motion(u_apply)
    du_target_mean, du_target_max = _tail_step_motion(u_target)
    dy_target_mean, dy_target_max = _tail_step_motion(y_target)
    y_tail = _tail_window(y_err_phys)
    ys_tail = _tail_window(ys_err_phys)
    y_rmse_tail = float(np.sqrt(np.nanmean(y_tail ** 2))) if y_tail.size else float("nan")
    ys_mismatch_tail = _safe_nanmean(_row_inf_norm(ys_tail))
    unsafe_rate = _safe_nanmean(diagnostic_unsafe)
    unstable_rate = _safe_nanmean(diagnostic_unstable)
    contraction_margin_max = _safe_nanmax(contraction_margin)
    contraction_margin_pos = max(0.0, contraction_margin_max) if np.isfinite(contraction_margin_max) else float("nan")
    sign_changes = _sign_change_count(u_apply)
    input_phys = arrays.get("u_applied_phys", np.empty((0, 2)))
    cycle_metrics = {}
    if y_phys.ndim == 2 and y_phys.shape[0] > 1:
        cycle_metrics = _cycle_settling_metrics(
            y_err_phys=y_err_phys,
            y_output_phys=y_output_phys,
            u_apply_dev=u_apply,
            block_steps=PHASE1_SETPOINT_HOLD_STEPS,
            tail_steps=SETTLING_TAIL_STEPS,
            settling_band_phys=SETTLING_BAND_PHYS,
        )

    oscillation_score = (
        20.0 * (0.0 if not np.isfinite(du_apply_mean) else du_apply_mean)
        + 20.0 * (0.0 if not np.isfinite(du_target_mean) else du_target_mean)
        + 10.0 * (0.0 if not np.isfinite(dy_target_mean) else dy_target_mean)
        + 0.05 * float(sign_changes)
    )
    if cycle_metrics:
        blocks = max(int(cycle_metrics["cycle_blocks"]), 1)
        nonsettled_rate = float(cycle_metrics["cycle_nonsettled_blocks"]) / float(blocks)
        cycle_tail_error = cycle_metrics["cycle_tail_error_norm_mean"]
        cycle_output_sign_changes = cycle_metrics["cycle_output_sign_changes"]
        cycle_input_sign_changes = cycle_metrics["cycle_input_sign_changes"]
        cycle_motion = cycle_metrics["cycle_tail_output_motion_mean"]
        oscillation_score = (
            50.0 * nonsettled_rate
            + 10.0 * (0.0 if not np.isfinite(cycle_tail_error) else float(cycle_tail_error))
            + 0.50 * float(cycle_output_sign_changes)
            + 0.20 * float(cycle_input_sign_changes)
            + 2.0 * (0.0 if not np.isfinite(cycle_motion) else float(cycle_motion))
        )
    safety_score = (
        50.0 * (0.0 if not np.isfinite(unsafe_rate) else unsafe_rate)
        + 50.0 * (0.0 if not np.isfinite(unstable_rate) else unstable_rate)
        + 10.0 * (0.0 if not np.isfinite(contraction_margin_pos) else contraction_margin_pos)
    )

    record = {
        "case_name": case_name,
        "eta_sp": float(y_phys.reshape(-1, 2)[0, 0]),
        "T_sp": float(y_phys.reshape(-1, 2)[0, 1]),
        "setpoint_cycle_y_phys": json.dumps(y_phys.tolist()),
        "search_score": float(oscillation_score + safety_score),
        "oscillation_score": float(oscillation_score),
        "safety_score": float(safety_score),
        "n_steps": int(summary.get("n_steps", len(u_apply))),
        "reward_mean": summary.get("reward_mean"),
        "solver_success_rate": summary.get("solver_success_rate"),
        "target_success_rate": summary.get("target_success_rate"),
        "diagnostic_unsafe_rate": summary.get("diagnostic_unsafe_rate", unsafe_rate),
        "diagnostic_unsafe_count": int(np.nansum(diagnostic_unsafe > 0.5)),
        "diagnostic_unstable_rate": summary.get("diagnostic_unstable_rate", unstable_rate),
        "diagnostic_unstable_count": int(np.nansum(diagnostic_unstable > 0.5)),
        "contraction_margin_max": contraction_margin_max,
        "contraction_margin_positive_max": contraction_margin_pos,
        "tail_du_apply_scaled_mean": du_apply_mean,
        "tail_du_apply_scaled_max": du_apply_max,
        "tail_du_target_scaled_mean": du_target_mean,
        "tail_du_target_scaled_max": du_target_max,
        "tail_dy_target_scaled_mean": dy_target_mean,
        "tail_dy_target_scaled_max": dy_target_max,
        "tail_input_sign_changes": int(sign_changes),
        "tail_output_rmse_phys": y_rmse_tail,
        "tail_ys_ysp_inf_phys_mean": ys_mismatch_tail,
        "input_Qc_min": _safe_nanmin(input_phys[:, 0]) if input_phys.ndim == 2 and input_phys.shape[0] else float("nan"),
        "input_Qc_max": _safe_nanmax(input_phys[:, 0]) if input_phys.ndim == 2 and input_phys.shape[0] else float("nan"),
        "input_Qm_min": _safe_nanmin(input_phys[:, 1]) if input_phys.ndim == 2 and input_phys.shape[0] else float("nan"),
        "input_Qm_max": _safe_nanmax(input_phys[:, 1]) if input_phys.ndim == 2 and input_phys.shape[0] else float("nan"),
        "target_reference_error_inf_mean": summary.get("target_reference_error_inf_mean"),
        "target_reference_error_inf_max": summary.get("target_reference_error_inf_max"),
        "output_reference_error_inf_mean": summary.get("output_reference_error_inf_mean"),
        "output_reference_error_inf_max": summary.get("output_reference_error_inf_max"),
        "debug_dir": str(debug_path),
    }
    record.update(cycle_metrics)
    return record


def _safe_nanmin(values) -> float:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0 or not np.any(np.isfinite(arr)):
        return float("nan")
    return float(np.nanmin(arr))


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _float_from_record(record: dict, key: str, default: float = float("nan")) -> float:
    value = record.get(key, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _int_from_record(record: dict, key: str, default: int = 0) -> int:
    value = record.get(key, default)
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return int(default)


def _load_search_records(summary_csv: str | Path) -> list[dict]:
    summary_csv = Path(summary_csv)
    with summary_csv.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return sorted(rows, key=lambda row: _float_from_record(row, "search_score", -np.inf), reverse=True)


def _search_plot_arrays(debug_dir: str | Path) -> dict[str, np.ndarray]:
    with np.load(Path(debug_dir) / "arrays.npz") as data:
        arrays = {name: data[name].copy() for name in data.files}
    y_system = np.asarray(arrays.get("y_system", np.empty((0, 2))), dtype=float)
    y_output = y_system[1:, :] if y_system.ndim == 2 and y_system.shape[0] > 1 else np.empty((0, 2))
    y_err = np.asarray(arrays.get("y_minus_y_sp_phys_store", np.empty((0, 2))), dtype=float)
    ys_err = np.asarray(arrays.get("y_s_minus_y_sp_phys_store", np.empty((0, 2))), dtype=float)
    n = min(y_output.shape[0], y_err.shape[0], ys_err.shape[0])
    if n <= 0:
        return {
            "y_output": np.empty((0, 2)),
            "y_sp": np.empty((0, 2)),
            "y_s": np.empty((0, 2)),
            "unsafe": np.array([], dtype=bool),
        }
    y_output = y_output[:n, :]
    y_err = y_err[:n, :]
    ys_err = ys_err[:n, :]
    unsafe = np.asarray(arrays.get("diagnostic_unsafe_flags", np.zeros(n)), dtype=float).reshape(-1)
    if unsafe.shape[0] < n:
        unsafe = np.pad(unsafe, (0, n - unsafe.shape[0]), constant_values=0.0)
    y_sp = y_output - y_err
    y_s = y_sp + ys_err
    return {
        "y_output": y_output,
        "y_sp": y_sp,
        "y_s": y_s,
        "unsafe": unsafe[:n] > 0.5,
    }


def _shade_unsafe_regions(ax, unsafe: np.ndarray) -> None:
    unsafe = np.asarray(unsafe, dtype=bool).reshape(-1)
    if unsafe.size == 0 or not np.any(unsafe):
        return
    padded = np.r_[False, unsafe, False]
    changes = np.flatnonzero(padded[1:] != padded[:-1])
    for start, stop in changes.reshape(-1, 2):
        ax.axvspan(start, stop, color="tab:red", alpha=0.10, linewidth=0)


def _draw_cycle_switches(ax, n_steps: int) -> None:
    for step in range(PHASE1_SETPOINT_HOLD_STEPS, int(n_steps), PHASE1_SETPOINT_HOLD_STEPS):
        ax.axvline(step, color="0.65", linewidth=0.8, linestyle=":", alpha=0.8)


def _plot_record_tracking(record: dict, path: Path, *, show_legend: bool = True) -> str:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    arrays = _search_plot_arrays(record["debug_dir"])
    y_output = arrays["y_output"]
    if y_output.size == 0:
        return ""
    y_sp = arrays["y_sp"]
    y_s = arrays["y_s"]
    unsafe = arrays["unsafe"]
    t = np.arange(y_output.shape[0])
    labels = [r"$\eta$", r"$T$"]
    units = ["", "K"]
    bands = SETTLING_BAND_PHYS.reshape(-1)

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(12.0, 6.8), sharex=True)
    for idx, ax in enumerate(axes):
        _shade_unsafe_regions(ax, unsafe)
        _draw_cycle_switches(ax, y_output.shape[0])
        ax.plot(t, y_output[:, idx], color="tab:blue", linewidth=1.6, label="OF-MPC output")
        ax.step(t, y_sp[:, idx], where="post", color="black", linestyle="--", linewidth=1.3, label="raw setpoint")
        ax.step(t, y_s[:, idx], where="post", color="tab:orange", linestyle="-.", linewidth=1.2, label="governed target")
        if idx < bands.shape[0]:
            ax.fill_between(
                t,
                y_sp[:, idx] - bands[idx],
                y_sp[:, idx] + bands[idx],
                step="post",
                color="0.75",
                alpha=0.18,
                label="settling band" if idx == 0 else None,
            )
        ylabel = labels[idx] if units[idx] == "" else f"{labels[idx]} ({units[idx]})"
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
    title = (
        f"{record['case_name']} | score={_float_from_record(record, 'search_score'):.1f}, "
        f"nonsettled={_int_from_record(record, 'cycle_nonsettled_blocks')}, "
        f"unsafe={_int_from_record(record, 'diagnostic_unsafe_count')}, "
        f"y sign changes={_int_from_record(record, 'cycle_output_sign_changes')}"
    )
    axes[0].set_title(title)
    axes[-1].set_xlabel("sample")
    if show_legend:
        handles, labels_seen = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels_seen, loc="lower center", ncol=4, frameon=False)
        fig.subplots_adjust(bottom=0.16, hspace=0.14)
    else:
        fig.subplots_adjust(bottom=0.10, hspace=0.14)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def _plot_records_grid(records: list[dict], path: Path, *, title: str) -> str:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    rows = []
    for record in records:
        arrays = _search_plot_arrays(record["debug_dir"])
        if arrays["y_output"].size:
            rows.append((record, arrays))
    if not rows:
        return ""

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(len(rows), 2, figsize=(14.0, max(3.0 * len(rows), 3.4)), sharex=True)
    axes = np.atleast_2d(axes)
    for row_idx, (record, arrays) in enumerate(rows):
        y_output = arrays["y_output"]
        y_sp = arrays["y_sp"]
        y_s = arrays["y_s"]
        unsafe = arrays["unsafe"]
        t = np.arange(y_output.shape[0])
        for col_idx, name in enumerate((r"$\eta$", r"$T$")):
            ax = axes[row_idx, col_idx]
            _shade_unsafe_regions(ax, unsafe)
            _draw_cycle_switches(ax, y_output.shape[0])
            ax.plot(t, y_output[:, col_idx], color="tab:blue", linewidth=1.1)
            ax.step(t, y_sp[:, col_idx], where="post", color="black", linestyle="--", linewidth=1.0)
            ax.step(t, y_s[:, col_idx], where="post", color="tab:orange", linestyle="-.", linewidth=0.95)
            ax.grid(True, alpha=0.20)
            ax.set_ylabel(name)
            if row_idx == 0:
                ax.set_title("viscosity-like output" if col_idx == 0 else "reactor temperature")
            if col_idx == 0:
                label = (
                    f"{record['case_name']}\n"
                    f"score={_float_from_record(record, 'search_score'):.1f}, "
                    f"nonsettled={_int_from_record(record, 'cycle_nonsettled_blocks')}, "
                    f"unsafe={_int_from_record(record, 'diagnostic_unsafe_count')}"
                )
                ax.text(
                    0.01,
                    0.96,
                    label,
                    transform=ax.transAxes,
                    va="top",
                    ha="left",
                    fontsize=8.5,
                    bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "edgecolor": "0.8", "alpha": 0.85},
                )
    for ax in axes[-1, :]:
        ax.set_xlabel("sample")
    fig.suptitle(title, y=0.995)
    fig.subplots_adjust(top=0.96, hspace=0.26, wspace=0.16)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def plot_setpoint_search_tracking(
    summary_csv: str | Path,
    *,
    top_n: int = 5,
    output_dir: str | Path | None = None,
) -> dict[str, str]:
    summary_csv = Path(summary_csv)
    records = _load_search_records(summary_csv)
    plot_dir = Path(output_dir) if output_dir is not None else summary_csv.parent / "tracking_plots"
    plot_paths: dict[str, str] = {}
    top_records = records[: int(top_n)]
    for record in top_records:
        plot_path = plot_dir / f"tracking_{record['case_name']}.png"
        saved = _plot_record_tracking(record, plot_path)
        if saved:
            plot_paths[f"individual_{record['case_name']}"] = saved
    saved = _plot_records_grid(
        top_records,
        plot_dir / "tracking_top_cycle_candidates.png",
        title=f"Top {len(top_records)} setpoint-cycle candidates",
    )
    if saved:
        plot_paths["top_candidates_grid"] = saved
    saved = _plot_records_grid(
        records,
        plot_dir / "tracking_all_cycle_candidates_grid.png",
        title="All screened setpoint-cycle candidates",
    )
    if saved:
        plot_paths["all_candidates_grid"] = saved
    note = {
        "source_summary_csv": str(summary_csv),
        "settling_band_phys": SETTLING_BAND_PHYS.tolist(),
        "cycle_switch_samples": int(PHASE1_SETPOINT_HOLD_STEPS),
        "diagnostic_unsafe_shading": "red spans mark diagnostic unsafe OF-MPC actions",
        "raw_setpoint": "black dashed",
        "governed_target": "orange dash-dot",
        "plant_output": "blue solid",
    }
    (plot_dir / "tracking_plot_notes.json").write_text(json.dumps(note, indent=2), encoding="utf-8")
    plot_paths["notes"] = str(plot_dir / "tracking_plot_notes.json")
    return plot_paths


def run_setpoint_search() -> dict:
    timestamp = str(TIMESTAMP) if TIMESTAMP is not None else datetime.now().strftime("%Y%m%d_%H%M%S_setpoint_search")
    search_root = Path(SEARCH_OUTPUT_ROOT) / SEARCH_STUDY_NAME / timestamp
    search_root.mkdir(parents=True, exist_ok=True)
    if SEARCH_PROFILE_MODE not in {"cycle", "held"}:
        raise ValueError("SEARCH_PROFILE_MODE must be 'cycle' or 'held'.")
    search_items = SEARCH_CYCLES_Y_PHYS if SEARCH_PROFILE_MODE == "cycle" else SEARCH_SETPOINTS_Y_PHYS
    records = []
    for idx, setpoint in enumerate(search_items):
        y_phys = np.asarray(setpoint, dtype=float)
        if SEARCH_PROFILE_MODE == "cycle":
            y_profile_phys = y_phys.reshape(-1, 2)
            case_name = _cycle_case_name(idx, y_profile_phys)
        else:
            y_profile_phys = y_phys.reshape(1, 2)
            case_name = _setpoint_case_name(idx, y_phys.reshape(2))
        candidate_timestamp = f"{timestamp}_{case_name}"
        setpoint_profile, disturbance_profile, metadata = _phase2_probe_profiles(
            phase2_setpoints_y_phys=y_profile_phys,
            phase2_episodes=int(SEARCH_EPISODES),
        )
        metadata = {
            **metadata,
            "probe": f"{SEARCH_PROFILE_MODE}_setpoint_search",
            "candidate_index": int(idx),
            "candidate_setpoint_y_phys": y_profile_phys.tolist(),
            "search_tail_steps": int(SEARCH_TAIL_STEPS),
            "settling_tail_steps": int(SETTLING_TAIL_STEPS),
            "settling_band_phys": SETTLING_BAND_PHYS.tolist(),
        }
        print(f"\n[{idx + 1}/{len(search_items)}] Screening {SEARCH_PROFILE_MODE} setpoint {y_profile_phys.tolist()}")
        result = run_offset_free_mpc_disturbance(
            episodes=int(metadata["rollout_n_tests"]),
            set_points_len=int(metadata["rollout_set_points_len"]),
            seed=SEED,
            save_plots=SEARCH_SAVE_PLOTS,
            timestamp=candidate_timestamp,
            output_root=SEARCH_OUTPUT_ROOT,
            study_name=SEARCH_STUDY_NAME,
            case_name=case_name,
            setpoint_profile=setpoint_profile,
            disturbance_profile=disturbance_profile,
            profile_metadata=metadata,
        )
        record = _search_record_from_debug_dir(
            debug_dir=result["debug_dir"],
            case_name=case_name,
            y_phys=y_profile_phys,
        )
        record["result_root"] = result["result_root"]
        records.append(record)
        print(
            "score:",
            f"{record['search_score']:.3g}",
            "| unsafe:",
            record["diagnostic_unsafe_count"],
            "| nonsettled:",
            record.get("cycle_nonsettled_blocks"),
            "| output sign changes:",
            record.get("cycle_output_sign_changes"),
        )

    records = sorted(records, key=lambda row: row["search_score"], reverse=True)
    summary_csv = search_root / "setpoint_search_summary.csv"
    summary_json = search_root / "setpoint_search_summary.json"
    _write_csv(summary_csv, records)
    summary_json.write_text(json.dumps(records, indent=2), encoding="utf-8")
    plot_paths = plot_setpoint_search_tracking(summary_csv)
    print("\nTop setpoint-search candidates:")
    for row in records[:5]:
        print(
            f"{row['case_name']}: score={row['search_score']:.3g}, "
            f"unsafe={row['diagnostic_unsafe_count']}, "
            f"nonsettled={row.get('cycle_nonsettled_blocks')}, "
            f"output_sign_changes={row.get('cycle_output_sign_changes')}"
        )
    print(f"\nSetpoint-search summary: {summary_csv}")
    if plot_paths:
        print(f"Tracking plots: {Path(plot_paths.get('top_candidates_grid', next(iter(plot_paths.values())))).parent}")
    return {
        "study_name": SEARCH_STUDY_NAME,
        "result_root": str(search_root),
        "summary_csv": str(summary_csv),
        "summary_json": str(summary_json),
        "plot_paths": plot_paths,
        "records": records,
    }


def run_configured_study() -> dict:
    if RUN_SETPOINT_SEARCH:
        return run_setpoint_search()

    setpoint_profile, disturbance_profile, metadata = _phase2_probe_profiles()
    config = {
        "phase2_episodes": int(PHASE2_EPISODES),
        "phase2_steps": int(PHASE2_STEPS),
        "reporting_window_steps": int(REPORTING_WINDOW_STEPS),
        "rollout_n_tests": int(metadata["rollout_n_tests"]),
        "rollout_set_points_len": int(metadata["rollout_set_points_len"]),
        "seed": int(SEED),
        "save_plots": bool(SAVE_PLOTS),
        "output_root": str(OUTPUT_ROOT),
        "study_name": STUDY_NAME,
        "case_name": CASE_NAME,
        "timestamp": TIMESTAMP,
        "phase2_setpoints_y_phys": PHASE2_SETPOINTS_Y_PHYS,
        "disturbance_start": metadata["disturbance_start"],
        "disturbance_end": metadata["disturbance_end"],
    }
    print("Offset-free MPC Phase-2 feasibility configuration:")
    pprint(config)
    return run_offset_free_mpc_disturbance(
        episodes=int(metadata["rollout_n_tests"]),
        set_points_len=int(metadata["rollout_set_points_len"]),
        seed=SEED,
        save_plots=SAVE_PLOTS,
        timestamp=TIMESTAMP,
        output_root=OUTPUT_ROOT,
        study_name=STUDY_NAME,
        case_name=CASE_NAME,
        setpoint_profile=setpoint_profile,
        disturbance_profile=disturbance_profile,
        profile_metadata=metadata,
    )


if __name__ == "__main__":
    run_configured_study()
