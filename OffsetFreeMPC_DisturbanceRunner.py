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
SEARCH_STUDY_NAME = "OffsetFreeMPC_SetpointSearch"
SEARCH_CASE_PREFIX = "setpoint"
SEARCH_EPISODES = 2
SEARCH_SAVE_PLOTS = False
SEARCH_TAIL_STEPS = 400
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
    if tail.ndim != 2 or tail.shape[0] <= 2:
        return 0
    du = np.diff(tail, axis=0)
    total = 0
    for col in range(du.shape[1]):
        signs = np.sign(du[:, col])
        signs[np.abs(du[:, col]) <= eps] = 0.0
        nonzero = signs[signs != 0.0]
        if nonzero.size > 1:
            total += int(np.sum(nonzero[1:] * nonzero[:-1] < 0.0))
    return total


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _search_record_from_debug_dir(*, debug_dir: str | Path, case_name: str, y_phys: np.ndarray) -> dict:
    debug_path = Path(debug_dir)
    summary = _load_json(debug_path / "summary.json")
    with np.load(debug_path / "arrays.npz") as data:
        arrays = {name: data[name].copy() for name in data.files}

    u_apply = arrays.get("u_apply_dev_store", np.empty((0, 2)))
    u_target = arrays.get("u_target_dev_store", np.empty((0, 2)))
    y_target = arrays.get("y_target_store", np.empty((0, 2)))
    y_err_phys = arrays.get("y_minus_y_sp_phys_store", np.empty((0, 2)))
    ys_err_phys = arrays.get("y_s_minus_y_sp_phys_store", np.empty((0, 2)))
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

    oscillation_score = (
        20.0 * (0.0 if not np.isfinite(du_apply_mean) else du_apply_mean)
        + 20.0 * (0.0 if not np.isfinite(du_target_mean) else du_target_mean)
        + 10.0 * (0.0 if not np.isfinite(dy_target_mean) else dy_target_mean)
        + 0.05 * float(sign_changes)
    )
    safety_score = (
        100.0 * (0.0 if not np.isfinite(unsafe_rate) else unsafe_rate)
        + 100.0 * (0.0 if not np.isfinite(unstable_rate) else unstable_rate)
        + 10.0 * (0.0 if not np.isfinite(contraction_margin_pos) else contraction_margin_pos)
    )

    return {
        "case_name": case_name,
        "eta_sp": float(y_phys[0]),
        "T_sp": float(y_phys[1]),
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


def run_setpoint_search() -> dict:
    timestamp = str(TIMESTAMP) if TIMESTAMP is not None else datetime.now().strftime("%Y%m%d_%H%M%S_setpoint_search")
    search_root = Path(SEARCH_OUTPUT_ROOT) / SEARCH_STUDY_NAME / timestamp
    search_root.mkdir(parents=True, exist_ok=True)
    records = []
    for idx, setpoint in enumerate(SEARCH_SETPOINTS_Y_PHYS):
        y_phys = np.asarray(setpoint, dtype=float).reshape(2)
        case_name = _setpoint_case_name(idx, y_phys)
        candidate_timestamp = f"{timestamp}_{case_name}"
        setpoint_profile, disturbance_profile, metadata = _phase2_probe_profiles(
            phase2_setpoints_y_phys=y_phys.reshape(1, 2),
            phase2_episodes=int(SEARCH_EPISODES),
        )
        metadata = {
            **metadata,
            "probe": "held_setpoint_search",
            "candidate_index": int(idx),
            "candidate_setpoint_y_phys": y_phys.tolist(),
            "search_tail_steps": int(SEARCH_TAIL_STEPS),
        }
        print(f"\n[{idx + 1}/{len(SEARCH_SETPOINTS_Y_PHYS)}] Screening setpoint {tuple(y_phys)}")
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
            y_phys=y_phys,
        )
        record["result_root"] = result["result_root"]
        records.append(record)
        print(
            "score:",
            f"{record['search_score']:.3g}",
            "| unsafe:",
            record["diagnostic_unsafe_count"],
            "| tail du:",
            f"{record['tail_du_apply_scaled_mean']:.3g}",
            "| sign changes:",
            record["tail_input_sign_changes"],
        )

    records = sorted(records, key=lambda row: row["search_score"], reverse=True)
    summary_csv = search_root / "setpoint_search_summary.csv"
    summary_json = search_root / "setpoint_search_summary.json"
    _write_csv(summary_csv, records)
    summary_json.write_text(json.dumps(records, indent=2), encoding="utf-8")
    print("\nTop setpoint-search candidates:")
    for row in records[:5]:
        print(
            f"{row['case_name']}: score={row['search_score']:.3g}, "
            f"unsafe={row['diagnostic_unsafe_count']}, "
            f"tail_du={row['tail_du_apply_scaled_mean']:.3g}, "
            f"sign_changes={row['tail_input_sign_changes']}"
        )
    print(f"\nSetpoint-search summary: {summary_csv}")
    return {
        "study_name": SEARCH_STUDY_NAME,
        "result_root": str(search_root),
        "summary_csv": str(summary_csv),
        "summary_json": str(summary_json),
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
