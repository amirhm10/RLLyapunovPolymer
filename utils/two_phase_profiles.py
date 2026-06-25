from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from utils.scaling_helpers import apply_min_max


PHASE1_SETPOINT_Y_PHYS = np.array(
    [
        [4.5, 324.0],
        [3.4, 321.0],
    ],
    dtype=float,
)

PHASE2_SETPOINT_Y_PHYS = np.array(
    [
        [4.5, 324.0],
        [3.4, 321.0],
    ],
    dtype=float,
)


@dataclass(frozen=True)
class TwoPhaseExperimentSpec:
    phase1_episodes: int = 150
    phase2_episodes: int | None = 50
    phase2_steps: int | None = None
    set_points_len: int = 400
    reporting_window_steps: int = 800
    phase1_setpoints_y_phys: np.ndarray = field(default_factory=lambda: PHASE1_SETPOINT_Y_PHYS.copy())
    phase2_setpoints_y_phys: np.ndarray = field(default_factory=lambda: PHASE2_SETPOINT_Y_PHYS.copy())
    nominal_qi: float = 108.0
    nominal_qs: float = 459.0
    nominal_ha: float = 1.05e6
    phase1_qi_multiplier: float = 0.95
    phase1_qs_multiplier: float = 1.05
    phase1_ha_multiplier: float = 0.92
    phase2_qi_multiplier: float = 1.02
    phase2_qs_multiplier: float = 0.97
    phase2_ha_multiplier: float = 0.90


def _validate_spec(spec: TwoPhaseExperimentSpec) -> None:
    if int(spec.phase1_episodes) <= 0:
        raise ValueError("phase1_episodes must be positive.")
    if spec.phase2_episodes is None and spec.phase2_steps is None:
        raise ValueError("Either phase2_episodes or phase2_steps must be provided.")
    if spec.phase2_episodes is not None and spec.phase2_steps is not None:
        raise ValueError("Use either phase2_episodes or phase2_steps, not both.")
    if spec.phase2_episodes is not None and int(spec.phase2_episodes) < 0:
        raise ValueError("phase2_episodes must be nonnegative when provided.")
    if spec.phase2_steps is not None and int(spec.phase2_steps) < 0:
        raise ValueError("phase2_steps must be nonnegative when provided.")
    if int(spec.set_points_len) <= 0:
        raise ValueError("set_points_len must be positive.")
    if int(spec.reporting_window_steps) <= 0:
        raise ValueError("reporting_window_steps must be positive.")
    p1 = np.asarray(spec.phase1_setpoints_y_phys, dtype=float)
    p2 = np.asarray(spec.phase2_setpoints_y_phys, dtype=float)
    if p1.ndim != 2 or p2.ndim != 2:
        raise ValueError("phase setpoints must be 2-D arrays.")
    if p1.shape[0] <= 0 or p2.shape[0] <= 0:
        raise ValueError("each phase must define at least one setpoint.")
    if p1.shape[1] != p2.shape[1]:
        raise ValueError(f"phase setpoint output dimensions must match, got {p1.shape} and {p2.shape}.")
    if not np.all(np.isfinite(p1)) or not np.all(np.isfinite(p2)):
        raise ValueError("phase setpoints must contain finite values.")


def episode_len_from_spec(spec: TwoPhaseExperimentSpec) -> int:
    _validate_spec(spec)
    return int(spec.set_points_len) * int(np.asarray(spec.phase1_setpoints_y_phys).shape[0])


def phase2_steps_from_spec(spec: TwoPhaseExperimentSpec) -> int:
    _validate_spec(spec)
    if spec.phase2_steps is not None:
        return int(spec.phase2_steps)
    return int(spec.phase2_episodes) * episode_len_from_spec(spec)


def _phase_setpoint_steps(
    setpoints_y_phys: np.ndarray,
    episodes: int,
    set_points_len: int,
    episode_len: int,
) -> np.ndarray:
    if int(episodes) <= 0:
        return np.empty((0, int(setpoints_y_phys.shape[1])), dtype=float)
    blocks = [np.repeat(row.reshape(1, -1), int(set_points_len), axis=0) for row in setpoints_y_phys]
    cycle = np.concatenate(blocks, axis=0)
    if cycle.shape[0] == int(episode_len):
        episode_cycle = cycle
    elif len(setpoints_y_phys) == 1:
        episode_cycle = np.repeat(setpoints_y_phys[0].reshape(1, -1), int(episode_len), axis=0)
    else:
        raise ValueError(
            "phase setpoint cycle length must match episode_len unless the phase has one held setpoint; "
            f"got cycle length {cycle.shape[0]} and episode_len {episode_len}."
        )
    return np.concatenate([episode_cycle.copy() for _ in range(int(episodes))], axis=0)


def _fixed_duration_setpoint_steps(
    setpoints_y_phys: np.ndarray,
    total_steps: int,
    set_points_len: int,
) -> np.ndarray:
    if int(total_steps) <= 0:
        return np.empty((0, int(setpoints_y_phys.shape[1])), dtype=float)
    blocks = [np.repeat(row.reshape(1, -1), int(set_points_len), axis=0) for row in setpoints_y_phys]
    cycle = np.concatenate(blocks, axis=0)
    repeats = int(np.ceil(int(total_steps) / float(cycle.shape[0])))
    return np.concatenate([cycle.copy() for _ in range(repeats)], axis=0)[: int(total_steps)]


def _phase_indices(spec: TwoPhaseExperimentSpec) -> dict[str, np.ndarray]:
    phase1_episode_len = episode_len_from_spec(spec)
    reporting_window_steps = int(spec.reporting_window_steps)
    n1 = int(spec.phase1_episodes) * phase1_episode_len
    n2 = phase2_steps_from_spec(spec)
    total_steps = n1 + n2
    phase_id = np.concatenate(
        [
            np.ones(n1, dtype=np.int32),
            np.full(n2, 2, dtype=np.int32),
        ],
        axis=0,
    )
    step = np.arange(total_steps, dtype=np.int64)
    report_window = (step // reporting_window_steps) + 1
    step_in_report_window = step % reporting_window_steps
    phase_step = np.empty(total_steps, dtype=np.int64)
    phase_step[:n1] = step[:n1]
    phase_step[n1:] = step[n1:] - n1
    episode_in_phase = np.empty(total_steps, dtype=np.int32)
    episode_in_phase[:n1] = (step[:n1] // reporting_window_steps) + 1
    episode_in_phase[n1:] = (phase_step[n1:] // reporting_window_steps) + 1
    phase1_episode = np.zeros(total_steps, dtype=np.int32)
    phase1_episode[:n1] = (step[:n1] // phase1_episode_len) + 1
    phase2_report_window = np.zeros(total_steps, dtype=np.int32)
    phase2_report_window[n1:] = episode_in_phase[n1:]
    return {
        "phase_id": phase_id,
        "episode": report_window.astype(np.int32),
        "episode_in_phase": episode_in_phase,
        "step_in_episode": step_in_report_window.astype(np.int32),
        "report_window": report_window.astype(np.int32),
        "step_in_report_window": step_in_report_window.astype(np.int32),
        "phase_step": phase_step.astype(np.int32),
        "phase1_episode": phase1_episode,
        "phase2_report_window": phase2_report_window,
    }


def _disturbance_profile(spec: TwoPhaseExperimentSpec) -> dict[str, np.ndarray]:
    phase1_episode_len = episode_len_from_spec(spec)
    n1 = int(spec.phase1_episodes) * phase1_episode_len
    n2 = phase2_steps_from_spec(spec)

    d0 = np.array([spec.nominal_qi, spec.nominal_qs, spec.nominal_ha], dtype=float)
    d1 = np.array(
        [
            spec.nominal_qi * spec.phase1_qi_multiplier,
            spec.nominal_qs * spec.phase1_qs_multiplier,
            spec.nominal_ha * spec.phase1_ha_multiplier,
        ],
        dtype=float,
    )
    d2 = np.array(
        [
            spec.nominal_qi * spec.phase2_qi_multiplier,
            spec.nominal_qs * spec.phase2_qs_multiplier,
            spec.nominal_ha * spec.phase2_ha_multiplier,
        ],
        dtype=float,
    )

    phase1 = np.column_stack([np.linspace(d0[i], d1[i], n1) for i in range(3)])
    phase2 = np.column_stack([np.linspace(d1[i], d2[i], n2) for i in range(3)])
    values = np.vstack([phase1, phase2])
    return {
        "qi": values[:, 0].copy(),
        "qs": values[:, 1].copy(),
        "ha": values[:, 2].copy(),
    }


def build_two_phase_profiles(
    *,
    spec: TwoPhaseExperimentSpec,
    data_min: np.ndarray,
    data_max: np.ndarray,
    steady_outputs: np.ndarray,
    n_inputs: int,
) -> dict[str, Any]:
    _validate_spec(spec)
    n_inputs = int(n_inputs)
    phase1_episode_len = episode_len_from_spec(spec)
    phase2_steps = phase2_steps_from_spec(spec)
    reporting_window_steps = int(spec.reporting_window_steps)
    total_steps = int(spec.phase1_episodes) * phase1_episode_len + phase2_steps
    if total_steps % reporting_window_steps != 0:
        raise ValueError(
            "total two-phase profile length must be divisible by reporting_window_steps; "
            f"got total_steps={total_steps} and reporting_window_steps={reporting_window_steps}."
        )
    y_phys = np.vstack(
        [
            _phase_setpoint_steps(
                np.asarray(spec.phase1_setpoints_y_phys, dtype=float),
                int(spec.phase1_episodes),
                int(spec.set_points_len),
                phase1_episode_len,
            ),
            _phase_setpoint_steps(
                np.asarray(spec.phase2_setpoints_y_phys, dtype=float),
                int(spec.phase2_episodes),
                int(spec.set_points_len),
                phase1_episode_len,
            )
            if spec.phase2_episodes is not None
            else _fixed_duration_setpoint_steps(
                np.asarray(spec.phase2_setpoints_y_phys, dtype=float),
                int(phase2_steps),
                int(spec.set_points_len),
            ),
        ]
    )
    y_scaled = apply_min_max(y_phys, data_min[n_inputs:], data_max[n_inputs:]) - apply_min_max(
        np.asarray(steady_outputs, dtype=float),
        data_min[n_inputs:],
        data_max[n_inputs:],
    )
    idx = _phase_indices(spec)
    phase1_steps = int(spec.phase1_episodes) * phase1_episode_len
    total_steps = y_phys.shape[0]
    total_reporting_windows = total_steps // reporting_window_steps
    phase1_reporting_windows = phase1_steps // reporting_window_steps
    phase2_reporting_windows = phase2_steps // reporting_window_steps
    phase_windows = [
        {
            "name": "phase1_learning",
            "phase_id": 1,
            "episode_start": 1,
            "episode_end": int(phase1_reporting_windows),
            "learning_episode_start": 1,
            "learning_episode_end": int(spec.phase1_episodes),
            "report_window_start": 1,
            "report_window_end": int(phase1_reporting_windows),
            "step_start": 0,
            "step_end_exclusive": phase1_steps,
        },
    ]
    if phase2_steps > 0:
        phase_windows.extend(
            [
                {
                    "name": "phase2_immediate",
                    "phase_id": 2,
                    "episode_start": int(phase1_reporting_windows) + 1,
                    "episode_end": min(int(phase1_reporting_windows) + 5, int(total_reporting_windows)),
                    "report_window_start": int(phase1_reporting_windows) + 1,
                    "report_window_end": min(int(phase1_reporting_windows) + 5, int(total_reporting_windows)),
                    "phase2_report_window_start": 1,
                    "phase2_report_window_end": min(5, int(phase2_reporting_windows)),
                    "step_start": phase1_steps,
                    "step_end_exclusive": min(phase1_steps + 5 * reporting_window_steps, total_steps),
                },
                {
                    "name": "phase2_full",
                    "phase_id": 2,
                    "episode_start": int(phase1_reporting_windows) + 1,
                    "episode_end": int(total_reporting_windows),
                    "report_window_start": int(phase1_reporting_windows) + 1,
                    "report_window_end": int(total_reporting_windows),
                    "phase2_report_window_start": 1,
                    "phase2_report_window_end": int(phase2_reporting_windows),
                    "step_start": phase1_steps,
                    "step_end_exclusive": total_steps,
                },
                {
                    "name": "phase2_final",
                    "phase_id": 2,
                    "episode_start": max(int(phase1_reporting_windows) + 1, int(total_reporting_windows) - 9),
                    "episode_end": int(total_reporting_windows),
                    "report_window_start": max(int(phase1_reporting_windows) + 1, int(total_reporting_windows) - 9),
                    "report_window_end": int(total_reporting_windows),
                    "phase2_report_window_start": max(1, int(phase2_reporting_windows) - 9),
                    "phase2_report_window_end": int(phase2_reporting_windows),
                    "step_start": max(phase1_steps, total_steps - 10 * reporting_window_steps),
                    "step_end_exclusive": total_steps,
                },
            ]
        )
    return {
        "spec": spec,
        "total_episodes": int(total_reporting_windows),
        "total_reporting_windows": int(total_reporting_windows),
        "phase1_episodes": int(spec.phase1_episodes),
        "phase2_episodes": None if spec.phase2_episodes is None else int(spec.phase2_episodes),
        "phase1_episode_len": int(phase1_episode_len),
        "phase1_reporting_windows": int(phase1_reporting_windows),
        "phase2_steps": int(phase2_steps),
        "phase2_reporting_windows": int(phase2_reporting_windows),
        "reporting_window_steps": int(reporting_window_steps),
        "episode_len": int(reporting_window_steps),
        "phase1_steps": int(phase1_steps),
        "total_steps": int(total_steps),
        "setpoint_profile_phys": y_phys,
        "setpoint_profile_scaled_dev": y_scaled,
        "disturbance_profile": _disturbance_profile(spec),
        "phase_id": idx["phase_id"],
        "phase_name": np.where(idx["phase_id"] == 1, "phase1_learning", "phase2_continuation"),
        "episode": idx["episode"],
        "episode_in_phase": idx["episode_in_phase"],
        "step_in_episode": idx["step_in_episode"],
        "report_window": idx["report_window"],
        "step_in_report_window": idx["step_in_report_window"],
        "phase_step": idx["phase_step"],
        "phase1_episode": idx["phase1_episode"],
        "phase2_report_window": idx["phase2_report_window"],
        "phase_windows": phase_windows,
        "phase_boundary_steps": [int(phase1_steps)] if phase2_steps > 0 else [],
    }


def profile_metadata_for_step(profile: dict[str, Any] | None, step_idx: int) -> dict[str, Any]:
    if profile is None:
        return {}
    idx = int(step_idx)
    if idx < 0 or idx >= int(profile.get("total_steps", 0)):
        return {}
    phase_id = int(profile["phase_id"][idx])
    return {
        "phase_id": phase_id,
        "phase_name": str(profile["phase_name"][idx]),
        "episode": int(profile["episode"][idx]),
        "episode_in_phase": int(profile["episode_in_phase"][idx]),
        "step_in_episode": int(profile["step_in_episode"][idx]),
        "report_window": int(profile["report_window"][idx]),
        "step_in_report_window": int(profile["step_in_report_window"][idx]),
        "phase_step": int(profile["phase_step"][idx]),
        "phase1_episode": int(profile["phase1_episode"][idx]),
        "phase2_report_window": int(profile["phase2_report_window"][idx]),
    }


def jsonable_two_phase_profile(profile: dict[str, Any]) -> dict[str, Any]:
    spec = profile["spec"]
    return {
        "phase1_episodes": int(spec.phase1_episodes),
        "phase2_steps": int(phase2_steps_from_spec(spec)),
        "phase2_episodes": None if spec.phase2_episodes is None else int(spec.phase2_episodes),
        "set_points_len": int(spec.set_points_len),
        "reporting_window_steps": int(spec.reporting_window_steps),
        "phase1_setpoints_y_phys": np.asarray(spec.phase1_setpoints_y_phys, dtype=float).tolist(),
        "phase2_setpoints_y_phys": np.asarray(spec.phase2_setpoints_y_phys, dtype=float).tolist(),
        "nominal": {
            "qi": float(spec.nominal_qi),
            "qs": float(spec.nominal_qs),
            "ha": float(spec.nominal_ha),
        },
        "phase1_final": {
            "qi": float(spec.nominal_qi * spec.phase1_qi_multiplier),
            "qs": float(spec.nominal_qs * spec.phase1_qs_multiplier),
            "ha": float(spec.nominal_ha * spec.phase1_ha_multiplier),
        },
        "phase2_final": (
            None
            if int(profile.get("phase2_steps", 0)) <= 0
            else {
                "qi": float(spec.nominal_qi * spec.phase2_qi_multiplier),
                "qs": float(spec.nominal_qs * spec.phase2_qs_multiplier),
                "ha": float(spec.nominal_ha * spec.phase2_ha_multiplier),
            }
        ),
        "episode_len": int(profile["episode_len"]),
        "phase1_episode_len": int(profile["phase1_episode_len"]),
        "total_reporting_windows": int(profile["total_reporting_windows"]),
        "phase1_reporting_windows": int(profile["phase1_reporting_windows"]),
        "phase2_reporting_windows": int(profile["phase2_reporting_windows"]),
        "phase1_steps": int(profile["phase1_steps"]),
        "total_steps": int(profile["total_steps"]),
        "phase_windows": list(profile["phase_windows"]),
        "phase_boundary_steps": list(profile["phase_boundary_steps"]),
    }
