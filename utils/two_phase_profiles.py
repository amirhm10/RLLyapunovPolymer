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
        [3.35, 323.5],
    ],
    dtype=float,
)


@dataclass(frozen=True)
class TwoPhaseExperimentSpec:
    phase1_episodes: int = 200
    phase2_episodes: int = 50
    set_points_len: int = 400
    phase1_setpoints_y_phys: np.ndarray = field(default_factory=lambda: PHASE1_SETPOINT_Y_PHYS.copy())
    phase2_setpoints_y_phys: np.ndarray = field(default_factory=lambda: PHASE2_SETPOINT_Y_PHYS.copy())
    nominal_qi: float = 108.0
    nominal_qs: float = 459.0
    nominal_ha: float = 1.05e6
    phase1_qi_multiplier: float = 0.95
    phase1_qs_multiplier: float = 1.05
    phase1_ha_multiplier: float = 0.92
    phase2_qi_multiplier: float = 1.05
    phase2_qs_multiplier: float = 0.95
    phase2_ha_multiplier: float = 0.88


def _validate_spec(spec: TwoPhaseExperimentSpec) -> None:
    if int(spec.phase1_episodes) <= 0:
        raise ValueError("phase1_episodes must be positive.")
    if int(spec.phase2_episodes) <= 0:
        raise ValueError("phase2_episodes must be positive.")
    if int(spec.set_points_len) <= 0:
        raise ValueError("set_points_len must be positive.")
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


def _phase_setpoint_steps(
    setpoints_y_phys: np.ndarray,
    episodes: int,
    set_points_len: int,
    episode_len: int,
) -> np.ndarray:
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


def _phase_indices(spec: TwoPhaseExperimentSpec) -> dict[str, np.ndarray]:
    episode_len = episode_len_from_spec(spec)
    n1 = int(spec.phase1_episodes) * episode_len
    n2 = int(spec.phase2_episodes) * episode_len
    phase_id = np.concatenate(
        [
            np.ones(n1, dtype=np.int32),
            np.full(n2, 2, dtype=np.int32),
        ],
        axis=0,
    )
    episode = (np.arange(n1 + n2, dtype=np.int64) // episode_len) + 1
    step_in_episode = np.arange(n1 + n2, dtype=np.int64) % episode_len
    episode_in_phase = np.empty(n1 + n2, dtype=np.int32)
    episode_in_phase[:n1] = episode[:n1]
    episode_in_phase[n1:] = episode[n1:] - int(spec.phase1_episodes)
    return {
        "phase_id": phase_id,
        "episode": episode.astype(np.int32),
        "episode_in_phase": episode_in_phase,
        "step_in_episode": step_in_episode.astype(np.int32),
    }


def _disturbance_profile(spec: TwoPhaseExperimentSpec) -> dict[str, np.ndarray]:
    episode_len = episode_len_from_spec(spec)
    n1 = int(spec.phase1_episodes) * episode_len
    n2 = int(spec.phase2_episodes) * episode_len

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
    episode_len = episode_len_from_spec(spec)
    y_phys = np.vstack(
        [
            _phase_setpoint_steps(
                np.asarray(spec.phase1_setpoints_y_phys, dtype=float),
                int(spec.phase1_episodes),
                int(spec.set_points_len),
                episode_len,
            ),
            _phase_setpoint_steps(
                np.asarray(spec.phase2_setpoints_y_phys, dtype=float),
                int(spec.phase2_episodes),
                int(spec.set_points_len),
                episode_len,
            ),
        ]
    )
    y_scaled = apply_min_max(y_phys, data_min[n_inputs:], data_max[n_inputs:]) - apply_min_max(
        np.asarray(steady_outputs, dtype=float),
        data_min[n_inputs:],
        data_max[n_inputs:],
    )
    idx = _phase_indices(spec)
    phase1_steps = int(spec.phase1_episodes) * episode_len
    total_steps = y_phys.shape[0]
    phase_windows = [
        {
            "name": "phase1_learning",
            "phase_id": 1,
            "episode_start": 1,
            "episode_end": int(spec.phase1_episodes),
            "step_start": 0,
            "step_end_exclusive": phase1_steps,
        },
        {
            "name": "phase2_immediate",
            "phase_id": 2,
            "episode_start": int(spec.phase1_episodes) + 1,
            "episode_end": min(int(spec.phase1_episodes) + 5, int(spec.phase1_episodes + spec.phase2_episodes)),
            "step_start": phase1_steps,
            "step_end_exclusive": min(phase1_steps + 5 * episode_len, total_steps),
        },
        {
            "name": "phase2_full",
            "phase_id": 2,
            "episode_start": int(spec.phase1_episodes) + 1,
            "episode_end": int(spec.phase1_episodes + spec.phase2_episodes),
            "step_start": phase1_steps,
            "step_end_exclusive": total_steps,
        },
        {
            "name": "phase2_final",
            "phase_id": 2,
            "episode_start": max(int(spec.phase1_episodes) + 1, int(spec.phase1_episodes + spec.phase2_episodes) - 9),
            "episode_end": int(spec.phase1_episodes + spec.phase2_episodes),
            "step_start": max(phase1_steps, total_steps - 10 * episode_len),
            "step_end_exclusive": total_steps,
        },
    ]
    return {
        "spec": spec,
        "total_episodes": int(spec.phase1_episodes + spec.phase2_episodes),
        "episode_len": int(episode_len),
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
        "phase_windows": phase_windows,
        "phase_boundary_steps": [int(phase1_steps)],
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
    }


def jsonable_two_phase_profile(profile: dict[str, Any]) -> dict[str, Any]:
    spec = profile["spec"]
    return {
        "phase1_episodes": int(spec.phase1_episodes),
        "phase2_episodes": int(spec.phase2_episodes),
        "set_points_len": int(spec.set_points_len),
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
        "phase2_final": {
            "qi": float(spec.nominal_qi * spec.phase2_qi_multiplier),
            "qs": float(spec.nominal_qs * spec.phase2_qs_multiplier),
            "ha": float(spec.nominal_ha * spec.phase2_ha_multiplier),
        },
        "episode_len": int(profile["episode_len"]),
        "phase1_steps": int(profile["phase1_steps"]),
        "total_steps": int(profile["total_steps"]),
        "phase_windows": list(profile["phase_windows"]),
        "phase_boundary_steps": list(profile["phase_boundary_steps"]),
    }
