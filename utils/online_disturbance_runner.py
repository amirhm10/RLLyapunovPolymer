from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import time
from dataclasses import asdict, dataclass, is_dataclass, replace
from datetime import datetime
from pathlib import Path
from pprint import pprint
from typing import Any

import numpy as np
import torch

from TD3Agent.agent import TD3Agent
from TD3Agent.reward_functions import make_reward_fn_relative_QR
from Simulation.mpc import MpcSolver
from Simulation.run_rl_lyapunov import run_rl_train
from Lyapunov.direct_lyapunov_mpc import (
    build_direct_lyapunov_run_bundle,
    design_direct_lyapunov_mpc_solver,
    make_direct_lyapunov_comparison_record,
    run_direct_output_disturbance_lyapunov_mpc,
    run_offset_free_mpc_with_direct_diagnostics,
    save_direct_lyapunov_comparison_artifacts,
    save_direct_lyapunov_debug_artifacts,
)
from Lyapunov.safety_debug import (
    build_safety_filter_run_bundle,
    make_safety_filter_comparison_record,
    save_safety_filter_comparison_artifacts,
    save_safety_filter_debug_artifacts,
)
from utils.direct_lyapunov_study import (
    DIRECT_DISTURBANCE_N_TESTS,
    DIRECT_DISTURBANCE_SEED,
    DIRECT_DISTURBANCE_SETPOINT_LEN,
    DIRECT_TWO_SETPOINT_Y_PHYS,
    direct_disturbance_test_cycle,
)
from utils.lmpc_td3_workflow import (
    latest_lmpc_pretrained_checkpoint,
    resolve_checkpoint_layers,
)
from utils.of_mpc_td3_workflow import (
    CONTROL_HORIZON,
    HA_CHANGE,
    NOMINAL_HA,
    NOMINAL_QI,
    NOMINAL_QS,
    PREDICT_HORIZON,
    QI_CHANGE,
    QS_CHANGE,
    TD3Dimensions,
    U_MAX_PHYS,
    U_MIN_PHYS,
    build_polymer_setup,
    compute_td3_dimensions,
    latest_pretrained_checkpoint as latest_of_mpc_pretrained_checkpoint,
    load_of_mpc_system_data,
    make_observer_gain,
    make_polymer_system,
    set_seed,
)
from utils.direct_lmpc_selector_defaults import (
    DIRECT_LMPC_LYAP_TOL,
    DIRECT_LMPC_RHO_LYAP,
    DIRECT_LMPC_SLACK_PENALTY,
    DIRECT_LMPC_TARGET_MODE,
    DIRECT_LMPC_TARGET_SELECTOR_VARIANT,
    DIRECT_LMPC_U_REF_WEIGHT,
    DIRECT_LMPC_X_REF_WEIGHT,
    make_direct_lmpc_target_config,
)
from utils.gart_defaults import (
    GART_FINAL_LYAPUNOV_MODE,
    GART_FINAL_LYAP_EPS,
    GART_FINAL_MPC_OBJECTIVE,
    GART_FINAL_RHO_LYAP,
    GART_FINAL_SLACK_PENALTY,
    GART_FINAL_TARGET_CONFIG_OVERRIDES,
    GART_FINAL_TARGET_OVERRIDES,
    discover_gart_case_values,
    make_gart_mpc_config,
    make_gart_target_config,
)
from utils.path_helpers import repo_path, resolve_repo_path
from utils.polymer_td3_defaults import DEFAULT_TD3_SETPOINT_SCALER_Y_PHYS
from utils.scaling_helpers import apply_min_max


PLANT_MODE = "disturb"
DISTURBANCE_AFTER_STEP = False
FORCE_FINAL_TEST = False
USE_TARGET_OUTPUT_FOR_TRACKING = False

PREDICT_H = PREDICT_HORIZON
CONT_H = CONTROL_HORIZON
RHO_LYAP = DIRECT_LMPC_RHO_LYAP
LYAP_EPS = 1e-4
LYAP_TOL = DIRECT_LMPC_LYAP_TOL
SLACK_PENALTY = DIRECT_LMPC_SLACK_PENALTY
TARGET_MODE = DIRECT_LMPC_TARGET_MODE
TARGET_SELECTOR_VARIANT = DIRECT_LMPC_TARGET_SELECTOR_VARIANT
GART_LMPC_OBJECTIVE = GART_FINAL_MPC_OBJECTIVE
GART_LMPC_LYAPUNOV_MODE = GART_FINAL_LYAPUNOV_MODE

QY_MPC_DIAG = np.array([5.0, 1.0], dtype=float)
SU_MPC_DIAG = np.array([1.0, 1.0], dtype=float)
RDU_MPC_DIAG = np.array([1.0, 1.0], dtype=float)
QY_REWARD_DIAG = np.array([12.0, 6.0], dtype=float)
RDU_REWARD_DIAG = np.array([1.0, 1.0], dtype=float)

U_PREV_PENALTY_WEIGHT = DIRECT_LMPC_U_REF_WEIGHT
XS_PREV_PENALTY_WEIGHT = DIRECT_LMPC_X_REF_WEIGHT

DEFAULT_ACTOR_LAYER_SIZES = (256, 256, 256)
DEFAULT_CRITIC_LAYER_SIZES = (256, 256, 256)
BUFFER_CAPACITY = 40000
ACTOR_LR = 5e-5
CRITIC_LR = 5e-4
GAMMA = 0.99
TAU = 0.005
MAX_ACTION = 1.0
POLICY_DELAY = 2
BATCH_SIZE = 256
STD_START = 0.0
STD_END = 0.005
STD_DECAY_RATE = 0.99992
STD_DECAY_MODE = "exp"

PRETRAINED_SMOOTHING_STD = 0.01
COLD_START_SMOOTHING_STD = 0.1
NOISE_CLIP = 0.01
PRETRAINED_EXPLORATION_STD_START = 0.05
COLD_START_EXPLORATION_STD_START = 0.1
FULL_RL_EXPLORATION_STD_END = 0.02
PRETRAINED_BC_EXPLORATION_STD = 0.05
COLD_START_BC_EXPLORATION_STD = 0.05
PRETRAINED_HANDOFF_EXPLORATION_STD_START = 0.0
PRETRAINED_HANDOFF_EXPLORATION_STD_END = 0.05
COLD_START_HANDOFF_EXPLORATION_STD_START = 0.0
COLD_START_HANDOFF_EXPLORATION_STD_END = 0.05
DEFAULT_RESET_PRETRAINED_CRITIC = True

WARMUP_EPISODES = 0
BC_TEACHER_EPISODES = 20
HANDOFF_EPISODES = 5
PRETRAINED_HANDOFF_EPISODES = 10
COLD_START_HANDOFF_EPISODES = HANDOFF_EPISODES
NOISY_TEACHER_EPISODES = 10
NOISY_TEACHER_HANDOFF_EPISODES = 10

STANDARD_RL_OBSERVATION_MODE = "standard"
GART_RL_OBSERVATION_MODE = "gart"
DIRECT_GATE_PROJECTION_BACKEND = "direct_accept_or_fallback"
MPC_ONLY_DIAGNOSTIC_BACKEND = "mpc_only_diagnostic"
DEFAULT_SECTION16_CERT_MARGIN_SCALE = 1.0
DEFAULT_SECTION16_CERT_SIGMA_FLOOR = 0.0


@dataclass(frozen=True)
class OnlineTD3Preset:
    key: str
    study_name: str
    label: str
    safety_gate: bool
    pretrain_source: str | None
    teacher_source: str
    direct_target_mode: str = TARGET_MODE
    fallback_controller: str = "direct_lyapunov_mpc"


@dataclass(frozen=True)
class DisturbanceContext:
    setup: Any
    system_data: dict[str, Any]
    dimensions: TD3Dimensions
    y_sp_scenario: np.ndarray
    observer_gain: np.ndarray
    lmpc_obj: Any
    of_mpc_obj: MpcSolver
    ic_opt_template: np.ndarray
    bnds: tuple[tuple[float, float], ...]
    cons: tuple
    u_dev_min: np.ndarray
    u_dev_max: np.ndarray
    reward_config: dict[str, Any]
    reward_fn: Any
    target_config: Any
    gart_discovered: dict[str, Any] | None = None


ONLINE_TD3_PRESETS: dict[str, OnlineTD3Preset] = {
    "lmpc_pretrained_safety_gate": OnlineTD3Preset(
        key="lmpc_pretrained_safety_gate",
        study_name="OnlineTD3_LMPCPretrained_SafetyGate",
        label="LMPC-pretrained online TD3 with Direct LMPC safety gate",
        safety_gate=True,
        pretrain_source="lmpc",
        teacher_source="direct_lyapunov_mpc",
    ),
    "ofmpc_pretrained_safety_gate": OnlineTD3Preset(
        key="ofmpc_pretrained_safety_gate",
        study_name="OnlineTD3_OFMPCPretrained_SafetyGate",
        label="OF-MPC-pretrained online TD3 with GART-LMPC safety gate",
        safety_gate=True,
        pretrain_source="of_mpc",
        teacher_source="gart_lmpc",
        direct_target_mode="gart",
        fallback_controller="gart_lmpc",
    ),
    "lmpc_pretrained_no_safety_gate": OnlineTD3Preset(
        key="lmpc_pretrained_no_safety_gate",
        study_name="OnlineTD3_LMPCPretrained_NoSafetyGate",
        label="LMPC-pretrained online TD3 without safety intervention",
        safety_gate=False,
        pretrain_source="lmpc",
        teacher_source="offset_free_mpc",
    ),
    "ofmpc_pretrained_no_safety_gate": OnlineTD3Preset(
        key="ofmpc_pretrained_no_safety_gate",
        study_name="OnlineTD3_OFMPCPretrained_NoSafetyGate",
        label="OF-MPC-pretrained online TD3 without safety intervention and GART-LMPC BC",
        safety_gate=False,
        pretrain_source="of_mpc",
        teacher_source="gart_lmpc",
        direct_target_mode="gart",
        fallback_controller="none",
    ),
    "cold_start_safety_gate": OnlineTD3Preset(
        key="cold_start_safety_gate",
        study_name="OnlineTD3_ColdStart_SafetyGate",
        label="Cold-start online TD3 with GART-LMPC safety gate",
        safety_gate=True,
        pretrain_source=None,
        teacher_source="gart_lmpc",
        direct_target_mode="gart",
        fallback_controller="gart_lmpc",
    ),
    "cold_start_no_safety_gate": OnlineTD3Preset(
        key="cold_start_no_safety_gate",
        study_name="OnlineTD3_ColdStart_NoSafetyGate",
        label="Cold-start online TD3 without safety intervention and GART-LMPC BC",
        safety_gate=False,
        pretrain_source=None,
        teacher_source="gart_lmpc",
        direct_target_mode="gart",
        fallback_controller="none",
    ),
}


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    if isinstance(value, Path):
        return os.fspath(value)
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(_jsonable(payload), handle, indent=2)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _jsonable(value) for key, value in row.items()})


def _link_or_copy(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _mirror_primary_debug_files(debug_dir: str | Path, study_root: Path) -> None:
    debug_path = Path(debug_dir)
    for filename in ("summary.json", "step_table.csv", "episode_table.csv", "arrays.npz"):
        _link_or_copy(debug_path / filename, study_root / filename)


def _episode_records_from_direct_bundle(bundle: dict[str, Any]) -> list[dict[str, Any]]:
    n_steps = int(bundle.get("nFE", 0))
    episode_len = int(bundle.get("time_in_sub_episodes", 0))
    rewards = np.asarray(bundle.get("rewards", []), dtype=float).reshape(-1)
    info_storage = list(bundle.get("direct_info_storage", []))
    if episode_len <= 0:
        episode_len = max(n_steps, 1)

    rows: list[dict[str, Any]] = []
    n_episodes = int(np.ceil(n_steps / float(episode_len))) if n_steps > 0 else 0
    for episode_idx in range(n_episodes):
        start = int(episode_idx * episode_len)
        stop = int(min((episode_idx + 1) * episode_len, n_steps))
        infos = info_storage[start:stop]
        reward_slice = rewards[start:stop]

        def count_true(*keys: str) -> int:
            total = 0
            for info in infos:
                if any(bool(info.get(key, False)) for key in keys):
                    total += 1
            return total

        rows.append(
            {
                "episode": episode_idx + 1,
                "step_start": start,
                "step_end_exclusive": stop,
                "n_steps": stop - start,
                "reward_mean": None if reward_slice.size == 0 else float(np.nanmean(reward_slice)),
                "reward_sum": None if reward_slice.size == 0 else float(np.nansum(reward_slice)),
                "solver_success_count": count_true("success", "solver_success"),
                "target_success_count": count_true("target_success"),
                "hard_contraction_count": count_true(
                    "hard_contraction_satisfied",
                    "first_step_contraction_satisfied",
                    "final_lyap_ok",
                ),
                "diagnostic_unsafe_count": count_true("diagnostic_unsafe"),
                "actual_intervention_count": count_true("actual_intervention_active", "actual_intervention"),
            }
        )
    return rows


def _write_direct_episode_table(debug_dir: str | Path, bundle: dict[str, Any]) -> Path:
    path = Path(debug_dir) / "episode_table.csv"
    _write_csv(path, _episode_records_from_direct_bundle(bundle))
    return path


def _td3_online_hparams(agent: TD3Agent) -> dict[str, Any]:
    return {
        "gamma": float(agent.gamma),
        "actor_lr": float(agent.actor_lr),
        "critic_lr": float(agent.critic_lr),
        "batch_size": int(agent.batch_size),
        "policy_delay": int(agent.policy_delay),
        "target_policy_smoothing_noise_std": float(agent.t_std),
        "noise_clip": float(agent.noise_clip),
        "tau": float(agent.tau),
        "max_action": float(agent.max_action),
        "actor_hidden": list(agent.actor_hidden),
        "critic_hidden": list(agent.critic_hidden),
    }


def normalize_rl_observation_mode(mode: str | None) -> str:
    if mode is None:
        return STANDARD_RL_OBSERVATION_MODE
    mode = str(mode).strip().lower()
    aliases = {
        "standard": STANDARD_RL_OBSERVATION_MODE,
        "default": STANDARD_RL_OBSERVATION_MODE,
        "legacy": STANDARD_RL_OBSERVATION_MODE,
        "td3": STANDARD_RL_OBSERVATION_MODE,
        "gart": GART_RL_OBSERVATION_MODE,
        "section16": GART_RL_OBSERVATION_MODE,
        "gart_section16": GART_RL_OBSERVATION_MODE,
    }
    if mode not in aliases:
        raise ValueError("RL_OBSERVATION_MODE must be 'standard' or 'gart'.")
    return aliases[mode]


def gart_observation_state_dim(dimensions: TD3Dimensions) -> int:
    n_aug = int(dimensions.state_dim) - int(dimensions.set_points_number) - int(dimensions.inputs_number)
    return int(n_aug + 4 * int(dimensions.set_points_number) + 2 * int(dimensions.inputs_number))


def dimensions_for_observation_mode(dimensions: TD3Dimensions, mode: str | None) -> TD3Dimensions:
    normalized = normalize_rl_observation_mode(mode)
    if normalized == GART_RL_OBSERVATION_MODE:
        return replace(dimensions, state_dim=gart_observation_state_dim(dimensions))
    return dimensions


def _normalize_projection_backend_override(value: str | None, *, safety_gate: bool) -> str:
    if value is None:
        return DIRECT_GATE_PROJECTION_BACKEND if safety_gate else MPC_ONLY_DIAGNOSTIC_BACKEND
    backend = str(value).strip().lower()
    aliases = {
        "direct_accept_or_fallback": DIRECT_GATE_PROJECTION_BACKEND,
        "direct_gate": DIRECT_GATE_PROJECTION_BACKEND,
        "mpc_only": MPC_ONLY_DIAGNOSTIC_BACKEND,
        "mpc_only_diagnostic": MPC_ONLY_DIAGNOSTIC_BACKEND,
    }
    if backend not in aliases:
        raise ValueError(
            "Online safety-gate presets no longer support Section-16 QCQP projection. "
            "Use PROJECTION_BACKEND='direct_accept_or_fallback' for GART-LMPC fallback "
            "or 'mpc_only_diagnostic' for no-gate diagnostics."
        )
    return aliases[backend]


def _scaling_contract(setup: Any, context: DisturbanceContext) -> dict[str, Any]:
    system_data = context.system_data
    min_max_dict = system_data["min_max_dict"]
    n_inputs = int(context.dimensions.inputs_number)
    data_min = system_data["data_min"]
    data_max = system_data["data_max"]
    y_ss_scaled = apply_min_max(setup.steady_states["y_ss"], data_min[n_inputs:], data_max[n_inputs:])
    expected_range = DEFAULT_TD3_SETPOINT_SCALER_Y_PHYS.copy()
    actual_range = np.asarray(system_data.get("setpoint_range_y_used"), dtype=float)
    expected_range_scaled_dev = apply_min_max(expected_range, data_min[n_inputs:], data_max[n_inputs:]) - y_ss_scaled
    expected_y_sp_min = np.min(expected_range_scaled_dev, axis=0)
    expected_y_sp_max = np.max(expected_range_scaled_dev, axis=0)

    if not np.allclose(actual_range, expected_range, rtol=0.0, atol=1e-12):
        raise RuntimeError(
            "TD3 setpoint feature scaler mismatch. Online TD3 checkpoints expect "
            f"{expected_range.tolist()}, got {actual_range.tolist()}."
        )
    if not np.allclose(min_max_dict["y_sp_min"], expected_y_sp_min, rtol=0.0, atol=1e-10):
        raise RuntimeError("TD3 y_sp_min does not match the expected broad setpoint scaler envelope.")
    if not np.allclose(min_max_dict["y_sp_max"], expected_y_sp_max, rtol=0.0, atol=1e-10):
        raise RuntimeError("TD3 y_sp_max does not match the expected broad setpoint scaler envelope.")
    if np.any(context.y_sp_scenario < min_max_dict["y_sp_min"] - 1e-10) or np.any(
        context.y_sp_scenario > min_max_dict["y_sp_max"] + 1e-10
    ):
        raise RuntimeError("Online rollout setpoints are outside the TD3 setpoint feature scaler envelope.")

    return {
        "state_bounds_source": system_data.get("state_bounds_source"),
        "setpoint_bounds_source": system_data.get("setpoint_bounds_source"),
        "td3_setpoint_scaler_y_phys": actual_range.copy(),
        "rollout_setpoint_y_phys": DIRECT_TWO_SETPOINT_Y_PHYS.copy(),
        "y_sp_min": np.asarray(min_max_dict["y_sp_min"], dtype=float).copy(),
        "y_sp_max": np.asarray(min_max_dict["y_sp_max"], dtype=float).copy(),
        "rollout_y_sp_scaled_deviation": np.asarray(context.y_sp_scenario, dtype=float).copy(),
        "u_min_dev": np.asarray(min_max_dict["u_min"], dtype=float).copy(),
        "u_max_dev": np.asarray(min_max_dict["u_max"], dtype=float).copy(),
    }


def _phase_plot_boundaries(
    episodes: int,
    set_points_len: int,
    training_phase_config: dict[str, Any] | None = None,
) -> np.ndarray:
    time_in_sub_episodes = int(DIRECT_TWO_SETPOINT_Y_PHYS.shape[0]) * int(set_points_len)
    phase_episodes = WARMUP_EPISODES + BC_TEACHER_EPISODES
    if training_phase_config is not None:
        phase_episodes = int(training_phase_config.get("warmup_buffer_only_episodes", 0)) + int(
            training_phase_config.get("behavior_clone_teacher_episodes", 0)
        )
    return np.array(
        [phase_episodes * time_in_sub_episodes],
        dtype=int,
    )


def _teacher_noise_std(pretrained: bool) -> float:
    return float(PRETRAINED_BC_EXPLORATION_STD if pretrained else COLD_START_BC_EXPLORATION_STD)


def _handoff_noise_std_end(pretrained: bool) -> float:
    return float(PRETRAINED_HANDOFF_EXPLORATION_STD_END if pretrained else COLD_START_HANDOFF_EXPLORATION_STD_END)


def noisy_teacher_buffer_warmup_overrides(
    *,
    teacher_source: str = "gart_lmpc",
    pretrained: bool = False,
    teacher_episodes: int = NOISY_TEACHER_EPISODES,
    handoff_episodes: int = NOISY_TEACHER_HANDOFF_EPISODES,
) -> dict[str, Any]:
    """Alternative A: noisy teacher rollout fills replay before TD3 updates."""
    noise_std = _teacher_noise_std(pretrained)
    return {
        "warmup_buffer_only_episodes": int(teacher_episodes),
        "warmup_behavior_source": teacher_source,
        "warmup_behavior_noise": "none" if noise_std <= 0.0 else "gaussian",
        "warmup_exploration_std": noise_std,
        "warmup_exploration_space": "input_dev",
        "behavior_clone_teacher_episodes": 0,
        "bc_teacher_policy": teacher_source,
        "bc_behavior_source": teacher_source,
        "bc_update_mode": "buffer_only",
        "bc_behavior_noise": "none",
        "bc_exploration_std": noise_std,
        "bc_exploration_space": "input_dev",
        "handoff_episodes": int(handoff_episodes),
        "handoff_update_mode": "td3_full",
        "handoff_actor_bc_updates_per_step": 0,
        "handoff_behavior_noise": "gaussian",
        "handoff_exploration_std_start": 0.0,
        "handoff_exploration_std_end": _handoff_noise_std_end(pretrained),
        "handoff_exploration_space": "input_dev",
        "full_rl_exploration_space": "input_dev",
    }


def noisy_teacher_critic_warmup_overrides(
    *,
    teacher_source: str = "gart_lmpc",
    pretrained: bool = False,
    teacher_episodes: int = NOISY_TEACHER_EPISODES,
    handoff_episodes: int = NOISY_TEACHER_HANDOFF_EPISODES,
) -> dict[str, Any]:
    """Alternative B: noisy teacher rollout trains critic only, with no actor BC."""
    noise_std = _teacher_noise_std(pretrained)
    return {
        "warmup_buffer_only_episodes": 0,
        "warmup_behavior_source": teacher_source,
        "warmup_behavior_noise": "none",
        "warmup_exploration_std": noise_std,
        "warmup_exploration_space": "input_dev",
        "behavior_clone_teacher_episodes": int(teacher_episodes),
        "bc_teacher_policy": teacher_source,
        "bc_behavior_source": teacher_source,
        "bc_update_mode": "critic_td_only",
        "bc_actor_updates_per_step": 1,
        "bc_behavior_noise": "none" if noise_std <= 0.0 else "gaussian",
        "bc_exploration_std": noise_std,
        "bc_exploration_space": "input_dev",
        "handoff_episodes": int(handoff_episodes),
        "handoff_update_mode": "td3_full",
        "handoff_actor_bc_updates_per_step": 0,
        "handoff_behavior_noise": "gaussian",
        "handoff_exploration_std_start": 0.0,
        "handoff_exploration_std_end": _handoff_noise_std_end(pretrained),
        "handoff_exploration_space": "input_dev",
        "full_rl_exploration_space": "input_dev",
    }


def default_noisy_teacher_critic_warmup_overrides(
    *,
    teacher_source: str = "gart_lmpc",
    pretrained: bool = False,
) -> dict[str, Any]:
    return noisy_teacher_critic_warmup_overrides(
        teacher_source=teacher_source,
        pretrained=pretrained,
        teacher_episodes=NOISY_TEACHER_EPISODES,
        handoff_episodes=NOISY_TEACHER_HANDOFF_EPISODES,
    )


def _training_phase_config(
    *,
    teacher_source: str,
    pretrained: bool,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if teacher_source not in {"direct_lyapunov_mpc", "offset_free_mpc", "gart_lmpc"}:
        raise ValueError(f"Unsupported teacher source: {teacher_source!r}")
    exploration_std = (
        PRETRAINED_EXPLORATION_STD_START
        if pretrained
        else COLD_START_EXPLORATION_STD_START
    )
    bc_exploration_std = (
        PRETRAINED_BC_EXPLORATION_STD
        if pretrained
        else COLD_START_BC_EXPLORATION_STD
    )
    handoff_exploration_std_start = (
        PRETRAINED_HANDOFF_EXPLORATION_STD_START
        if pretrained
        else COLD_START_HANDOFF_EXPLORATION_STD_START
    )
    handoff_exploration_std_end = (
        PRETRAINED_HANDOFF_EXPLORATION_STD_END
        if pretrained
        else COLD_START_HANDOFF_EXPLORATION_STD_END
    )
    handoff_episodes = (
        PRETRAINED_HANDOFF_EPISODES
        if pretrained
        else COLD_START_HANDOFF_EPISODES
    )
    handoff_update_mode = (
        "critic_td_plus_actor_bc"
        if pretrained
        else "td3_full"
    )
    cfg = {
        "episode_unit": "cycle",
        "warmup_buffer_only_episodes": WARMUP_EPISODES,
        "behavior_clone_teacher_episodes": BC_TEACHER_EPISODES,
        "bc_update_mode": "critic_td_plus_actor_bc",
        "bc_actor_updates_per_step": 4,
        "bc_exploration_std": bc_exploration_std,
        "handoff_exploration_std_start": handoff_exploration_std_start,
        "handoff_exploration_std_end": handoff_exploration_std_end,
        "handoff_noise_policy_side_only": True,
        "full_rl_exploration_std_start": exploration_std,
        "full_rl_exploration_std_end": FULL_RL_EXPLORATION_STD_END,
        "full_rl_exploration_decay_mode": "linear",
        "bc_teacher_policy": teacher_source,
        "bc_behavior_source": teacher_source,
        "handoff_episodes": handoff_episodes,
        "handoff_blend": "linear",
        "handoff_update_mode": handoff_update_mode,
        "handoff_actor_bc_updates_per_step": 1 if pretrained else 0,
        "warmup_behavior_source": teacher_source,
        "warmup_behavior_noise": "none",
        "bc_behavior_noise": "none" if bc_exploration_std <= 0.0 else "gaussian",
        "handoff_behavior_noise": "none" if handoff_exploration_std_end <= 0.0 else "gaussian",
        "full_rl_behavior_noise": "gaussian",
    }
    if overrides:
        cfg.update({key: value for key, value in dict(overrides).items() if value is not None})
    return cfg


def _preset_uses_gart_family(preset: OnlineTD3Preset) -> bool:
    return bool(
        str(preset.direct_target_mode).strip().lower() == "gart"
        or str(preset.teacher_source).strip().lower() == "gart_lmpc"
        or str(preset.fallback_controller).strip().lower() == "gart_lmpc"
    )


def _lyap_params_for_preset(preset: OnlineTD3Preset) -> tuple[float, float, str]:
    if _preset_uses_gart_family(preset):
        return (
            float(GART_FINAL_RHO_LYAP),
            float(GART_FINAL_LYAP_EPS),
            "utils.gart_defaults final GART constants",
        )
    return float(RHO_LYAP), float(LYAP_EPS), "direct_lmpc_online_defaults"


def _build_reward(
    data_min: np.ndarray,
    data_max: np.ndarray,
    n_inputs: int,
    *,
    fallback_penalty_enabled: bool,
    gamma_fallback: float | None = None,
    fallback_event_penalty: float | None = None,
) -> tuple[dict[str, Any], Any]:
    gamma_fallback = (
        (3.0 if fallback_penalty_enabled else 0.0)
        if gamma_fallback is None
        else float(gamma_fallback)
    )
    fallback_event_penalty = (
        (10.0 if fallback_penalty_enabled else 0.0)
        if fallback_event_penalty is None
        else float(fallback_event_penalty)
    )
    return make_reward_fn_relative_QR(
        data_min=data_min,
        data_max=data_max,
        n_inputs=n_inputs,
        k_rel=np.array([0.0015, 0.00015], dtype=float),
        band_floor_phys=np.array([0.003, 0.035], dtype=float),
        Q_diag=QY_REWARD_DIAG,
        R_diag=RDU_REWARD_DIAG,
        tau_frac=0.5,
        gamma_out=1.0,
        gamma_in=3.0,
        beta=1.0,
        gate="geom",
        lam_in=3.0,
        bonus_kind="quadratic",
        gamma_fallback=gamma_fallback,
        fallback_event_penalty=fallback_event_penalty,
        R_fallback_diag=RDU_REWARD_DIAG,
        maintenance_band_scale=0.5,
        maintenance_move_weight=0.0,
        jitter_weight=0.0,
        dwell_bonus=0.0,
    )


def _bounded_mixed_target_config() -> dict[str, float]:
    return make_direct_lmpc_target_config()


def _gart_target_final_overrides() -> dict[str, Any]:
    return dict(GART_FINAL_TARGET_CONFIG_OVERRIDES)


def _target_config_for_mode(
    target_mode: str,
    system_data: dict[str, Any],
    setup: Any,
) -> tuple[Any, dict[str, Any] | None]:
    if str(target_mode).strip().lower() == "gart":
        discovered = discover_gart_case_values(system_data, setup, results_roots=None)
        return make_gart_target_config(discovered, **_gart_target_final_overrides()), discovered
    return _bounded_mixed_target_config(), None


def _gart_mpc_config_for_context(
    context: DisturbanceContext,
    *,
    rho_lyap: float,
    lyap_eps: float,
) -> Any | None:
    if context.gart_discovered is None:
        return None
    overrides = dict(GART_FINAL_TARGET_OVERRIDES)
    overrides.update(
        {
            "rho": float(rho_lyap),
            "eps": float(lyap_eps),
            "slack_penalty": float(GART_FINAL_SLACK_PENALTY),
            "first_step_contraction_on": True,
        }
    )
    return make_gart_mpc_config(
        context.gart_discovered,
        objective=GART_LMPC_OBJECTIVE,
        lyapunov_mode=GART_LMPC_LYAPUNOV_MODE,
        **overrides,
    )


def _pretrained_selector_note(pretrain_source: str | None, target_mode: str) -> str:
    target_label = str(target_mode).strip().lower()
    if pretrain_source is None:
        return f"cold-start run; no pretrained checkpoint was loaded; online target selector is {target_label}"
    if target_label == "gart":
        return (
            f"{pretrain_source} checkpoint loading is unchanged; online teacher/fallback "
            "uses GART-LMPC with the GART target selector"
        )
    return (
        f"{pretrain_source} checkpoint loading is unchanged; online Direct LMPC "
        f"gate/diagnostic target selector is {TARGET_SELECTOR_VARIANT}"
    )


def build_disturbance_context(target_mode: str = TARGET_MODE) -> DisturbanceContext:
    setup = build_polymer_setup()
    system_data = load_of_mpc_system_data(
        setup,
        setpoint_y_phys=DIRECT_TWO_SETPOINT_Y_PHYS.copy(),
    )
    a_aug = system_data["A_aug"]
    b_aug = system_data["B_aug"]
    c_aug = system_data["C_aug"]
    data_min = system_data["data_min"]
    data_max = system_data["data_max"]
    dimensions = compute_td3_dimensions(a_aug, b_aug, c_aug)
    n_inputs = int(dimensions.inputs_number)

    y_sp_scenario = apply_min_max(
        DIRECT_TWO_SETPOINT_Y_PHYS,
        data_min[n_inputs:],
        data_max[n_inputs:],
    ) - apply_min_max(
        setup.steady_states["y_ss"],
        data_min[n_inputs:],
        data_max[n_inputs:],
    )
    observer_gain = make_observer_gain(a_aug, c_aug)

    u_ss = apply_min_max(setup.steady_states["ss_inputs"], data_min[:n_inputs], data_max[:n_inputs])
    u_min_scaled = apply_min_max(U_MIN_PHYS, data_min[:n_inputs], data_max[:n_inputs])
    u_max_scaled = apply_min_max(U_MAX_PHYS, data_min[:n_inputs], data_max[:n_inputs])
    u_dev_min = u_min_scaled - u_ss
    u_dev_max = u_max_scaled - u_ss
    bnds = tuple((float(lo), float(hi)) for lo, hi in zip(u_dev_min, u_dev_max)) * CONT_H
    ic_opt_template = np.zeros(n_inputs * CONT_H, dtype=float)

    lmpc_obj = design_direct_lyapunov_mpc_solver(
        A_aug=a_aug,
        B_aug=b_aug,
        C_aug=c_aug,
        Qy_diag=QY_MPC_DIAG,
        NP=PREDICT_H,
        NC=CONT_H,
        Su_diag=SU_MPC_DIAG,
        u_min=u_dev_min,
        u_max=u_dev_max,
        Rdu_diag=RDU_MPC_DIAG,
        terminal_set_on=True,
        terminal_alpha_scale=1.0,
    )
    of_mpc_obj = MpcSolver(
        a_aug,
        b_aug,
        c_aug,
        Q_out=QY_MPC_DIAG,
        R_in=RDU_MPC_DIAG,
        NP=PREDICT_H,
        NC=CONT_H,
    )
    reward_config, reward_fn = _build_reward(
        data_min,
        data_max,
        n_inputs,
        fallback_penalty_enabled=False,
    )
    target_config, gart_discovered = _target_config_for_mode(target_mode, system_data, setup)

    return DisturbanceContext(
        setup=setup,
        system_data=system_data,
        dimensions=dimensions,
        y_sp_scenario=y_sp_scenario,
        observer_gain=observer_gain,
        lmpc_obj=lmpc_obj,
        of_mpc_obj=of_mpc_obj,
        ic_opt_template=ic_opt_template,
        bnds=bnds,
        cons=(),
        u_dev_min=u_dev_min,
        u_dev_max=u_dev_max,
        reward_config=reward_config,
        reward_fn=reward_fn,
        target_config=target_config,
        gart_discovered=gart_discovered,
    )


def _resolve_pretrained_checkpoint(source: str, agent_path: str | None) -> Path:
    if source not in {"lmpc", "of_mpc"}:
        raise ValueError(f"Unsupported pretrained source: {source!r}")
    if agent_path:
        candidate = resolve_repo_path(agent_path)
        if not candidate.exists():
            raise FileNotFoundError(f"TD3 checkpoint not found: {candidate}")
        return candidate

    env_names = (
        ("LMPC_PRETRAINED_TD3_AGENT_PATH", "PRETRAINED_TD3_AGENT_PATH")
        if source == "lmpc"
        else ("OFMPC_PRETRAINED_TD3_AGENT_PATH", "PRETRAINED_TD3_AGENT_PATH")
    )
    for env_name in env_names:
        requested = os.environ.get(env_name)
        if requested:
            candidate = resolve_repo_path(requested)
            if not candidate.exists():
                raise FileNotFoundError(f"{env_name} points to a missing TD3 checkpoint: {candidate}")
            return candidate

    latest = latest_lmpc_pretrained_checkpoint() if source == "lmpc" else latest_of_mpc_pretrained_checkpoint()
    if latest is not None:
        return latest

    fallback = repo_path("Data", "agent_2507171027.pkl")
    if fallback.exists():
        return fallback
    raise FileNotFoundError(
        f"No generated {source} TD3 checkpoint found and fallback checkpoint is missing: {fallback}"
    )


def _make_td3_agent(
    *,
    dimensions: TD3Dimensions,
    actor_layers: tuple[int, ...],
    critic_layers: tuple[int, ...],
    smoothing_std: float,
    set_points_len: int,
) -> TD3Agent:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return TD3Agent(
        state_dim=int(dimensions.state_dim),
        action_dim=int(dimensions.action_dim),
        actor_hidden=list(actor_layers),
        critic_hidden=list(critic_layers),
        gamma=GAMMA,
        actor_lr=ACTOR_LR,
        critic_lr=CRITIC_LR,
        batch_size=BATCH_SIZE,
        policy_delay=POLICY_DELAY,
        target_policy_smoothing_noise_std=float(smoothing_std),
        noise_clip=NOISE_CLIP,
        max_action=MAX_ACTION,
        tau=TAU,
        std_start=STD_START,
        std_end=STD_END,
        std_decay_rate=STD_DECAY_RATE,
        std_decay_mode=STD_DECAY_MODE,
        buffer_size=BUFFER_CAPACITY,
        device=device,
        actor_freeze=0 * int(set_points_len),
    )


def _agent_for_preset(
    preset: OnlineTD3Preset,
    *,
    context: DisturbanceContext,
    set_points_len: int,
    agent_path: str | None,
    reset_pretrained_critic: bool,
) -> tuple[TD3Agent, str | None, dict[str, Any] | None, dict[str, Any]]:
    checkpoint_arch = None
    resolved_agent_path: str | None = None
    smoothing_std = PRETRAINED_SMOOTHING_STD if preset.pretrain_source else COLD_START_SMOOTHING_STD
    actor_layers = DEFAULT_ACTOR_LAYER_SIZES
    critic_layers = DEFAULT_CRITIC_LAYER_SIZES

    if preset.pretrain_source is not None:
        checkpoint_path = _resolve_pretrained_checkpoint(preset.pretrain_source, agent_path)
        actor_layers, critic_layers, checkpoint_arch = resolve_checkpoint_layers(
            checkpoint_path=checkpoint_path,
            actor_override=None,
            critic_override=None,
            dimensions=context.dimensions,
        )
        resolved_agent_path = os.fspath(checkpoint_path)

    agent = _make_td3_agent(
        dimensions=context.dimensions,
        actor_layers=tuple(actor_layers),
        critic_layers=tuple(critic_layers),
        smoothing_std=smoothing_std,
        set_points_len=set_points_len,
    )
    if resolved_agent_path is not None:
        agent.load(resolved_agent_path)
    critic_reset_requested = bool(resolved_agent_path is not None and reset_pretrained_critic)
    critic_reset_applied = critic_reset_requested
    if critic_reset_applied:
        agent.reset_critic()
        print("Pretrained actor retained; critic and target critic reset for online training.")
    critic_reset_metadata = {
        "pretrained_critic_reset": critic_reset_applied,
        "critic_reset_requested": critic_reset_requested,
        "critic_reset_scope": "critic_and_critic_target" if critic_reset_applied else None,
        "actor_loaded_from_checkpoint": resolved_agent_path is not None,
        "critic_loaded_from_checkpoint": bool(resolved_agent_path is not None and not critic_reset_applied),
    }
    return agent, resolved_agent_path, checkpoint_arch, critic_reset_metadata


def _study_root(study_name: str, timestamp: str | None = None) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") if timestamp is None else str(timestamp)
    root = repo_path("results", study_name, timestamp)
    root.mkdir(parents=True, exist_ok=True)
    return root


def _timing_metadata(start_time: float, *, n_steps: int, episode_len: int) -> dict[str, Any]:
    elapsed = float(time.perf_counter() - start_time)
    n_episodes = int(np.ceil(n_steps / float(episode_len))) if episode_len > 0 else 0
    return {
        "wall_clock_seconds": elapsed,
        "wall_clock_seconds_per_episode": None if n_episodes <= 0 else elapsed / float(n_episodes),
        "wall_clock_seconds_per_step": None if n_steps <= 0 else elapsed / float(n_steps),
        "wall_clock_steps_per_second": None if elapsed <= 0.0 else n_steps / elapsed,
        "wall_clock_n_steps": int(n_steps),
        "wall_clock_n_episodes": int(n_episodes),
    }


def run_online_td3_disturbance_preset(
    preset_key: str,
    *,
    episodes: int = DIRECT_DISTURBANCE_N_TESTS,
    set_points_len: int = DIRECT_DISTURBANCE_SETPOINT_LEN,
    seed: int = DIRECT_DISTURBANCE_SEED,
    save_plots: bool = True,
    agent_path: str | None = None,
    reset_pretrained_critic: bool = DEFAULT_RESET_PRETRAINED_CRITIC,
    timestamp: str | None = None,
    rl_observation_mode: str = STANDARD_RL_OBSERVATION_MODE,
    projection_backend: str | None = None,
    reward_fallback_penalty_enabled: bool | None = None,
    gamma_fallback: float | None = None,
    fallback_event_penalty: float | None = None,
    rho_lyap: float | None = None,
    lyap_eps: float | None = None,
    lyap_tol: float | None = None,
    training_phase_overrides: dict[str, Any] | None = None,
    section16_projection_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if preset_key not in ONLINE_TD3_PRESETS:
        known = ", ".join(sorted(ONLINE_TD3_PRESETS))
        raise ValueError(f"Unknown online TD3 preset {preset_key!r}. Known presets: {known}")
    preset = ONLINE_TD3_PRESETS[preset_key]
    episodes = int(episodes)
    set_points_len = int(set_points_len)
    if episodes <= 0:
        raise ValueError("episodes must be positive.")
    if set_points_len <= 0:
        raise ValueError("set_points_len must be positive.")

    set_seed(int(seed))
    context = build_disturbance_context(preset.direct_target_mode)
    rl_observation_mode = normalize_rl_observation_mode(rl_observation_mode)
    context = replace(
        context,
        dimensions=dimensions_for_observation_mode(context.dimensions, rl_observation_mode),
    )
    scaling_contract = _scaling_contract(context.setup, context)
    fallback_penalty_enabled = (
        bool(preset.safety_gate)
        if reward_fallback_penalty_enabled is None
        else bool(reward_fallback_penalty_enabled)
    )
    reward_config, reward_fn = _build_reward(
        context.system_data["data_min"],
        context.system_data["data_max"],
        context.dimensions.inputs_number,
        fallback_penalty_enabled=fallback_penalty_enabled,
        gamma_fallback=gamma_fallback,
        fallback_event_penalty=fallback_event_penalty,
    )
    agent, resolved_agent_path, checkpoint_arch, critic_reset_metadata = _agent_for_preset(
        preset,
        context=context,
        set_points_len=set_points_len,
        agent_path=agent_path,
        reset_pretrained_critic=bool(reset_pretrained_critic),
    )
    study_root = _study_root(preset.study_name, timestamp=timestamp)
    test_cycle = direct_disturbance_test_cycle(episodes)
    training_phase_config = _training_phase_config(
        teacher_source=preset.teacher_source,
        pretrained=preset.pretrain_source is not None,
        overrides=training_phase_overrides,
    )
    case_rho_lyap, case_lyap_eps, lyap_param_source = _lyap_params_for_preset(preset)
    if rho_lyap is not None:
        case_rho_lyap = float(rho_lyap)
        lyap_param_source = f"{lyap_param_source}; runner override rho_lyap"
    if lyap_eps is not None:
        case_lyap_eps = float(lyap_eps)
        lyap_param_source = f"{lyap_param_source}; runner override lyap_eps"
    case_lyap_tol = float(LYAP_TOL if lyap_tol is None else lyap_tol)
    projection_backend = _normalize_projection_backend_override(
        projection_backend,
        safety_gate=bool(preset.safety_gate),
    )
    if projection_backend == MPC_ONLY_DIAGNOSTIC_BACKEND:
        effective_fallback_controller = "none"
    else:
        effective_fallback_controller = preset.fallback_controller if preset.safety_gate else "none"
    uses_gart_lmpc_controller = bool(
        preset.teacher_source == "gart_lmpc"
        or effective_fallback_controller == "gart_lmpc"
        or rl_observation_mode == GART_RL_OBSERVATION_MODE
    )
    if uses_gart_lmpc_controller:
        controller_family = "gart_lmpc"
    elif preset.teacher_source == "offset_free_mpc":
        controller_family = "offset_free_mpc"
    else:
        controller_family = "direct_lmpc"
    case_slack_penalty = float(GART_FINAL_SLACK_PENALTY if uses_gart_lmpc_controller else SLACK_PENALTY)
    gart_mpc_config = (
        _gart_mpc_config_for_context(context, rho_lyap=case_rho_lyap, lyap_eps=case_lyap_eps)
        if uses_gart_lmpc_controller
        else None
    )
    if uses_gart_lmpc_controller and gart_mpc_config is None:
        raise RuntimeError("GART-LMPC online control requires a GART disturbance context.")
    case_mpc_obj = context.lmpc_obj
    teacher_mpc_obj = None
    if preset.teacher_source == "offset_free_mpc":
        if preset.safety_gate:
            teacher_mpc_obj = context.of_mpc_obj
        else:
            case_mpc_obj = context.of_mpc_obj

    case_config = {
        "study_name": preset.study_name,
        "case_name": preset.study_name,
        "label": preset.label,
        "controller_mode": "td3_safety_gate" if preset.safety_gate else "td3_no_safety_gate",
        "controller_family": controller_family,
        "projection_backend": projection_backend,
        "rl_observation_mode": rl_observation_mode,
        "rl_observation_state_dim": int(context.dimensions.state_dim),
        "standard_rl_observation_state_dim": int(compute_td3_dimensions(
            context.system_data["A_aug"],
            context.system_data["B_aug"],
            context.system_data["C_aug"],
        ).state_dim),
        "safety_gate_active": bool(preset.safety_gate),
        "pretrain_source": preset.pretrain_source,
        "teacher_source": preset.teacher_source,
        "fallback_controller": effective_fallback_controller,
        "target_mode": preset.direct_target_mode,
        "target_selector_variant": "gart" if preset.direct_target_mode == "gart" else TARGET_SELECTOR_VARIANT,
        "target_config": _jsonable(context.target_config),
        "target_config_source": (
            "utils.gart_defaults final GART constants"
            if preset.direct_target_mode == "gart"
            else "direct_lmpc_selector_defaults"
        ),
        "gart_target_overrides": (
            _jsonable(_gart_target_final_overrides()) if preset.direct_target_mode == "gart" else None
        ),
        "gart_lmpc_config": _jsonable(gart_mpc_config),
        "gart_lmpc_config_source": (
            "utils.gart_defaults final GART constants" if gart_mpc_config is not None else None
        ),
        "gart_lmpc_objective": GART_LMPC_OBJECTIVE if gart_mpc_config is not None else None,
        "gart_lmpc_lyapunov_mode": GART_LMPC_LYAPUNOV_MODE if gart_mpc_config is not None else None,
        "pretrained_checkpoint_selector_note": _pretrained_selector_note(preset.pretrain_source, preset.direct_target_mode),
        "plant_mode": PLANT_MODE,
        "n_tests": episodes,
        "n_episodes": episodes,
        "set_points_len": set_points_len,
        "force_final_test": FORCE_FINAL_TEST,
        "disturbance_after_step": DISTURBANCE_AFTER_STEP,
        "use_target_output_for_tracking": USE_TARGET_OUTPUT_FOR_TRACKING,
        "seed": int(seed),
        "Qy_mpc_diag": QY_MPC_DIAG.copy(),
        "Su_mpc_diag": SU_MPC_DIAG.copy(),
        "Rdu_mpc_diag": RDU_MPC_DIAG.copy(),
        "Qy_reward_diag": QY_REWARD_DIAG.copy(),
        "Rdu_reward_diag": RDU_REWARD_DIAG.copy(),
        "reward_config": dict(reward_config),
        "reward_fallback_penalty_enabled": fallback_penalty_enabled,
        "reward_fallback_penalty_activation_rule": (
            f"fallback_active when {projection_backend} changes the TD3 candidate action"
            if fallback_penalty_enabled and projection_backend != MPC_ONLY_DIAGNOSTIC_BACKEND
            else "disabled; no-safety-gate runs log diagnostics only and never apply fallback penalties"
        ),
        "u_prev_penalty_weight": U_PREV_PENALTY_WEIGHT,
        "xs_prev_penalty_weight": XS_PREV_PENALTY_WEIGHT,
        "rho_lyap": case_rho_lyap,
        "lyap_eps": case_lyap_eps,
        "lyap_param_source": lyap_param_source,
        "rho_lyap_default": RHO_LYAP,
        "lyap_eps_default": LYAP_EPS,
        "gart_rho_lyap_default": GART_FINAL_RHO_LYAP if uses_gart_lmpc_controller else None,
        "gart_lyap_eps_default": GART_FINAL_LYAP_EPS if uses_gart_lmpc_controller else None,
        "lyap_eps_pretrained_online_override": None,
        "lyap_eps_override_reason": "online target-diagnostic and fallback-controller Lyapunov epsilon",
        "lyap_tol": case_lyap_tol,
        "slack_penalty": case_slack_penalty,
        "gart_slack_penalty_default": GART_FINAL_SLACK_PENALTY if uses_gart_lmpc_controller else None,
        "training_phase_config": dict(training_phase_config),
        "training_phase_overrides": _jsonable(training_phase_overrides),
        "section16_projection_config": _jsonable(section16_projection_config),
        "initial_agent_path": resolved_agent_path,
        "checkpoint_architecture": checkpoint_arch,
        "online_td3_hparams": _td3_online_hparams(agent),
        "scaling_contract": scaling_contract,
    }
    case_config.update(critic_reset_metadata)

    print(f"Running {preset.study_name} into {study_root}")
    timer_start = time.perf_counter()
    results = run_rl_train(
        system=make_polymer_system(context.setup),
        y_sp_scenario=context.y_sp_scenario,
        n_tests=episodes,
        set_points_len=set_points_len,
        steady_states=context.setup.steady_states,
        min_max_dict=context.system_data["min_max_dict"],
        agent=agent,
        MPC_obj=case_mpc_obj,
        L=context.observer_gain,
        data_min=context.system_data["data_min"],
        data_max=context.system_data["data_max"],
        warm_start=0,
        test_cycle=test_cycle,
        nominal_qi=NOMINAL_QI,
        nominal_qs=NOMINAL_QS,
        nominal_ha=NOMINAL_HA,
        qi_change=QI_CHANGE,
        qs_change=QS_CHANGE,
        ha_change=HA_CHANGE,
        reward_fn=reward_fn,
        mode=PLANT_MODE,
        rho_lyap=case_rho_lyap,
        lyap_eps=case_lyap_eps,
        lyap_tol=case_lyap_tol,
        seed=int(seed),
        use_lyap=bool(preset.safety_gate),
        IC_opt=context.ic_opt_template.copy(),
        bnds=context.bnds,
        cons=context.cons,
        reuse_mpc_solution_as_ic=False,
        reset_system_on_entry=True,
        projection_backend=projection_backend,
        first_step_contraction_on=True,
        direct_target_mode=preset.direct_target_mode,
        direct_target_config=context.target_config,
        gart_mpc_config=gart_mpc_config,
        fallback_controller=effective_fallback_controller,
        direct_tracking_use_target_output=USE_TARGET_OUTPUT_FOR_TRACKING,
        diagnostic_lmpc_obj=context.lmpc_obj,
        teacher_mpc_obj=teacher_mpc_obj,
        disturbance_after_step=DISTURBANCE_AFTER_STEP,
        training_phase_config=training_phase_config,
        force_final_test=FORCE_FINAL_TEST,
        rl_observation_mode=rl_observation_mode,
        gart_section16_config=section16_projection_config,
    )
    timing = _timing_metadata(timer_start, n_steps=int(results[5]), episode_len=int(results[6]))
    case_config.update(timing)

    bundle = build_safety_filter_run_bundle(
        source=preset.study_name,
        results=results,
        steady_states=context.setup.steady_states,
        config=case_config,
        min_max_dict=context.system_data["min_max_dict"],
        data_min=context.system_data["data_min"],
        data_max=context.system_data["data_max"],
        extra={
            "delta_t": context.setup.delta_t,
            "phase_plot_boundaries": _phase_plot_boundaries(
                episodes,
                set_points_len,
                training_phase_config=training_phase_config,
            ),
            "start_plot_idx": 10,
            "agent_path": resolved_agent_path,
            "reward_config": reward_config,
            "actor_losses": agent.actor_losses,
            "critic_losses": agent.critic_losses,
            "timing": timing,
            "gart_discovered": context.gart_discovered,
            "gart_lmpc_config": _jsonable(gart_mpc_config),
        },
    )
    debug_dir = save_safety_filter_debug_artifacts(
        bundle=bundle,
        directory=study_root,
        prefix_name=preset.study_name,
        save_plots=save_plots,
    )
    trained_agent_path = agent.save(debug_dir, prefix="trained_agent", include_optim=False)
    bundle["extra"]["trained_agent_path"] = trained_agent_path
    bundle["config"]["trained_agent_path"] = trained_agent_path

    record = make_safety_filter_comparison_record(preset.study_name, bundle, debug_dir)
    record.update(timing)
    record["trained_agent_path"] = trained_agent_path
    comparison_artifacts = save_safety_filter_comparison_artifacts(
        [record],
        {preset.study_name: bundle},
        os.fspath(study_root),
        save_plots=save_plots,
    )
    _mirror_primary_debug_files(debug_dir, study_root)
    _write_json(study_root / "record.json", record)
    run_summary = {
        "study_name": preset.study_name,
        "case_name": preset.study_name,
        "pretrained_critic_reset": bool(critic_reset_metadata["pretrained_critic_reset"]),
        "critic_reset_scope": critic_reset_metadata["critic_reset_scope"],
        "actor_loaded_from_checkpoint": bool(critic_reset_metadata["actor_loaded_from_checkpoint"]),
        "result_root": os.fspath(study_root),
        "debug_dir": debug_dir,
        "comparison_artifacts": comparison_artifacts,
        "record_json": os.fspath(study_root / "record.json"),
        "trained_agent_path": trained_agent_path,
        "config": case_config,
    }
    _write_json(study_root / "run_summary.json", run_summary)
    print(f"Completed {preset.study_name}")
    pprint(record)
    return run_summary


def _base_direct_config(study_name: str, episodes: int, set_points_len: int, seed: int) -> dict[str, Any]:
    return {
        "study_name": study_name,
        "plant_mode": PLANT_MODE,
        "n_tests": int(episodes),
        "n_episodes": int(episodes),
        "set_points_len": int(set_points_len),
        "force_final_test": FORCE_FINAL_TEST,
        "disturbance_after_step": DISTURBANCE_AFTER_STEP,
        "use_target_output_for_tracking": USE_TARGET_OUTPUT_FOR_TRACKING,
        "seed": int(seed),
        "predict_h": PREDICT_H,
        "cont_h": CONT_H,
        "rho_lyap": RHO_LYAP,
        "lyap_eps": LYAP_EPS,
        "lyap_tol": LYAP_TOL,
        "slack_penalty": SLACK_PENALTY,
        "target_mode": TARGET_MODE,
        "target_selector_variant": TARGET_SELECTOR_VARIANT,
        "target_config": _bounded_mixed_target_config(),
        "Qy_mpc_diag": QY_MPC_DIAG.copy(),
        "Su_mpc_diag": SU_MPC_DIAG.copy(),
        "Rdu_mpc_diag": RDU_MPC_DIAG.copy(),
        "Qy_reward_diag": QY_REWARD_DIAG.copy(),
        "Rdu_reward_diag": RDU_REWARD_DIAG.copy(),
        "reward_fallback_penalty_enabled": False,
        "reward_fallback_penalty_activation_rule": "disabled for MPC-only baselines",
        "u_prev_penalty_weight": U_PREV_PENALTY_WEIGHT,
        "xs_prev_penalty_weight": XS_PREV_PENALTY_WEIGHT,
    }


def _save_direct_baseline_outputs(
    *,
    study_root: Path,
    study_name: str,
    case_name: str,
    bundle: dict[str, Any],
    save_plots: bool,
) -> dict[str, Any]:
    debug_dir = save_direct_lyapunov_debug_artifacts(
        bundle,
        directory=study_root,
        prefix_name=case_name,
        save_plots=save_plots,
    )
    _write_direct_episode_table(debug_dir, bundle)
    record = make_direct_lyapunov_comparison_record(case_name, bundle, debug_dir)
    comparison_artifacts = save_direct_lyapunov_comparison_artifacts(
        [record],
        {case_name: bundle},
        os.fspath(study_root),
        save_plots=save_plots,
    )
    _mirror_primary_debug_files(debug_dir, study_root)
    _write_json(study_root / "record.json", record)
    run_summary = {
        "study_name": study_name,
        "case_name": case_name,
        "result_root": os.fspath(study_root),
        "debug_dir": debug_dir,
        "comparison_artifacts": comparison_artifacts,
        "record_json": os.fspath(study_root / "record.json"),
        "config": bundle.get("config", {}),
    }
    _write_json(study_root / "run_summary.json", run_summary)
    pprint(record)
    return run_summary


def run_direct_lmpc_disturbance(
    *,
    episodes: int = DIRECT_DISTURBANCE_N_TESTS,
    set_points_len: int = DIRECT_DISTURBANCE_SETPOINT_LEN,
    seed: int = DIRECT_DISTURBANCE_SEED,
    save_plots: bool = True,
) -> dict[str, Any]:
    episodes = int(episodes)
    set_points_len = int(set_points_len)
    if episodes <= 0:
        raise ValueError("episodes must be positive.")
    if set_points_len <= 0:
        raise ValueError("set_points_len must be positive.")

    set_seed(int(seed))
    context = build_disturbance_context()
    study_name = "DirectLMPCDisturbance"
    case_name = "direct_lmpc_disturbance"
    study_root = _study_root(study_name)
    case_config = {
        **_base_direct_config(study_name, episodes, set_points_len, seed),
        "case_name": case_name,
        "controller_mode": "direct_lyapunov_mpc",
        "target_mode": TARGET_MODE,
        "lyapunov_mode": "hard",
        "target_config": dict(context.target_config),
        "reward_config": dict(context.reward_config),
    }

    print(f"Running {study_name} into {study_root}")
    timer_start = time.perf_counter()
    results = run_direct_output_disturbance_lyapunov_mpc(
        system=make_polymer_system(context.setup),
        LMPC_obj=context.lmpc_obj,
        y_sp_scenario=context.y_sp_scenario,
        n_tests=episodes,
        set_points_len=set_points_len,
        steady_states=context.setup.steady_states,
        IC_opt=context.ic_opt_template.copy(),
        bnds=context.bnds,
        L=context.observer_gain,
        data_min=context.system_data["data_min"],
        data_max=context.system_data["data_max"],
        test_cycle=direct_disturbance_test_cycle(episodes),
        reward_fn=context.reward_fn,
        nominal_qi=NOMINAL_QI,
        nominal_qs=NOMINAL_QS,
        nominal_ha=NOMINAL_HA,
        qi_change=QI_CHANGE,
        qs_change=QS_CHANGE,
        ha_change=HA_CHANGE,
        target_mode=TARGET_MODE,
        lyapunov_mode="hard",
        target_config=dict(context.target_config),
        target_H=None,
        mode=PLANT_MODE,
        disturbance_after_step=DISTURBANCE_AFTER_STEP,
        use_target_output_for_tracking=USE_TARGET_OUTPUT_FOR_TRACKING,
        skip_terminal_if_alpha_small=True,
        alpha_terminal_min=1e-8,
        use_target_on_solver_fail=False,
        rho_lyap=RHO_LYAP,
        lyap_eps=LYAP_EPS,
        slack_penalty=SLACK_PENALTY,
        first_step_contraction_on=True,
        reset_system_on_entry=True,
        solver_options={"warm_start": True},
        force_final_test=FORCE_FINAL_TEST,
    )
    timing = _timing_metadata(
        timer_start,
        n_steps=int(results["nFE"]),
        episode_len=int(results["time_in_sub_episodes"]),
    )
    case_config.update(timing)
    bundle = build_direct_lyapunov_run_bundle(
        source=case_name,
        results=results,
        steady_states=context.setup.steady_states,
        config=case_config,
        data_min=context.system_data["data_min"],
        data_max=context.system_data["data_max"],
        extra={
            "reward_config": context.reward_config,
            "min_max_dict": context.system_data["min_max_dict"],
            "timing": timing,
        },
    )
    return _save_direct_baseline_outputs(
        study_root=study_root,
        study_name=study_name,
        case_name=case_name,
        bundle=bundle,
        save_plots=save_plots,
    )


def run_offset_free_mpc_disturbance(
    *,
    episodes: int = DIRECT_DISTURBANCE_N_TESTS,
    set_points_len: int = DIRECT_DISTURBANCE_SETPOINT_LEN,
    seed: int = DIRECT_DISTURBANCE_SEED,
    save_plots: bool = True,
) -> dict[str, Any]:
    episodes = int(episodes)
    set_points_len = int(set_points_len)
    if episodes <= 0:
        raise ValueError("episodes must be positive.")
    if set_points_len <= 0:
        raise ValueError("set_points_len must be positive.")

    set_seed(int(seed))
    context = build_disturbance_context()
    study_name = "OffsetFreeMPCDisturbance"
    case_name = "offset_free_mpc_disturbance"
    study_root = _study_root(study_name)
    case_config = {
        **_base_direct_config(study_name, episodes, set_points_len, seed),
        "case_name": case_name,
        "controller_mode": "offset_free_mpc",
        "target_mode": TARGET_MODE,
        "lyapunov_mode": "diagnostic_only",
        "target_config": dict(context.target_config),
        "diagnostic_lmpc_enabled": True,
        "reward_config": dict(context.reward_config),
    }

    print(f"Running {study_name} into {study_root}")
    timer_start = time.perf_counter()
    results = run_offset_free_mpc_with_direct_diagnostics(
        system=make_polymer_system(context.setup),
        MPC_obj=context.of_mpc_obj,
        diagnostic_LMPC_obj=context.lmpc_obj,
        y_sp_scenario=context.y_sp_scenario,
        n_tests=episodes,
        set_points_len=set_points_len,
        steady_states=context.setup.steady_states,
        IC_opt=context.ic_opt_template.copy(),
        bnds=context.bnds,
        L=context.observer_gain,
        data_min=context.system_data["data_min"],
        data_max=context.system_data["data_max"],
        test_cycle=direct_disturbance_test_cycle(episodes),
        reward_fn=context.reward_fn,
        nominal_qi=NOMINAL_QI,
        nominal_qs=NOMINAL_QS,
        nominal_ha=NOMINAL_HA,
        qi_change=QI_CHANGE,
        qs_change=QS_CHANGE,
        ha_change=HA_CHANGE,
        target_mode=TARGET_MODE,
        target_config=dict(context.target_config),
        target_H=None,
        mode=PLANT_MODE,
        disturbance_after_step=DISTURBANCE_AFTER_STEP,
        use_target_output_for_tracking=USE_TARGET_OUTPUT_FOR_TRACKING,
        rho_lyap=RHO_LYAP,
        lyap_eps=LYAP_EPS,
        first_step_contraction_on=True,
        reset_system_on_entry=True,
        solver_options={"warm_start": True},
        force_final_test=FORCE_FINAL_TEST,
    )
    timing = _timing_metadata(
        timer_start,
        n_steps=int(results["nFE"]),
        episode_len=int(results["time_in_sub_episodes"]),
    )
    case_config.update(timing)
    bundle = build_direct_lyapunov_run_bundle(
        source=case_name,
        results=results,
        steady_states=context.setup.steady_states,
        config=case_config,
        data_min=context.system_data["data_min"],
        data_max=context.system_data["data_max"],
        extra={
            "reward_config": context.reward_config,
            "min_max_dict": context.system_data["min_max_dict"],
            "timing": timing,
        },
    )
    return _save_direct_baseline_outputs(
        study_root=study_root,
        study_name=study_name,
        case_name=case_name,
        bundle=bundle,
        save_plots=save_plots,
    )


def build_online_arg_parser(description: str, *, include_agent_path: bool = False) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--episodes", type=int, default=DIRECT_DISTURBANCE_N_TESTS)
    parser.add_argument("--set-points-len", type=int, default=DIRECT_DISTURBANCE_SETPOINT_LEN)
    parser.add_argument("--seed", type=int, default=DIRECT_DISTURBANCE_SEED)
    parser.add_argument("--save-plots", dest="save_plots", action="store_true", default=True)
    parser.add_argument("--no-save-plots", dest="save_plots", action="store_false")
    if include_agent_path:
        parser.add_argument("--agent-path", default=None)
        parser.add_argument(
            "--keep-pretrained-critic",
            action="store_true",
            default=False,
            help="Keep the critic loaded from the pretrained checkpoint instead of resetting it for online training.",
        )
    return parser


def _main_online(preset_key: str, argv: list[str] | None = None) -> dict[str, Any]:
    preset = ONLINE_TD3_PRESETS[preset_key]
    parser = build_online_arg_parser(
        preset.label,
        include_agent_path=preset.pretrain_source is not None,
    )
    args = parser.parse_args(argv)
    return run_online_td3_disturbance_preset(
        preset_key,
        episodes=args.episodes,
        set_points_len=args.set_points_len,
        seed=args.seed,
        save_plots=args.save_plots,
        agent_path=getattr(args, "agent_path", None),
        reset_pretrained_critic=not bool(getattr(args, "keep_pretrained_critic", False)),
    )


def _main_direct_baseline(kind: str, argv: list[str] | None = None) -> dict[str, Any]:
    if kind not in {"direct_lmpc", "offset_free_mpc"}:
        raise ValueError(f"Unknown baseline kind: {kind!r}")
    parser = build_online_arg_parser(
        "Direct LMPC disturbance baseline" if kind == "direct_lmpc" else "Offset-free MPC disturbance baseline",
        include_agent_path=False,
    )
    args = parser.parse_args(argv)
    runner = run_direct_lmpc_disturbance if kind == "direct_lmpc" else run_offset_free_mpc_disturbance
    return runner(
        episodes=args.episodes,
        set_points_len=args.set_points_len,
        seed=args.seed,
        save_plots=args.save_plots,
    )


def main_lmpc_pretrained_safety_gate(argv: list[str] | None = None) -> dict[str, Any]:
    return _main_online("lmpc_pretrained_safety_gate", argv)


def main_ofmpc_pretrained_safety_gate(argv: list[str] | None = None) -> dict[str, Any]:
    return _main_online("ofmpc_pretrained_safety_gate", argv)


def main_lmpc_pretrained_no_safety_gate(argv: list[str] | None = None) -> dict[str, Any]:
    return _main_online("lmpc_pretrained_no_safety_gate", argv)


def main_ofmpc_pretrained_no_safety_gate(argv: list[str] | None = None) -> dict[str, Any]:
    return _main_online("ofmpc_pretrained_no_safety_gate", argv)


def main_cold_start_safety_gate(argv: list[str] | None = None) -> dict[str, Any]:
    return _main_online("cold_start_safety_gate", argv)


def main_cold_start_no_safety_gate(argv: list[str] | None = None) -> dict[str, Any]:
    return _main_online("cold_start_no_safety_gate", argv)


def main_direct_lmpc_disturbance(argv: list[str] | None = None) -> dict[str, Any]:
    return _main_direct_baseline("direct_lmpc", argv)


def main_offset_free_mpc_disturbance(argv: list[str] | None = None) -> dict[str, Any]:
    return _main_direct_baseline("offset_free_mpc", argv)
