from __future__ import annotations

import os
import pickle
import re
import time
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from Lyapunov.direct_lyapunov_mpc import (
    design_direct_lyapunov_mpc_solver,
    prepare_direct_output_disturbance_step,
    run_direct_output_disturbance_lyapunov_mpc,
    run_offset_free_mpc_with_direct_diagnostics,
    solve_direct_tracking_from_target,
)
from Plotting_fns.mpc_plot_fns import plot_mpc_rl_results_cstr
from Simulation.mpc import MpcSolver
from TD3Agent.agent import TD3Agent
from TD3Agent.reward_functions import make_reward_fn_mpc_quadratic
from utils.direct_lyapunov_study import governed_reference_case_spec
from utils.helpers import generate_setpoints_training_rl_gradually
from utils.of_mpc_td3_workflow import (
    COMPARISON_SETPOINT_Y_PHYS,
    CONTROL_HORIZON,
    HA_CHANGE,
    NOMINAL_HA,
    NOMINAL_QI,
    NOMINAL_QS,
    OBSERVER_POLES,
    PolymerSetup,
    QI_CHANGE,
    QS_CHANGE,
    TD3Dimensions,
    U_MAX_PHYS,
    U_MIN_PHYS,
    array_stats,
    build_polymer_setup,
    compute_td3_dimensions,
    jsonable,
    load_of_mpc_system_data,
    make_observer_gain,
    make_of_mpc_components,
    make_polymer_system,
    mode_list,
    relative_to_repo,
    resolve_device,
    save_loss_artifacts,
    scaled_setpoint_scenario,
    set_seed,
    trajectory_metrics,
    validate_layer_sizes,
    write_csv,
    write_json,
    y_sp_phys_from_scaled,
)
from utils.path_helpers import repo_path, resolve_repo_path
from utils.scaling_helpers import apply_min_max, apply_min_max_pm1, reverse_min_max
from utils.td3_helpers import ReplayDataset


PREDICT_HORIZON = 9
QY_MPC = np.array([5.0, 1.0], dtype=float)
SU_MPC = np.array([1.0, 1.0], dtype=float)
RDU_MPC = np.array([1.0, 1.0], dtype=float)

RHO_LYAP = 0.99
LYAP_EPS = 1.0e-9
SLACK_PENALTY = 1.0e6
TARGET_MODE = "governed_reference"
LYAPUNOV_MODE = "hard"
FIRST_STEP_CONTRACTION_ON = True
USE_TARGET_OUTPUT_FOR_TRACKING = False
USE_TARGET_ON_SOLVER_FAIL = False
DISTURBANCE_AFTER_STEP = False

U_PREV_PENALTY_WEIGHT = 0.0
XS_PREV_PENALTY_WEIGHT = 0.0

DEFAULT_LMPC_SAMPLES = 100_000
DEFAULT_STEADY_SAMPLES = 10_000
DEFAULT_CANDIDATE_CHUNK_SIZE = 512
DEFAULT_WORKER_BATCH_SIZE = 32
DEFAULT_MAX_ATTEMPT_MULTIPLIER = 5.0
DEFAULT_ACTOR_EPOCHS = 500
DEFAULT_CRITIC_EPOCHS = 200
DEFAULT_PRETRAIN_BATCH_SIZE = 4096
DEFAULT_ACTOR_LAYER_SIZES = (256, 256, 256)
DEFAULT_CRITIC_LAYER_SIZES = (256, 256, 256)


@dataclass(frozen=True)
class LMPCComponents:
    lmpc_obj: Any
    ic_opt: np.ndarray
    bnds: tuple[tuple[float, float], ...]
    u_dev_min: np.ndarray
    u_dev_max: np.ndarray
    target_config: dict[str, Any]


@dataclass(frozen=True)
class LMPCPretrainingRunConfig:
    lmpc_samples: int = DEFAULT_LMPC_SAMPLES
    steady_samples: int = DEFAULT_STEADY_SAMPLES
    candidate_chunk_size: int = DEFAULT_CANDIDATE_CHUNK_SIZE
    worker_batch_size: int = DEFAULT_WORKER_BATCH_SIZE
    max_attempt_multiplier: float = DEFAULT_MAX_ATTEMPT_MULTIPLIER
    actor_epochs: int = DEFAULT_ACTOR_EPOCHS
    critic_epochs: int = DEFAULT_CRITIC_EPOCHS
    pretrain_batch_size: int = DEFAULT_PRETRAIN_BATCH_SIZE
    actor_layer_sizes: tuple[int, ...] = DEFAULT_ACTOR_LAYER_SIZES
    critic_layer_sizes: tuple[int, ...] = DEFAULT_CRITIC_LAYER_SIZES
    seed: int = 123
    device_requested: str = "auto"
    output_root: str = os.path.join("results", "PretrainLMPC")


@dataclass(frozen=True)
class LMPCComparisonRunConfig:
    actor_layer_sizes: tuple[int, ...] | None = None
    critic_layer_sizes: tuple[int, ...] | None = None
    agent_path: str | None = None
    modes: tuple[str, ...] = ("nominal", "disturb")
    n_tests: int = 2
    set_points_len: int = 400
    seed: int = 123
    device_requested: str = "auto"
    output_root: str = os.path.join("results", "PretrainLMPCComparison")
    baseline_cache_dir: str = os.path.join("results", "PretrainLMPCComparison", "baselines")
    force_baseline_refresh: bool = False
    disturbance_after_step: bool = DISTURBANCE_AFTER_STEP


def validate_lmpc_pretraining_config(config: LMPCPretrainingRunConfig) -> None:
    if config.lmpc_samples < 0 or config.steady_samples < 0:
        raise ValueError("lmpc_samples and steady_samples must be nonnegative.")
    if config.lmpc_samples + config.steady_samples <= 0:
        raise ValueError("At least one accepted LMPC replay label is required.")
    if config.candidate_chunk_size <= 0:
        raise ValueError("candidate_chunk_size must be positive.")
    if config.worker_batch_size <= 0:
        raise ValueError("worker_batch_size must be positive.")
    if config.max_attempt_multiplier <= 0:
        raise ValueError("max_attempt_multiplier must be positive.")
    if config.actor_epochs < 0 or config.critic_epochs < 0:
        raise ValueError("actor_epochs and critic_epochs must be nonnegative.")
    if config.actor_epochs + config.critic_epochs <= 0:
        raise ValueError("At least one actor or critic pretraining epoch is required.")
    if config.pretrain_batch_size <= 0:
        raise ValueError("pretrain_batch_size must be positive.")
    validate_layer_sizes(tuple(config.actor_layer_sizes), "actor_layer_sizes")
    validate_layer_sizes(tuple(config.critic_layer_sizes), "critic_layer_sizes")


def validate_lmpc_comparison_config(config: LMPCComparisonRunConfig) -> None:
    if config.n_tests <= 0:
        raise ValueError("n_tests must be positive.")
    if config.set_points_len <= 0:
        raise ValueError("set_points_len must be positive.")
    if not config.modes:
        raise ValueError("At least one comparison mode is required.")
    unknown = [mode for mode in config.modes if mode not in {"nominal", "disturb"}]
    if unknown:
        raise ValueError(f"Unsupported modes: {unknown}")
    if config.actor_layer_sizes is not None:
        validate_layer_sizes(tuple(config.actor_layer_sizes), "actor_layer_sizes")
    if config.critic_layer_sizes is not None:
        validate_layer_sizes(tuple(config.critic_layer_sizes), "critic_layer_sizes")


def make_lmpc_target_config() -> dict[str, Any]:
    return governed_reference_case_spec(
        QY_MPC,
        case_name="lmpc_td3_pretraining_governed_reference",
        controller_mode="direct_lyapunov_mpc",
        lyapunov_mode=LYAPUNOV_MODE,
        u_ref_weight=U_PREV_PENALTY_WEIGHT,
        x_ref_weight=XS_PREV_PENALTY_WEIGHT,
    )["target_config"]


def make_lmpc_components(system_data: dict[str, Any]) -> LMPCComponents:
    u_dev_min = np.asarray(system_data["b_min"], dtype=float).reshape(-1)
    u_dev_max = np.asarray(system_data["b_max"], dtype=float).reshape(-1)
    inputs_number = int(system_data["B_aug"].shape[1])
    bnds = tuple(
        (float(u_dev_min[idx]), float(u_dev_max[idx]))
        for _ in range(CONTROL_HORIZON)
        for idx in range(inputs_number)
    )
    target_config = make_lmpc_target_config()
    lmpc_obj = design_direct_lyapunov_mpc_solver(
        A_aug=system_data["A_aug"],
        B_aug=system_data["B_aug"],
        C_aug=system_data["C_aug"],
        Qy_diag=QY_MPC,
        NP=PREDICT_HORIZON,
        NC=CONTROL_HORIZON,
        Su_diag=SU_MPC,
        u_min=u_dev_min,
        u_max=u_dev_max,
        Rdu_diag=RDU_MPC,
        terminal_set_on=True,
        terminal_alpha_scale=1.0,
    )
    return LMPCComponents(
        lmpc_obj=lmpc_obj,
        ic_opt=np.zeros(inputs_number * CONTROL_HORIZON, dtype=float),
        bnds=bnds,
        u_dev_min=u_dev_min,
        u_dev_max=u_dev_max,
        target_config=target_config,
    )


def make_lmpc_offline_reward() -> tuple[dict[str, Any], Any]:
    return make_reward_fn_mpc_quadratic(Q_diag=QY_MPC, R_diag=RDU_MPC)


def make_lmpc_td3_agent(
    dimensions: TD3Dimensions,
    *,
    buffer_size: int,
    device: torch.device,
    actor_hidden: tuple[int, ...] | list[int],
    critic_hidden: tuple[int, ...] | list[int],
) -> TD3Agent:
    return TD3Agent(
        state_dim=dimensions.state_dim,
        action_dim=dimensions.action_dim,
        actor_hidden=list(actor_hidden),
        critic_hidden=list(critic_hidden),
        gamma=0.99,
        actor_lr=5.0e-5,
        critic_lr=5.0e-4,
        batch_size=256,
        policy_delay=2,
        target_policy_smoothing_noise_std=0.01,
        noise_clip=0.01,
        max_action=1.0,
        tau=0.005,
        std_start=0.0,
        std_end=0.0,
        std_decay_rate=0.99992,
        std_decay_mode="exp",
        buffer_size=buffer_size,
        device=device,
        mode="mpc",
    )


def _sample_candidates(
    min_max_dict: dict[str, np.ndarray],
    count: int,
    *,
    kind: str,
    state_dim: int,
    output_dim: int,
    input_dim: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_min = np.asarray(min_max_dict["x_min"], dtype=float)
    x_max = np.asarray(min_max_dict["x_max"], dtype=float)
    y_sp_min = np.asarray(min_max_dict["y_sp_min"], dtype=float)
    y_sp_max = np.asarray(min_max_dict["y_sp_max"], dtype=float)
    u_min = np.asarray(min_max_dict["u_min"], dtype=float)
    u_max = np.asarray(min_max_dict["u_max"], dtype=float)

    if kind == "broad":
        x_aug = np.random.uniform(low=x_min, high=x_max, size=(count, state_dim))
        y_sp = np.random.uniform(low=y_sp_min, high=y_sp_max, size=(count, output_dim))
        u_prev = np.random.uniform(low=u_min, high=u_max, size=(count, input_dim))
    elif kind == "steady":
        sigma = (x_max - x_min) / 1.0e5
        x_aug = np.random.normal(0.0, sigma, size=(count, state_dim))
        y_sp = np.zeros((count, output_dim), dtype=float)
        u_prev = np.random.uniform(low=0.0, high=1.0e-8, size=(count, input_dim))
        u_prev = np.clip(u_prev, u_min, u_max)
    else:
        raise ValueError("kind must be 'broad' or 'steady'.")
    return x_aug, y_sp, u_prev


def _failure_key(target_info: dict[str, Any], step_info: dict[str, Any]) -> str:
    if not bool(target_info.get("success", False)):
        return f"target:{target_info.get('solve_stage') or target_info.get('status') or 'failed'}"
    status = step_info.get("status") or "unknown_status"
    message = step_info.get("message") or step_info.get("method") or "solver_failed"
    return f"tracking:{status}:{message}"


def _label_candidate(
    *,
    system_data: dict[str, Any],
    setup: PolymerSetup,
    dimensions: TD3Dimensions,
    lmpc: LMPCComponents,
    reward_fn: Any,
    x_aug: np.ndarray,
    y_sp: np.ndarray,
    u_prev: np.ndarray,
    step_idx: int,
) -> tuple[bool, dict[str, Any], dict[str, np.ndarray] | None]:
    step_context = prepare_direct_output_disturbance_step(
        LMPC_obj=lmpc.lmpc_obj,
        x0_aug=x_aug,
        y_sp_k=y_sp,
        u_prev_dev=u_prev,
        u_dev_min=lmpc.u_dev_min,
        u_dev_max=lmpc.u_dev_max,
        target_mode=TARGET_MODE,
        target_config=lmpc.target_config,
        target_H=None,
        x_target_prev_success=None,
        r_cmd_prev_success=None,
        step_idx=step_idx,
        y_prev_scaled=None,
        plant_mode=None,
        disturbance_after_step=None,
        use_target_output_for_tracking=USE_TARGET_OUTPUT_FOR_TRACKING,
        slack_penalty=SLACK_PENALTY,
        rho_lyap=RHO_LYAP,
        lyap_eps=LYAP_EPS,
    )
    target_info = step_context["target_info"]
    step_info = step_context["step_info"]
    u_apply, _ic_next, step_info = solve_direct_tracking_from_target(
        LMPC_obj=lmpc.lmpc_obj,
        x0_aug=x_aug,
        y_sp_k=y_sp,
        u_prev_dev=u_prev,
        target_info=target_info,
        step_info=step_info,
        IC_opt=lmpc.ic_opt.copy(),
        bnds=lmpc.bnds,
        u_dev_min=lmpc.u_dev_min,
        u_dev_max=lmpc.u_dev_max,
        rho_lyap=RHO_LYAP,
        lyap_eps=LYAP_EPS,
        lyapunov_mode=LYAPUNOV_MODE,
        use_target_output_for_tracking=USE_TARGET_OUTPUT_FOR_TRACKING,
        use_target_on_solver_fail=USE_TARGET_ON_SOLVER_FAIL,
        slack_penalty=SLACK_PENALTY,
        first_step_contraction_on=FIRST_STEP_CONTRACTION_ON,
        solver_options={"warm_start": True},
    )
    diagnostic = {
        "target_success": bool(target_info.get("success", False)),
        "target_stage": target_info.get("solve_stage"),
        "target_status": target_info.get("status"),
        "target_message": target_info.get("message"),
        "governor_active": target_info.get("governor_active"),
        "governor_probe_success": target_info.get("governor_probe_success"),
        "target_mismatch_inf": target_info.get("target_error_inf"),
        "success": bool(step_info.get("success", False)),
        "method": step_info.get("method"),
        "status": step_info.get("status"),
        "message": step_info.get("message"),
        "tracking_solver": step_info.get("tracking_solver"),
        "contraction_margin": step_info.get("contraction_margin"),
        "first_step_contraction_satisfied": step_info.get("first_step_contraction_satisfied"),
        "terminal_constraint_skipped": step_info.get("terminal_constraint_skipped"),
        "alpha_terminal_used": step_info.get("alpha_terminal_used"),
        "failure_key": _failure_key(target_info, step_info),
    }
    if not bool(step_info.get("success", False)):
        return False, diagnostic, None

    u_apply = np.asarray(u_apply, dtype=float).reshape(dimensions.inputs_number)
    next_x_aug = system_data["A_aug"] @ x_aug + system_data["B_aug"] @ u_apply
    y_pred = system_data["C_aug"] @ next_x_aug
    delta_y = y_pred - y_sp
    delta_u = u_apply - u_prev
    y_sp_phys = y_sp_phys_from_scaled(
        y_sp,
        steady_states=setup.steady_states,
        data_min=system_data["data_min"],
        data_max=system_data["data_max"],
        inputs_number=dimensions.inputs_number,
    )
    reward = float(reward_fn(delta_y, delta_u, y_sp_phys))
    transition = {
        "state": _rl_scaled(system_data["min_max_dict"], x_aug, y_sp, u_prev),
        "action": apply_min_max_pm1(u_apply, lmpc.u_dev_min, lmpc.u_dev_max),
        "reward": np.asarray(reward, dtype=np.float32),
        "next_state": _rl_scaled(system_data["min_max_dict"], next_x_aug, y_sp, u_apply),
    }
    return True, diagnostic, transition


def _rl_scaled(min_max_dict: dict[str, np.ndarray], x_aug: np.ndarray, y_sp: np.ndarray, u_dev: np.ndarray) -> np.ndarray:
    x_scaled = apply_min_max_pm1(x_aug, min_max_dict["x_min"], min_max_dict["x_max"])
    y_scaled = apply_min_max_pm1(y_sp, min_max_dict["y_sp_min"], min_max_dict["y_sp_max"])
    u_scaled = apply_min_max_pm1(u_dev, min_max_dict["u_min"], min_max_dict["u_max"])
    return np.hstack((x_scaled, y_scaled, u_scaled)).astype(np.float32)


def _flush_transitions(agent: TD3Agent, transitions: list[dict[str, np.ndarray]]) -> None:
    if not transitions:
        return
    states = np.vstack([item["state"] for item in transitions]).astype(np.float32)
    actions = np.vstack([item["action"] for item in transitions]).astype(np.float32)
    rewards = np.asarray([float(item["reward"]) for item in transitions], dtype=np.float32)
    next_states = np.vstack([item["next_state"] for item in transitions]).astype(np.float32)
    agent.buffer.pretrain_add(states, actions, rewards, next_states)
    transitions.clear()


def _generate_lmpc_label_subset(
    *,
    agent: TD3Agent,
    system_data: dict[str, Any],
    setup: PolymerSetup,
    dimensions: TD3Dimensions,
    lmpc: LMPCComponents,
    reward_fn: Any,
    requested: int,
    kind: str,
    candidate_chunk_size: int,
    worker_batch_size: int,
    max_attempt_multiplier: float,
) -> dict[str, Any]:
    requested = int(requested)
    if requested <= 0:
        return {
            "kind": kind,
            "requested": 0,
            "accepted": 0,
            "attempted": 0,
            "acceptance_rate": None,
            "status": "skipped",
            "failure_reasons": {},
            "target_stage_counts": {},
            "tracking_status_counts": {},
            "sample_records": [],
        }

    max_attempts = max(requested, int(np.ceil(requested * float(max_attempt_multiplier))))
    accepted = 0
    attempted = 0
    failure_reasons: Counter[str] = Counter()
    target_stage_counts: Counter[str] = Counter()
    tracking_status_counts: Counter[str] = Counter()
    sample_records: list[dict[str, Any]] = []
    pending: list[dict[str, np.ndarray]] = []
    wall_start = time.perf_counter()

    while accepted < requested and attempted < max_attempts:
        remaining_attempts = max_attempts - attempted
        draw_count = min(int(candidate_chunk_size), remaining_attempts)
        x_batch, y_batch, u_batch = _sample_candidates(
            system_data["min_max_dict"],
            draw_count,
            kind=kind,
            state_dim=system_data["A_aug"].shape[0],
            output_dim=system_data["C_aug"].shape[0],
            input_dim=system_data["B_aug"].shape[1],
        )

        for row_idx in range(draw_count):
            if accepted >= requested:
                break
            attempted += 1
            success, diagnostic, transition = _label_candidate(
                system_data=system_data,
                setup=setup,
                dimensions=dimensions,
                lmpc=lmpc,
                reward_fn=reward_fn,
                x_aug=x_batch[row_idx, :],
                y_sp=y_batch[row_idx, :],
                u_prev=u_batch[row_idx, :],
                step_idx=attempted - 1,
            )
            target_stage_counts[str(diagnostic.get("target_stage"))] += 1
            tracking_status_counts[str(diagnostic.get("status"))] += 1
            if success and transition is not None:
                accepted += 1
                pending.append(transition)
                if len(sample_records) < 50:
                    record = dict(diagnostic)
                    record["kind"] = kind
                    record["accepted_index"] = accepted
                    sample_records.append(record)
                if len(pending) >= int(worker_batch_size):
                    _flush_transitions(agent, pending)
            else:
                failure_reasons[str(diagnostic.get("failure_key", "unknown"))] += 1
                if len(sample_records) < 50:
                    record = dict(diagnostic)
                    record["kind"] = kind
                    record["accepted_index"] = None
                    sample_records.append(record)

        print(
            f"[lmpc-labels][{kind}] accepted={accepted}/{requested} "
            f"attempted={attempted}/{max_attempts}"
        )

    _flush_transitions(agent, pending)
    seconds = float(time.perf_counter() - wall_start)
    status = "completed" if accepted >= requested else "insufficient_labels"
    return {
        "kind": kind,
        "requested": requested,
        "accepted": int(accepted),
        "attempted": int(attempted),
        "max_attempts": int(max_attempts),
        "acceptance_rate": float(accepted / attempted) if attempted > 0 else None,
        "elapsed_seconds": seconds,
        "status": status,
        "failure_reasons": dict(failure_reasons),
        "target_stage_counts": dict(target_stage_counts),
        "tracking_status_counts": dict(tracking_status_counts),
        "sample_records": sample_records,
    }


def fill_lmpc_replay_buffer(
    *,
    agent: TD3Agent,
    system_data: dict[str, Any],
    setup: PolymerSetup,
    dimensions: TD3Dimensions,
    lmpc: LMPCComponents,
    reward_fn: Any,
    lmpc_samples: int,
    steady_samples: int,
    candidate_chunk_size: int,
    worker_batch_size: int,
    max_attempt_multiplier: float,
) -> dict[str, Any]:
    broad_diag = _generate_lmpc_label_subset(
        agent=agent,
        system_data=system_data,
        setup=setup,
        dimensions=dimensions,
        lmpc=lmpc,
        reward_fn=reward_fn,
        requested=int(lmpc_samples),
        kind="broad",
        candidate_chunk_size=int(candidate_chunk_size),
        worker_batch_size=int(worker_batch_size),
        max_attempt_multiplier=float(max_attempt_multiplier),
    )
    steady_diag = _generate_lmpc_label_subset(
        agent=agent,
        system_data=system_data,
        setup=setup,
        dimensions=dimensions,
        lmpc=lmpc,
        reward_fn=reward_fn,
        requested=int(steady_samples),
        kind="steady",
        candidate_chunk_size=int(candidate_chunk_size),
        worker_batch_size=int(worker_batch_size),
        max_attempt_multiplier=float(max_attempt_multiplier),
    )
    requested_total = int(lmpc_samples + steady_samples)
    accepted_total = int(broad_diag["accepted"] + steady_diag["accepted"])
    attempted_total = int(broad_diag["attempted"] + steady_diag["attempted"])
    status = "completed" if accepted_total >= requested_total else "insufficient_labels"
    return {
        "status": status,
        "requested_total": requested_total,
        "accepted_total": accepted_total,
        "attempted_total": attempted_total,
        "acceptance_rate": float(accepted_total / attempted_total) if attempted_total > 0 else None,
        "broad": broad_diag,
        "steady": steady_diag,
    }


def run_lmpc_pretraining(config: LMPCPretrainingRunConfig) -> dict[str, Any]:
    validate_lmpc_pretraining_config(config)
    device = resolve_device(config.device_requested)
    set_seed(config.seed)

    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = resolve_repo_path(config.output_root, create=True)
    run_dir = output_root / run_timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    wall_start = time.perf_counter()
    setup = build_polymer_setup()
    system_data = load_of_mpc_system_data(setup)
    dimensions = compute_td3_dimensions(system_data["A_aug"], system_data["B_aug"], system_data["C_aug"])
    total_samples = int(config.lmpc_samples + config.steady_samples)
    agent = make_lmpc_td3_agent(
        dimensions,
        buffer_size=total_samples,
        device=device,
        actor_hidden=config.actor_layer_sizes,
        critic_hidden=config.critic_layer_sizes,
    )
    lmpc = make_lmpc_components(system_data)
    reward_config, reward_fn = make_lmpc_offline_reward()

    label_start = time.perf_counter()
    label_diagnostics = fill_lmpc_replay_buffer(
        agent=agent,
        system_data=system_data,
        setup=setup,
        dimensions=dimensions,
        lmpc=lmpc,
        reward_fn=reward_fn,
        lmpc_samples=config.lmpc_samples,
        steady_samples=config.steady_samples,
        candidate_chunk_size=config.candidate_chunk_size,
        worker_batch_size=config.worker_batch_size,
        max_attempt_multiplier=config.max_attempt_multiplier,
    )
    label_seconds = float(time.perf_counter() - label_start)
    label_diagnostics_path = run_dir / "label_diagnostics.json"
    write_json(label_diagnostics_path, label_diagnostics)

    if label_diagnostics["status"] != "completed":
        summary_path = run_dir / "summary.json"
        summary = {
            "status": "failed_insufficient_lmpc_labels",
            "run_timestamp": run_timestamp,
            "run_dir": relative_to_repo(run_dir),
            "label_generation_seconds": label_seconds,
            "label_diagnostics": relative_to_repo(label_diagnostics_path),
            "requested_total": label_diagnostics["requested_total"],
            "accepted_total": label_diagnostics["accepted_total"],
            "attempted_total": label_diagnostics["attempted_total"],
        }
        write_json(summary_path, summary)
        raise RuntimeError(
            "LMPC label generation could not reach the requested accepted count. "
            f"Accepted {label_diagnostics['accepted_total']} of {label_diagnostics['requested_total']} "
            f"after {label_diagnostics['attempted_total']} attempts. Diagnostics: {label_diagnostics_path}"
        )

    buffer_size = len(agent.buffer)
    dataset = ReplayDataset(
        agent.buffer.states[:buffer_size],
        agent.buffer.actions[:buffer_size],
        agent.buffer.rewards[:buffer_size],
        agent.buffer.next_states[:buffer_size],
        agent.buffer.dones[:buffer_size],
    )
    data_loader = DataLoader(
        dataset,
        batch_size=min(config.pretrain_batch_size, buffer_size),
        shuffle=True,
        drop_last=False,
        pin_memory=(device.type == "cuda"),
    )

    train_start = time.perf_counter()
    agent.pretrain_from_buffer(
        num_actor_epochs=config.actor_epochs,
        num_critic_epochs=config.critic_epochs,
        data_loader=data_loader,
        use_target_noise_critic=True,
        log_interval=1,
        mode="mpc",
    )
    train_seconds = float(time.perf_counter() - train_start)

    checkpoint_path = Path(
        agent.save(
            str(run_dir),
            prefix="lmpc_pretrained_td3",
            include_optim=False,
        )
    )
    loss_paths = save_loss_artifacts(run_dir, agent)

    full_config = {
        "run_timestamp": run_timestamp,
        "method": "td3_pretraining_from_direct_lyapunov_mpc",
        "run_config": asdict(config),
        "device_used": str(device),
        "controller": {
            "augmentation_style": "rawlings",
            "augmentation_mode": "output_disturbance",
            "setpoint_scaler_y_phys": system_data.get("setpoint_range_y_used"),
            "rollout_setpoint_y_phys": COMPARISON_SETPOINT_Y_PHYS,
            "u_min_phys": U_MIN_PHYS,
            "u_max_phys": U_MAX_PHYS,
            "predict_horizon": PREDICT_HORIZON,
            "control_horizon": CONTROL_HORIZON,
            "Qy_mpc_diag": QY_MPC,
            "Su_mpc_diag": SU_MPC,
            "Rdu_mpc_diag": RDU_MPC,
            "rho_lyap": RHO_LYAP,
            "lyap_eps": LYAP_EPS,
            "slack_penalty": SLACK_PENALTY,
            "target_mode": TARGET_MODE,
            "target_config": lmpc.target_config,
            "lyapunov_mode": LYAPUNOV_MODE,
            "first_step_contraction_on": FIRST_STEP_CONTRACTION_ON,
            "use_target_output_for_tracking": USE_TARGET_OUTPUT_FOR_TRACKING,
            "use_target_on_solver_fail": USE_TARGET_ON_SOLVER_FAIL,
        },
        "td3": {
            "dimensions": dimensions,
            "actor_hidden": config.actor_layer_sizes,
            "critic_hidden": config.critic_layer_sizes,
            "gamma": agent.gamma,
            "actor_lr": agent.actor_lr,
            "critic_lr": agent.critic_lr,
            "batch_size": agent.batch_size,
            "policy_delay": agent.policy_delay,
            "tau": agent.tau,
            "max_action": agent.max_action,
        },
        "reward_config": reward_config,
        "system": {
            "system_dict_path": system_data["system_dict_path"],
            "Bd_used": system_data["Bd_used"],
            "Cd_used": system_data["Cd_used"],
            "steady_states": setup.steady_states,
            "min_max_dict": system_data["min_max_dict"],
        },
    }
    config_path = run_dir / "config.json"
    write_json(config_path, full_config)

    elapsed_seconds = float(time.perf_counter() - wall_start)
    summary = {
        "status": "completed",
        "run_timestamp": run_timestamp,
        "elapsed_seconds": elapsed_seconds,
        "label_generation_seconds": label_seconds,
        "pretraining_seconds": train_seconds,
        "buffer_size": buffer_size,
        "lmpc_samples": int(config.lmpc_samples),
        "steady_samples": int(config.steady_samples),
        "state_dim": int(dimensions.state_dim),
        "action_dim": int(dimensions.action_dim),
        "checkpoint_path": relative_to_repo(checkpoint_path),
        "config_path": relative_to_repo(config_path),
        "label_diagnostics_path": relative_to_repo(label_diagnostics_path),
        **loss_paths,
        "label_diagnostics_summary": {
            "accepted_total": label_diagnostics["accepted_total"],
            "attempted_total": label_diagnostics["attempted_total"],
            "acceptance_rate": label_diagnostics["acceptance_rate"],
            "broad_acceptance_rate": label_diagnostics["broad"]["acceptance_rate"],
            "steady_acceptance_rate": label_diagnostics["steady"]["acceptance_rate"],
        },
        "reward_stats": array_stats(agent.buffer.rewards[:buffer_size]),
        "action_stats": array_stats(agent.buffer.actions[:buffer_size]),
        "state_stats": array_stats(agent.buffer.states[:buffer_size]),
    }
    summary_path = run_dir / "summary.json"
    write_json(summary_path, summary)

    return {
        "run_dir": run_dir,
        "checkpoint_path": checkpoint_path,
        "config_path": config_path,
        "summary_path": summary_path,
        "label_diagnostics_path": label_diagnostics_path,
        "summary": summary,
    }


def latest_lmpc_pretrained_checkpoint() -> Path | None:
    root = repo_path("results", "PretrainLMPC")
    if not root.exists():
        return None
    candidates = list(root.glob("**/lmpc_pretrained_td3_*.pkl"))
    if not candidates:
        return None
    return max(candidates, key=lambda path: (path.stat().st_mtime, str(path)))


def resolve_lmpc_pretrained_checkpoint(agent_path: str | None) -> Path:
    if agent_path:
        candidate = resolve_repo_path(agent_path)
        if not candidate.exists():
            raise FileNotFoundError(f"TD3 checkpoint not found: {candidate}")
        return candidate

    latest = latest_lmpc_pretrained_checkpoint()
    if latest is not None:
        return latest

    fallback = repo_path("Data", "agent_2507171027.pkl")
    if not fallback.exists():
        raise FileNotFoundError(
            "No generated LMPC pretraining checkpoint was found, and the default "
            f"fallback checkpoint is missing: {fallback}"
        )
    return fallback


def load_checkpoint_payload(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Checkpoint payload must be a dictionary: {path}")
    return payload


def _infer_hidden_from_state_dict(state_dict: dict[str, Any], prefix: str) -> tuple[int, ...] | None:
    layers: list[tuple[int, int]] = []
    pattern = re.compile(rf"^{re.escape(prefix)}\.layer(\d+)\.weight$")
    for key, value in state_dict.items():
        match = pattern.match(str(key))
        if not match:
            continue
        weight = np.asarray(value.detach().cpu().numpy() if hasattr(value, "detach") else value)
        if weight.ndim != 2:
            continue
        layers.append((int(match.group(1)), int(weight.shape[0])))
    if not layers:
        return None
    return tuple(size for _, size in sorted(layers, key=lambda item: item[0]))


def infer_checkpoint_architecture(path: Path) -> dict[str, Any]:
    payload = load_checkpoint_payload(path)
    hparams = dict(payload.get("hparams", {}))
    actor_state = payload.get("actor_state_dict", {})
    critic_state = payload.get("critic_state_dict", {})
    actor_layers = hparams.get("actor_hidden")
    critic_layers = hparams.get("critic_hidden")
    if actor_layers is None:
        actor_layers = _infer_hidden_from_state_dict(actor_state, "model")
    if critic_layers is None:
        critic_layers = _infer_hidden_from_state_dict(critic_state, "q1_network")

    state_dim = hparams.get("state_dim")
    action_dim = hparams.get("action_dim")
    actor_layer0 = actor_state.get("model.layer0.weight") if isinstance(actor_state, dict) else None
    actor_output = actor_state.get("model.output_layer.weight") if isinstance(actor_state, dict) else None
    if state_dim is None and actor_layer0 is not None:
        state_dim = int(actor_layer0.shape[1])
    if action_dim is None and actor_output is not None:
        action_dim = int(actor_output.shape[0])

    return {
        "state_dim": None if state_dim is None else int(state_dim),
        "action_dim": None if action_dim is None else int(action_dim),
        "actor_layer_sizes": None if actor_layers is None else tuple(int(v) for v in actor_layers),
        "critic_layer_sizes": None if critic_layers is None else tuple(int(v) for v in critic_layers),
        "hparams": hparams,
    }


def resolve_checkpoint_layers(
    *,
    checkpoint_path: Path,
    actor_override: tuple[int, ...] | None,
    critic_override: tuple[int, ...] | None,
    dimensions: TD3Dimensions,
) -> tuple[tuple[int, ...], tuple[int, ...], dict[str, Any]]:
    arch = infer_checkpoint_architecture(checkpoint_path)
    if arch["state_dim"] is not None and int(arch["state_dim"]) != int(dimensions.state_dim):
        raise ValueError(
            f"Checkpoint state_dim={arch['state_dim']} does not match computed state_dim={dimensions.state_dim}."
        )
    if arch["action_dim"] is not None and int(arch["action_dim"]) != int(dimensions.action_dim):
        raise ValueError(
            f"Checkpoint action_dim={arch['action_dim']} does not match computed action_dim={dimensions.action_dim}."
        )

    actor_layers = actor_override or arch["actor_layer_sizes"] or DEFAULT_ACTOR_LAYER_SIZES
    critic_layers = critic_override or arch["critic_layer_sizes"] or DEFAULT_CRITIC_LAYER_SIZES
    validate_layer_sizes(tuple(actor_layers), "actor_layer_sizes")
    validate_layer_sizes(tuple(critic_layers), "critic_layer_sizes")
    return tuple(actor_layers), tuple(critic_layers), arch


def _baseline_cache_path(
    cache_dir: Path,
    *,
    controller: str,
    mode: str,
    n_tests: int,
    set_points_len: int,
    disturbance_after_step: bool,
) -> Path:
    timing = "after" if bool(disturbance_after_step) else "before"
    weights = _weights_cache_token(QY_MPC, RDU_MPC)
    return cache_dir / (
        f"{controller}_{mode}_n{int(n_tests)}_len{int(set_points_len)}_"
        f"disturb_{timing}_{weights}.pickle"
    )


def _weights_cache_token(q_diag: np.ndarray, r_diag: np.ndarray) -> str:
    def format_value(value: float) -> str:
        text = f"{float(value):.6g}"
        return text.replace("-", "m").replace(".", "p")

    q_token = "_".join(format_value(value) for value in np.asarray(q_diag, dtype=float).reshape(-1))
    r_token = "_".join(format_value(value) for value in np.asarray(r_diag, dtype=float).reshape(-1))
    return f"q{q_token}_r{r_token}"


def _save_rollout_payload(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(payload, handle)


def run_pretrained_td3_lmpc_rollout(
    *,
    system: Any,
    agent: TD3Agent,
    model_obj: Any,
    y_sp_scenario: np.ndarray,
    n_tests: int,
    set_points_len: int,
    steady_states: dict[str, np.ndarray],
    observer_gain: np.ndarray,
    data_min: np.ndarray,
    data_max: np.ndarray,
    test_cycle: list[bool],
    reward_fn: Any,
    min_max_dict: dict[str, np.ndarray],
    mode: str,
    disturbance_after_step: bool,
) -> dict[str, Any]:
    y_sp, nFE, sub_changes, time_in_sub_episodes, _test_train, _warm_start, qi, qs, ha = (
        generate_setpoints_training_rl_gradually(
            y_sp_scenario,
            n_tests,
            set_points_len,
            0,
            test_cycle,
            NOMINAL_QI,
            NOMINAL_QS,
            NOMINAL_HA,
            QI_CHANGE,
            QS_CHANGE,
            HA_CHANGE,
            force_final_test=True,
        )
    )
    n_inputs = int(model_obj.B.shape[1])
    n_outputs = int(model_obj.C.shape[0])
    n_states = int(model_obj.A.shape[0])
    ss_scaled_inputs = apply_min_max(steady_states["ss_inputs"], data_min[:n_inputs], data_max[:n_inputs])
    y_ss_scaled = apply_min_max(steady_states["y_ss"], data_min[n_inputs:], data_max[n_inputs:])
    u_min = np.asarray(min_max_dict["u_min"], dtype=float)
    u_max = np.asarray(min_max_dict["u_max"], dtype=float)

    y_system = np.zeros((nFE + 1, n_outputs), dtype=float)
    y_system[0, :] = system.current_output
    u_applied_phys = np.zeros((nFE, n_inputs), dtype=float)
    yhat = np.zeros((n_outputs, nFE), dtype=float)
    xhatdhat = np.zeros((n_states, nFE + 1), dtype=float)
    rewards = np.zeros(nFE, dtype=float)
    avg_rewards = []
    delta_y_storage = []
    delta_u_storage = []

    def map_to_bounds(action: np.ndarray) -> np.ndarray:
        action = np.asarray(action, dtype=float).reshape(n_inputs)
        return u_min + ((action + 1.0) / 2.0) * (u_max - u_min)

    mode = str(mode).strip().lower()
    disturbance_after_step = bool(disturbance_after_step)
    system.Qi = NOMINAL_QI
    system.Qs = NOMINAL_QS
    system.hA = NOMINAL_HA

    for step_idx in range(nFE):
        scaled_current_input = apply_min_max(system.current_input, data_min[:n_inputs], data_max[:n_inputs])
        u_prev_dev = scaled_current_input - ss_scaled_inputs
        current_rl_state = _rl_scaled(min_max_dict, xhatdhat[:, step_idx], y_sp[step_idx, :], u_prev_dev)
        action = agent.act_eval(current_rl_state)
        u_dev_apply = np.clip(map_to_bounds(action), u_min, u_max)
        u_scaled = u_dev_apply + ss_scaled_inputs
        u_phys = reverse_min_max(u_scaled, data_min[:n_inputs], data_max[:n_inputs])
        u_applied_phys[step_idx, :] = u_phys.copy()
        delta_u = u_scaled - scaled_current_input

        y_prev_scaled = apply_min_max(y_system[step_idx, :], data_min[n_inputs:], data_max[n_inputs:]) - y_ss_scaled
        yhat[:, step_idx] = model_obj.C @ xhatdhat[:, step_idx]
        innovation = y_prev_scaled - yhat[:, step_idx]

        if mode == "disturb" and not disturbance_after_step:
            system.hA = ha[step_idx]
            system.Qs = qs[step_idx]
            system.Qi = qi[step_idx]

        system.current_input = u_phys
        system.step()

        if mode == "disturb" and disturbance_after_step:
            system.hA = ha[step_idx]
            system.Qs = qs[step_idx]
            system.Qi = qi[step_idx]

        y_system[step_idx + 1, :] = system.current_output
        y_current_scaled = apply_min_max(y_system[step_idx + 1, :], data_min[n_inputs:], data_max[n_inputs:]) - y_ss_scaled
        delta_y = y_current_scaled - y_sp[step_idx, :]
        xhatdhat[:, step_idx + 1] = model_obj.A @ xhatdhat[:, step_idx] + model_obj.B @ u_dev_apply + observer_gain @ innovation

        y_sp_phys = reverse_min_max(y_sp[step_idx, :] + y_ss_scaled, data_min[n_inputs:], data_max[n_inputs:])
        rewards[step_idx] = float(reward_fn(delta_y, delta_u, y_sp_phys))
        delta_y_storage.append(delta_y.copy())
        delta_u_storage.append(delta_u.copy())

        if step_idx in sub_changes:
            avg_rewards.append(float(np.mean(rewards[step_idx - time_in_sub_episodes + 1 : step_idx + 1])))
            print("Sub_Episode:", sub_changes[step_idx], "| td3 avg. reward:", avg_rewards[-1])

    return {
        "source": "pretrained_td3",
        "y_system": y_system,
        "u_applied_phys": u_applied_phys,
        "avg_rewards": avg_rewards,
        "rewards": rewards,
        "xhatdhat": xhatdhat,
        "nFE": int(nFE),
        "time_in_sub_episodes": int(time_in_sub_episodes),
        "y_sp": np.asarray(y_sp, dtype=float).copy(),
        "yhat": yhat,
        "delta_y_storage": delta_y_storage,
        "delta_u_storage": delta_u_storage,
        "qi": np.asarray(qi, dtype=float).copy(),
        "qs": np.asarray(qs, dtype=float).copy(),
        "ha": np.asarray(ha, dtype=float).copy(),
        "plant_mode": mode,
        "disturbance_after_step": disturbance_after_step,
    }


def load_or_generate_lmpc_baseline(
    *,
    config: LMPCComparisonRunConfig,
    setup: PolymerSetup,
    system_data: dict[str, Any],
    lmpc: LMPCComponents,
    y_sp_scenario: np.ndarray,
    observer_gain: np.ndarray,
    reward_fn: Any,
    mode: str,
) -> tuple[dict[str, Any], Path, bool]:
    cache_dir = resolve_repo_path(config.baseline_cache_dir, create=True)
    path = _baseline_cache_path(
        cache_dir,
        controller="direct_lmpc",
        mode=mode,
        n_tests=config.n_tests,
        set_points_len=config.set_points_len,
        disturbance_after_step=config.disturbance_after_step,
    )
    if path.exists() and not config.force_baseline_refresh:
        with path.open("rb") as handle:
            return pickle.load(handle), path, False

    payload = run_direct_output_disturbance_lyapunov_mpc(
        make_polymer_system(setup),
        lmpc.lmpc_obj,
        y_sp_scenario,
        config.n_tests,
        config.set_points_len,
        setup.steady_states,
        lmpc.ic_opt.copy(),
        lmpc.bnds,
        observer_gain,
        system_data["data_min"],
        system_data["data_max"],
        [False] * config.n_tests,
        reward_fn,
        NOMINAL_QI,
        NOMINAL_QS,
        NOMINAL_HA,
        QI_CHANGE,
        QS_CHANGE,
        HA_CHANGE,
        target_mode=TARGET_MODE,
        lyapunov_mode=LYAPUNOV_MODE,
        target_config=lmpc.target_config,
        mode=mode,
        disturbance_after_step=config.disturbance_after_step,
        use_target_output_for_tracking=USE_TARGET_OUTPUT_FOR_TRACKING,
        use_target_on_solver_fail=USE_TARGET_ON_SOLVER_FAIL,
        rho_lyap=RHO_LYAP,
        lyap_eps=LYAP_EPS,
        slack_penalty=SLACK_PENALTY,
        first_step_contraction_on=FIRST_STEP_CONTRACTION_ON,
        reset_system_on_entry=True,
        solver_options={"warm_start": True},
        force_final_test=True,
    )
    payload["source"] = "direct_lmpc"
    _save_rollout_payload(path, payload)
    return payload, path, True


def load_or_generate_of_mpc_diagnostic_baseline(
    *,
    config: LMPCComparisonRunConfig,
    setup: PolymerSetup,
    system_data: dict[str, Any],
    lmpc: LMPCComponents,
    y_sp_scenario: np.ndarray,
    observer_gain: np.ndarray,
    reward_fn: Any,
    mode: str,
) -> tuple[dict[str, Any], Path, bool]:
    cache_dir = resolve_repo_path(config.baseline_cache_dir, create=True)
    path = _baseline_cache_path(
        cache_dir,
        controller="offset_free_mpc",
        mode=mode,
        n_tests=config.n_tests,
        set_points_len=config.set_points_len,
        disturbance_after_step=config.disturbance_after_step,
    )
    if path.exists() and not config.force_baseline_refresh:
        with path.open("rb") as handle:
            return pickle.load(handle), path, False

    of_mpc = make_of_mpc_components(system_data, q_mpc=QY_MPC, r_mpc=RDU_MPC)
    payload = run_offset_free_mpc_with_direct_diagnostics(
        make_polymer_system(setup),
        of_mpc.mpc_obj,
        lmpc.lmpc_obj,
        y_sp_scenario,
        config.n_tests,
        config.set_points_len,
        setup.steady_states,
        of_mpc.ic_opt.copy(),
        of_mpc.bnds,
        observer_gain,
        system_data["data_min"],
        system_data["data_max"],
        [False] * config.n_tests,
        reward_fn,
        NOMINAL_QI,
        NOMINAL_QS,
        NOMINAL_HA,
        QI_CHANGE,
        QS_CHANGE,
        HA_CHANGE,
        target_mode=TARGET_MODE,
        target_config=lmpc.target_config,
        mode=mode,
        disturbance_after_step=config.disturbance_after_step,
        use_target_output_for_tracking=USE_TARGET_OUTPUT_FOR_TRACKING,
        rho_lyap=RHO_LYAP,
        lyap_eps=LYAP_EPS,
        first_step_contraction_on=FIRST_STEP_CONTRACTION_ON,
        reset_system_on_entry=True,
        solver_options={"warm_start": True},
        force_final_test=True,
    )
    payload["source"] = "offset_free_mpc"
    _save_rollout_payload(path, payload)
    return payload, path, True


def _rollout_arrays(payload: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.asarray(payload["y_system"], dtype=float),
        np.asarray(payload["u_applied_phys"], dtype=float),
        np.asarray(payload["y_sp"], dtype=float),
        np.asarray(payload["rewards"], dtype=float),
    )


def _diagnostic_summary(payload: dict[str, Any]) -> dict[str, Any]:
    direct_info = list(payload.get("direct_info_storage", []) or [])
    target_info = list(payload.get("target_info_storage", []) or [])
    if not direct_info and not target_info:
        return {
            "solver_success_rate": None,
            "target_success_rate": None,
            "target_stage_counts": {},
            "contraction_satisfied_rate": None,
            "diagnostic_unsafe_count": None,
        }
    success_values = [bool(info.get("success", False)) for info in direct_info]
    target_success_values = [bool(info.get("target_success", info.get("success", False))) for info in target_info]
    contraction_values = [
        bool(info.get("first_step_contraction_satisfied"))
        for info in direct_info
        if info.get("first_step_contraction_satisfied") is not None
    ]
    return {
        "solver_success_rate": float(np.mean(success_values)) if success_values else None,
        "target_success_rate": float(np.mean(target_success_values)) if target_success_values else None,
        "target_stage_counts": dict(Counter(str(info.get("target_stage")) for info in direct_info)),
        "contraction_satisfied_rate": float(np.mean(contraction_values)) if contraction_values else None,
        "diagnostic_unsafe_count": int(
            sum(bool(info.get("diagnostic_unsafe", False)) for info in direct_info)
        )
        if direct_info
        else None,
    }


def controller_metric_record(
    *,
    mode: str,
    controller: str,
    payload: dict[str, Any],
    setup: PolymerSetup,
    system_data: dict[str, Any],
    dimensions: TD3Dimensions,
    artifact_path: Path,
) -> dict[str, Any]:
    y, u, y_sp, rewards = _rollout_arrays(payload)
    prefixed = trajectory_metrics(
        y=y,
        u=u,
        y_sp_scaled_dev=y_sp,
        setup=setup,
        system_data=system_data,
        inputs_number=dimensions.inputs_number,
        prefix="controller",
    )
    record = {
        "mode": mode,
        "controller": controller,
        "artifact_path": relative_to_repo(artifact_path),
        "nFE": int(payload.get("nFE", len(rewards))),
        "time_in_sub_episodes": int(payload.get("time_in_sub_episodes", 0)),
        "reward_mean": float(np.nanmean(rewards)) if rewards.size else None,
        "eta_rmse": prefixed["controller_eta_rmse"],
        "T_rmse": prefixed["controller_T_rmse"],
        "mean_rmse": prefixed["controller_mean_rmse"],
        "eta_iae": prefixed["controller_eta_iae"],
        "T_iae": prefixed["controller_T_iae"],
        "mean_abs_du": prefixed["controller_mean_abs_du"],
    }
    record.update(_diagnostic_summary(payload))
    return record


def run_pretrained_lmpc_comparison(config: LMPCComparisonRunConfig) -> dict[str, Any]:
    validate_lmpc_comparison_config(config)
    device = resolve_device(config.device_requested)
    set_seed(config.seed)

    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = resolve_repo_path(config.output_root, create=True)
    run_dir = output_root / run_timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    setup = build_polymer_setup()
    system_data = load_of_mpc_system_data(setup)
    dimensions = compute_td3_dimensions(system_data["A_aug"], system_data["B_aug"], system_data["C_aug"])
    lmpc = make_lmpc_components(system_data)
    observer_gain = make_observer_gain(system_data["A_aug"], system_data["C_aug"])
    y_sp_scenario = scaled_setpoint_scenario(
        COMPARISON_SETPOINT_Y_PHYS,
        steady_states=setup.steady_states,
        data_min=system_data["data_min"],
        data_max=system_data["data_max"],
        inputs_number=dimensions.inputs_number,
    )
    reward_config, reward_fn = make_lmpc_offline_reward()

    agent_path = resolve_lmpc_pretrained_checkpoint(config.agent_path)
    actor_layers, critic_layers, checkpoint_arch = resolve_checkpoint_layers(
        checkpoint_path=agent_path,
        actor_override=config.actor_layer_sizes,
        critic_override=config.critic_layer_sizes,
        dimensions=dimensions,
    )

    records: list[dict[str, Any]] = []
    artifacts: dict[str, Any] = {}
    for mode in config.modes:
        agent = make_lmpc_td3_agent(
            dimensions,
            buffer_size=300_000,
            device=device,
            actor_hidden=actor_layers,
            critic_hidden=critic_layers,
        )
        agent.load(str(agent_path))
        td3_payload = run_pretrained_td3_lmpc_rollout(
            system=make_polymer_system(setup),
            agent=agent,
            model_obj=lmpc.lmpc_obj,
            y_sp_scenario=y_sp_scenario,
            n_tests=config.n_tests,
            set_points_len=config.set_points_len,
            steady_states=setup.steady_states,
            observer_gain=observer_gain,
            data_min=system_data["data_min"],
            data_max=system_data["data_max"],
            test_cycle=[False] * config.n_tests,
            reward_fn=reward_fn,
            min_max_dict=system_data["min_max_dict"],
            mode=mode,
            disturbance_after_step=config.disturbance_after_step,
        )
        td3_path = run_dir / f"td3_results_{mode}.pickle"
        _save_rollout_payload(td3_path, td3_payload)

        lmpc_payload, lmpc_path, lmpc_generated = load_or_generate_lmpc_baseline(
            config=config,
            setup=setup,
            system_data=system_data,
            lmpc=lmpc,
            y_sp_scenario=y_sp_scenario,
            observer_gain=observer_gain,
            reward_fn=reward_fn,
            mode=mode,
        )
        of_payload, of_path, of_generated = load_or_generate_of_mpc_diagnostic_baseline(
            config=config,
            setup=setup,
            system_data=system_data,
            lmpc=lmpc,
            y_sp_scenario=y_sp_scenario,
            observer_gain=observer_gain,
            reward_fn=reward_fn,
            mode=mode,
        )

        plot_dir = plot_mpc_rl_results_cstr(
            td3_payload["y_sp"],
            setup.steady_states,
            td3_payload["nFE"],
            setup.delta_t,
            td3_payload["time_in_sub_episodes"],
            lmpc_payload["y_system"],
            lmpc_payload["u_applied_phys"],
            td3_payload["y_system"],
            td3_payload["u_applied_phys"],
            system_data["data_min"],
            system_data["data_max"],
            directory=run_dir,
            prefix_name=f"lmpc_pretrain_td3_vs_lmpc_{mode}",
        )

        records.append(
            controller_metric_record(
                mode=mode,
                controller="td3",
                payload=td3_payload,
                setup=setup,
                system_data=system_data,
                dimensions=dimensions,
                artifact_path=td3_path,
            )
        )
        records.append(
            controller_metric_record(
                mode=mode,
                controller="direct_lmpc",
                payload=lmpc_payload,
                setup=setup,
                system_data=system_data,
                dimensions=dimensions,
                artifact_path=lmpc_path,
            )
        )
        records.append(
            controller_metric_record(
                mode=mode,
                controller="offset_free_mpc",
                payload=of_payload,
                setup=setup,
                system_data=system_data,
                dimensions=dimensions,
                artifact_path=of_path,
            )
        )
        artifacts[mode] = {
            "td3_results_path": relative_to_repo(td3_path),
            "lmpc_baseline_path": relative_to_repo(lmpc_path),
            "of_mpc_baseline_path": relative_to_repo(of_path),
            "plot_dir": plot_dir,
            "lmpc_baseline_generated": bool(lmpc_generated),
            "of_mpc_baseline_generated": bool(of_generated),
        }

    metrics_json = run_dir / "comparison_metrics.json"
    metrics_csv = run_dir / "comparison_metrics.csv"
    write_json(metrics_json, {"records": records})
    write_csv(metrics_csv, records)

    summary = {
        "status": "completed",
        "run_timestamp": run_timestamp,
        "agent_path": relative_to_repo(agent_path),
        "run_dir": relative_to_repo(run_dir),
        "modes": list(config.modes),
        "n_tests": int(config.n_tests),
        "set_points_len": int(config.set_points_len),
        "disturbance_after_step": bool(config.disturbance_after_step),
        "td3_dimensions": dimensions,
        "checkpoint_architecture": checkpoint_arch,
        "actor_layers_used": actor_layers,
        "critic_layers_used": critic_layers,
        "controller": {
            "Qy_mpc_diag": QY_MPC,
            "Su_mpc_diag": SU_MPC,
            "Rdu_mpc_diag": RDU_MPC,
            "rho_lyap": RHO_LYAP,
            "lyap_eps": LYAP_EPS,
            "target_mode": TARGET_MODE,
            "target_config": lmpc.target_config,
            "lyapunov_mode": LYAPUNOV_MODE,
            "first_step_contraction_on": FIRST_STEP_CONTRACTION_ON,
            "use_target_output_for_tracking": USE_TARGET_OUTPUT_FOR_TRACKING,
        },
        "scaling": {
            "state_bounds_source": system_data.get("state_bounds_source"),
            "setpoint_bounds_source": system_data.get("setpoint_bounds_source"),
            "setpoint_scaler_y_phys": jsonable(system_data.get("setpoint_range_y_used")),
            "comparison_setpoint_y_phys": jsonable(COMPARISON_SETPOINT_Y_PHYS),
            "y_sp_min": jsonable(system_data["min_max_dict"]["y_sp_min"]),
            "y_sp_max": jsonable(system_data["min_max_dict"]["y_sp_max"]),
        },
        "reward_config": reward_config,
        "metrics_json": relative_to_repo(metrics_json),
        "metrics_csv": relative_to_repo(metrics_csv),
        "artifacts": artifacts,
    }
    summary_path = run_dir / "summary.json"
    write_json(summary_path, summary)
    return {
        "run_dir": run_dir,
        "summary_path": summary_path,
        "metrics_json": metrics_json,
        "metrics_csv": metrics_csv,
        "records": records,
        "summary": summary,
    }


__all__ = [
    "DEFAULT_ACTOR_EPOCHS",
    "DEFAULT_ACTOR_LAYER_SIZES",
    "DEFAULT_CANDIDATE_CHUNK_SIZE",
    "DEFAULT_CRITIC_EPOCHS",
    "DEFAULT_CRITIC_LAYER_SIZES",
    "DEFAULT_LMPC_SAMPLES",
    "DEFAULT_MAX_ATTEMPT_MULTIPLIER",
    "DEFAULT_PRETRAIN_BATCH_SIZE",
    "DEFAULT_STEADY_SAMPLES",
    "DEFAULT_WORKER_BATCH_SIZE",
    "LMPCComparisonRunConfig",
    "LMPCPretrainingRunConfig",
    "infer_checkpoint_architecture",
    "mode_list",
    "run_lmpc_pretraining",
    "run_pretrained_lmpc_comparison",
]
