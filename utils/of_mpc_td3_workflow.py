from __future__ import annotations

import csv
import json
import os
import pickle
import random
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from Plotting_fns.mpc_plot_fns import plot_mpc_rl_results_cstr
from Simulation.mpc import MpcSolver, compute_observer_gain
from Simulation.mpc_run import run_mpc
from Simulation.run_rl import run_rl_pre_trained
from Simulation.system_functions import PolymerCSTR
from TD3Agent.agent import TD3Agent
from TD3Agent.reward_functions import make_reward_fn_mpc_quadratic
from utils.path_helpers import repo_path, resolve_repo_path
from utils.polymer_td3_defaults import (
    DEFAULT_DIRECT_SETPOINT_Y_PHYS,
    DEFAULT_TD3_SETPOINT_SCALER_Y_PHYS,
    DEFAULT_U_MAX_PHYS,
    DEFAULT_U_MIN_PHYS,
)
from utils.scaling_helpers import apply_min_max, reverse_min_max
from utils.td3_helpers import (
    ReplayDataset,
    add_steady_state_samples,
    filling_the_buffer,
    load_and_prepare_system_data,
)


PREDICT_HORIZON = 9
CONTROL_HORIZON = 3
Q_MPC = np.array([5.0, 1.0], dtype=float)
R_MPC = np.array([1.0, 1.0], dtype=float)
# Keep these separate from Q_MPC/R_MPC: they label TD3 replay rewards only.
Q_REWARD_DIAG = np.array([12.0, 6.0], dtype=float)
R_REWARD_DIAG = np.array([1.0, 1.0], dtype=float)
Q_REWARD = np.diag(Q_REWARD_DIAG)
R_REWARD = np.diag(R_REWARD_DIAG)

PRETRAIN_SETPOINT_Y_PHYS = DEFAULT_TD3_SETPOINT_SCALER_Y_PHYS.copy()
COMPARISON_SETPOINT_Y_PHYS = DEFAULT_DIRECT_SETPOINT_Y_PHYS.copy()
U_MIN_PHYS = DEFAULT_U_MIN_PHYS.copy()
U_MAX_PHYS = DEFAULT_U_MAX_PHYS.copy()

OBSERVER_POLES = np.array(
    [
        0.44619852,
        0.33547649,
        0.36380595,
        0.70467118,
        0.3562966,
        0.42900673,
        0.4228262,
        0.96916776,
        0.91230187,
    ],
    dtype=float,
)

NOMINAL_QS = 459.0
NOMINAL_QI = 108.0
NOMINAL_HA = 1.05e6
QI_CHANGE = 0.95
QS_CHANGE = 1.05
HA_CHANGE = 0.92


@dataclass(frozen=True)
class PolymerSetup:
    system_params: np.ndarray
    system_design_params: np.ndarray
    system_steady_state_inputs: np.ndarray
    delta_t: float
    steady_states: dict[str, np.ndarray]


@dataclass(frozen=True)
class TD3Dimensions:
    state_dim: int
    action_dim: int
    inputs_number: int
    set_points_number: int


@dataclass(frozen=True)
class OFMPCComponents:
    mpc_obj: MpcSolver
    ic_opt: np.ndarray
    bnds: tuple[tuple[float, float], ...]
    cons: tuple


@dataclass(frozen=True)
class PretrainingRunConfig:
    mpc_samples: int
    steady_samples: int
    chunk_size: int
    actor_epochs: int
    critic_epochs: int
    pretrain_batch_size: int
    actor_layer_sizes: tuple[int, ...]
    critic_layer_sizes: tuple[int, ...]
    checkpoint_interval_epochs: int = 25
    seed: int = 123
    device_requested: str = "auto"
    output_root: str = os.path.join("results", "PretrainOFMPC")


@dataclass(frozen=True)
class ComparisonRunConfig:
    actor_layer_sizes: tuple[int, ...]
    critic_layer_sizes: tuple[int, ...]
    agent_path: str | None = None
    modes: tuple[str, ...] = ("nominal", "disturb")
    n_tests: int = 2
    set_points_len: int = 400
    seed: int = 123
    device_requested: str = "auto"
    output_root: str = os.path.join("results", "PretrainOFMPCComparison")
    baseline_cache_dir: str = os.path.join("results", "PretrainOFMPCComparison", "baselines")
    force_baseline_refresh: bool = False


def validate_pretraining_config(config: PretrainingRunConfig) -> None:
    if config.mpc_samples < 0 or config.steady_samples < 0:
        raise ValueError("mpc_samples and steady_samples must be nonnegative.")
    if config.mpc_samples + config.steady_samples <= 0:
        raise ValueError("At least one replay sample is required.")
    if config.chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    if config.actor_epochs < 0 or config.critic_epochs < 0:
        raise ValueError("actor_epochs and critic_epochs must be nonnegative.")
    if config.actor_epochs + config.critic_epochs <= 0:
        raise ValueError("At least one actor or critic pretraining epoch is required.")
    if config.pretrain_batch_size <= 0:
        raise ValueError("pretrain_batch_size must be positive.")
    if config.checkpoint_interval_epochs < 0:
        raise ValueError("checkpoint_interval_epochs must be nonnegative.")
    validate_layer_sizes(config.actor_layer_sizes, "actor_layer_sizes")
    validate_layer_sizes(config.critic_layer_sizes, "critic_layer_sizes")


def validate_comparison_config(config: ComparisonRunConfig) -> None:
    if config.n_tests <= 0:
        raise ValueError("n_tests must be positive.")
    if config.set_points_len <= 0:
        raise ValueError("set_points_len must be positive.")
    if not config.modes:
        raise ValueError("At least one comparison mode is required.")
    unknown = [mode for mode in config.modes if mode not in {"nominal", "disturb"}]
    if unknown:
        raise ValueError(f"Unsupported modes: {unknown}")
    validate_layer_sizes(config.actor_layer_sizes, "actor_layer_sizes")
    validate_layer_sizes(config.critic_layer_sizes, "critic_layer_sizes")


def validate_layer_sizes(layer_sizes: tuple[int, ...], name: str) -> None:
    if not layer_sizes:
        raise ValueError(f"{name} must contain at least one hidden layer size.")
    bad = [value for value in layer_sizes if int(value) <= 0]
    if bad:
        raise ValueError(f"{name} entries must be positive integers; got {bad}.")


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")
    return torch.device(name)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_polymer_setup() -> PolymerSetup:
    ad = 2.142e17
    ed = 14897.0
    ap = 3.816e10
    ep = 3557.0
    at = 4.50e12
    et = 843.0
    fi = 0.6
    minus_delta_h_r = -6.99e4
    ha = 1.05e6
    rhocp = 1506.0
    rhoccpc = 4043.0
    mm = 104.14
    system_params = np.array(
        [ad, ed, ap, ep, at, et, fi, minus_delta_h_r, ha, rhocp, rhoccpc, mm],
        dtype=float,
    )

    cif = 0.5888
    cmf = 8.6981
    qi = 108.0
    qs = 459.0
    tf = 330.0
    tcf = 295.0
    v = 3000.0
    vc = 3312.4
    system_design_params = np.array([cif, cmf, qi, qs, tf, tcf, v, vc], dtype=float)

    qc_ss = 471.6
    qm_ss = 378.0
    steady_inputs = np.array([qc_ss, qm_ss], dtype=float)
    delta_t = 0.5
    cstr_ss = PolymerCSTR(
        system_params,
        system_design_params,
        steady_inputs,
        delta_t,
        deviation_form=False,
    )
    steady_states = {
        "ss_inputs": steady_inputs.copy(),
        "y_ss": cstr_ss.y_ss.copy(),
    }
    return PolymerSetup(
        system_params=system_params,
        system_design_params=system_design_params,
        system_steady_state_inputs=steady_inputs,
        delta_t=delta_t,
        steady_states=steady_states,
    )


def make_polymer_system(setup: PolymerSetup) -> PolymerCSTR:
    return PolymerCSTR(
        setup.system_params,
        setup.system_design_params,
        setup.system_steady_state_inputs,
        setup.delta_t,
        deviation_form=False,
    )


def load_of_mpc_system_data(
    setup: PolymerSetup,
    *,
    setpoint_y_phys: np.ndarray = PRETRAIN_SETPOINT_Y_PHYS,
) -> dict[str, Any]:
    return load_and_prepare_system_data(
        steady_states=setup.steady_states,
        setpoint_y=setpoint_y_phys,
        u_min=U_MIN_PHYS,
        u_max=U_MAX_PHYS,
        system_dict_path=os.path.join("Data", "system_dict"),
        augmentation_style="rawlings",
        augmentation_mode="output_disturbance",
    )


def compute_td3_dimensions(A_aug: np.ndarray, B_aug: np.ndarray, C_aug: np.ndarray) -> TD3Dimensions:
    inputs_number = int(B_aug.shape[1])
    set_points_number = int(C_aug.shape[0])
    state_dim = int(A_aug.shape[0]) + set_points_number + inputs_number
    action_dim = inputs_number
    return TD3Dimensions(
        state_dim=state_dim,
        action_dim=action_dim,
        inputs_number=inputs_number,
        set_points_number=set_points_number,
    )


def make_td3_agent(
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
        actor_lr=1e-4,
        critic_lr=3e-4,
        batch_size=256,
        policy_delay=4,
        target_policy_smoothing_noise_std=0.05,
        noise_clip=0.1,
        max_action=1.0,
        tau=0.005,
        buffer_size=buffer_size,
        device=device,
        mode="mpc",
    )


def make_of_mpc_components(
    system_data: dict[str, Any],
    *,
    predict_horizon: int = PREDICT_HORIZON,
    control_horizon: int = CONTROL_HORIZON,
    q_mpc: np.ndarray = Q_MPC,
    r_mpc: np.ndarray = R_MPC,
) -> OFMPCComponents:
    A_aug = system_data["A_aug"]
    B_aug = system_data["B_aug"]
    C_aug = system_data["C_aug"]
    inputs_number = int(B_aug.shape[1])
    mpc_obj = MpcSolver(
        A_aug,
        B_aug,
        C_aug,
        Q_out=q_mpc,
        R_in=r_mpc,
        NP=predict_horizon,
        NC=control_horizon,
    )
    bnds = tuple(
        (float(system_data["b_min"][idx]), float(system_data["b_max"][idx]))
        for _ in range(control_horizon)
        for idx in range(inputs_number)
    )
    ic_opt = np.zeros(inputs_number * control_horizon, dtype=float)
    return OFMPCComponents(
        mpc_obj=mpc_obj,
        ic_opt=ic_opt,
        bnds=bnds,
        cons=(),
    )


def make_observer_gain(A_aug: np.ndarray, C_aug: np.ndarray) -> np.ndarray:
    return compute_observer_gain(A_aug, C_aug, OBSERVER_POLES)


def scaled_setpoint_scenario(
    setpoint_y_phys: np.ndarray,
    *,
    steady_states: dict[str, np.ndarray],
    data_min: np.ndarray,
    data_max: np.ndarray,
    inputs_number: int,
) -> np.ndarray:
    return apply_min_max(
        setpoint_y_phys,
        data_min[inputs_number:],
        data_max[inputs_number:],
    ) - apply_min_max(
        steady_states["y_ss"],
        data_min[inputs_number:],
        data_max[inputs_number:],
    )


def jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "__dataclass_fields__"):
        return jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(jsonable(payload), handle, indent=2, sort_keys=True)
        handle.write("\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows([{key: jsonable(value) for key, value in row.items()} for row in rows])


def relative_to_repo(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(repo_path().resolve()))
    except ValueError:
        return str(path.resolve())


def array_stats(array: np.ndarray) -> dict[str, Any]:
    if array.size == 0:
        return {"count": 0}
    return {
        "count": int(array.shape[0]),
        "mean": jsonable(np.mean(array, axis=0)),
        "std": jsonable(np.std(array, axis=0)),
        "min": jsonable(np.min(array, axis=0)),
        "max": jsonable(np.max(array, axis=0)),
    }


def save_partial_replay_buffer(run_dir: Path, agent: TD3Agent, buffer_size: int) -> str | None:
    buffer_size = int(buffer_size)
    if buffer_size <= 0:
        return None

    path = run_dir / "of_mpc_replay_partial.npz"
    np.savez_compressed(
        path,
        states=agent.buffer.states[:buffer_size],
        actions=agent.buffer.actions[:buffer_size],
        rewards=agent.buffer.rewards[:buffer_size],
        next_states=agent.buffer.next_states[:buffer_size],
        dones=agent.buffer.dones[:buffer_size],
    )
    return relative_to_repo(path)


def loss_series_stats(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0}
    array = np.asarray(values, dtype=float)
    return {
        "count": int(array.size),
        "first": float(array[0]),
        "last": float(array[-1]),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
        "mean": float(np.mean(array)),
    }


def save_loss_artifacts(
    run_dir: Path,
    agent: TD3Agent,
    *,
    expected_actor_epochs: int | None = None,
    expected_critic_epochs: int | None = None,
    pretraining_history: dict[str, Any] | None = None,
) -> dict[str, Any]:
    losses = {
        "actor_losses": [float(v) for v in agent.actor_losses],
        "actor_bc_losses": [float(v) for v in agent.actor_bc_losses],
        "critic_losses": [float(v) for v in agent.critic_losses],
    }
    history = pretraining_history or {}

    actor_history_count = len(history.get("actor_bc_losses", []))
    critic_history_count = len(history.get("critic_losses", []))
    actor_logged_count = actor_history_count if pretraining_history is not None else len(losses["actor_bc_losses"])
    critic_logged_count = critic_history_count if pretraining_history is not None else len(losses["critic_losses"])

    validation_issues: list[str] = []
    if expected_actor_epochs is not None and expected_actor_epochs > 0:
        if actor_logged_count == 0:
            validation_issues.append("actor behavioral-cloning losses are empty")
        elif actor_logged_count < expected_actor_epochs:
            validation_issues.append(
                f"actor behavioral-cloning losses have {actor_logged_count} entries, "
                f"expected at least {expected_actor_epochs}"
            )
    if expected_critic_epochs is not None and expected_critic_epochs > 0:
        if critic_logged_count == 0:
            validation_issues.append("critic TD losses are empty")
        elif critic_logged_count < expected_critic_epochs:
            validation_issues.append(
                f"critic TD losses have {critic_logged_count} entries, "
                f"expected at least {expected_critic_epochs}"
            )

    if validation_issues:
        raise RuntimeError(
            "Pretraining loss logging failed validation: " + "; ".join(validation_issues)
        )

    losses_json = run_dir / "loss_arrays.json"
    write_json(losses_json, losses)

    history_json_path: str | None = None
    if pretraining_history is not None:
        history_json = run_dir / "pretraining_history.json"
        write_json(history_json, pretraining_history)
        history_json_path = relative_to_repo(history_json)

    losses_csv = run_dir / "loss_arrays.csv"
    if any(losses.values()):
        history_columns = {
            "actor_bc_lr": [float(v) for v in history.get("actor_bc_lrs", [])],
            "critic_lr": [float(v) for v in history.get("critic_lrs", [])],
            "actor_bc_samples": [int(v) for v in history.get("actor_bc_samples", [])],
            "critic_samples": [int(v) for v in history.get("critic_samples", [])],
        }
        all_columns = {**losses, **history_columns}
        max_len = max(len(values) for values in all_columns.values())
        rows = []
        for idx in range(max_len):
            row = {"index": idx}
            for name, values in all_columns.items():
                row[name] = values[idx] if idx < len(values) else ""
            rows.append(row)
        write_csv(losses_csv, rows)
        losses_csv_path: str | None = relative_to_repo(losses_csv)
    else:
        losses_csv_path = None

    loss_summary = {
        "loss_logging_ok": True,
        "expected_actor_epochs": None if expected_actor_epochs is None else int(expected_actor_epochs),
        "expected_critic_epochs": None if expected_critic_epochs is None else int(expected_critic_epochs),
        "counts": {name: len(values) for name, values in losses.items()},
        "history_counts": {
            "actor_bc_losses": actor_history_count,
            "critic_losses": critic_history_count,
            "actor_bc_lrs": len(history.get("actor_bc_lrs", [])),
            "critic_lrs": len(history.get("critic_lrs", [])),
            "actor_bc_samples": len(history.get("actor_bc_samples", [])),
            "critic_samples": len(history.get("critic_samples", [])),
        },
        "series": {name: loss_series_stats(values) for name, values in losses.items()},
    }
    loss_summary_json = run_dir / "loss_summary.json"
    write_json(loss_summary_json, loss_summary)

    return {
        "loss_arrays_json": relative_to_repo(losses_json),
        "loss_arrays_csv": losses_csv_path,
        "loss_summary_json": relative_to_repo(loss_summary_json),
        "pretraining_history_json": history_json_path,
        "loss_logging_ok": True,
        "loss_counts": loss_summary["counts"],
    }


def run_of_mpc_pretraining(config: PretrainingRunConfig) -> dict[str, Any]:
    validate_pretraining_config(config)
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
    total_samples = int(config.mpc_samples + config.steady_samples)
    agent = make_td3_agent(
        dimensions,
        buffer_size=total_samples,
        device=device,
        actor_hidden=config.actor_layer_sizes,
        critic_hidden=config.critic_layer_sizes,
    )
    of_mpc = make_of_mpc_components(system_data)

    full_config = {
        "run_timestamp": run_timestamp,
        "method": "td3_pretraining_from_offset_free_mpc",
        "run_config": asdict(config),
        "device_used": str(device),
        "controller": {
            "augmentation_style": "rawlings",
            "augmentation_mode": "output_disturbance",
            "setpoint_y_phys": PRETRAIN_SETPOINT_Y_PHYS,
            "u_min_phys": U_MIN_PHYS,
            "u_max_phys": U_MAX_PHYS,
            "predict_horizon": PREDICT_HORIZON,
            "control_horizon": CONTROL_HORIZON,
            "Q_mpc": Q_MPC,
            "R_mpc": R_MPC,
            "Q_reward_diag": Q_REWARD_DIAG,
            "R_reward_diag": R_REWARD_DIAG,
            "Q_reward": Q_REWARD,
            "R_reward": R_REWARD,
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

    buffer_start = time.perf_counter()
    try:
        if config.mpc_samples > 0:
            filling_the_buffer(
                system_data["min_max_dict"],
                system_data["A_aug"],
                system_data["B_aug"],
                system_data["C_aug"],
                of_mpc.mpc_obj,
                config.mpc_samples,
                Q_REWARD,
                R_REWARD,
                agent,
                of_mpc.ic_opt,
                of_mpc.bnds,
                of_mpc.cons,
                chunk_size=config.chunk_size,
            )
        if config.steady_samples > 0:
            add_steady_state_samples(
                system_data["min_max_dict"],
                system_data["A_aug"],
                system_data["B_aug"],
                system_data["C_aug"],
                of_mpc.mpc_obj,
                config.steady_samples,
                Q_REWARD,
                R_REWARD,
                agent,
                of_mpc.ic_opt,
                of_mpc.bnds,
                of_mpc.cons,
                chunk_size=config.chunk_size,
            )
    except KeyboardInterrupt:
        buffer_seconds = float(time.perf_counter() - buffer_start)
        buffer_size = len(agent.buffer)
        completed_mpc_samples = min(buffer_size, int(config.mpc_samples))
        completed_steady_samples = max(0, buffer_size - int(config.mpc_samples))
        partial_replay_path = save_partial_replay_buffer(run_dir, agent, buffer_size)
        summary = {
            "status": "interrupted_buffer_generation",
            "run_timestamp": run_timestamp,
            "elapsed_seconds": float(time.perf_counter() - wall_start),
            "buffer_generation_seconds": buffer_seconds,
            "pretraining_seconds": 0.0,
            "buffer_size": int(buffer_size),
            "mpc_samples": int(config.mpc_samples),
            "steady_samples": int(config.steady_samples),
            "completed_mpc_samples": int(completed_mpc_samples),
            "completed_steady_samples": int(completed_steady_samples),
            "state_dim": int(dimensions.state_dim),
            "action_dim": int(dimensions.action_dim),
            "checkpoint_path": None,
            "config_path": relative_to_repo(config_path),
            "partial_replay_path": partial_replay_path,
            "loss_logging_ok": False,
            "interrupted_at": {
                "last_phase": "buffer_generation",
                "completed_samples": int(buffer_size),
                "chunk_size": int(config.chunk_size),
            },
            "reward_stats": array_stats(agent.buffer.rewards[:buffer_size]),
            "action_stats": array_stats(agent.buffer.actions[:buffer_size]),
            "state_stats": array_stats(agent.buffer.states[:buffer_size]),
        }
        summary_path = run_dir / "summary.json"
        write_json(summary_path, summary)
        print("OF-MPC TD3 pretraining interrupted during buffer generation.")
        if partial_replay_path is None:
            print("No full replay-buffer chunks had completed before the interrupt.")
        else:
            print(f"Partial replay buffer saved to: {partial_replay_path}")
        return {
            "run_dir": run_dir,
            "checkpoint_path": None,
            "config_path": config_path,
            "summary_path": summary_path,
            "summary": summary,
        }
    buffer_seconds = float(time.perf_counter() - buffer_start)

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

    def finish_run(
        status: str,
        checkpoint_path: Path,
        train_seconds: float,
        loss_paths: dict[str, Any],
        *,
        interrupted_at: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        elapsed_seconds = float(time.perf_counter() - wall_start)
        summary = {
            "status": status,
            "run_timestamp": run_timestamp,
            "elapsed_seconds": elapsed_seconds,
            "buffer_generation_seconds": buffer_seconds,
            "pretraining_seconds": train_seconds,
            "buffer_size": buffer_size,
            "mpc_samples": int(config.mpc_samples),
            "steady_samples": int(config.steady_samples),
            "state_dim": int(dimensions.state_dim),
            "action_dim": int(dimensions.action_dim),
            "checkpoint_path": relative_to_repo(checkpoint_path),
            "config_path": relative_to_repo(config_path),
            **loss_paths,
            "reward_stats": array_stats(agent.buffer.rewards[:buffer_size]),
            "action_stats": array_stats(agent.buffer.actions[:buffer_size]),
            "state_stats": array_stats(agent.buffer.states[:buffer_size]),
        }
        if interrupted_at is not None:
            summary["interrupted_at"] = interrupted_at
        summary_path = run_dir / "summary.json"
        write_json(summary_path, summary)

        return {
            "run_dir": run_dir,
            "checkpoint_path": checkpoint_path,
            "config_path": config_path,
            "summary_path": summary_path,
            "summary": summary,
        }

    train_start = time.perf_counter()
    try:
        pretraining_history = agent.pretrain_from_buffer(
            num_actor_epochs=config.actor_epochs,
            num_critic_epochs=config.critic_epochs,
            data_loader=data_loader,
            use_target_noise_critic=True,
            log_interval=1,
            mode="mpc",
            checkpoint_dir=str(run_dir),
            checkpoint_prefix="of_mpc_pretrained_td3_partial",
            checkpoint_interval_epochs=config.checkpoint_interval_epochs,
            include_checkpoint_optim=False,
        )
    except KeyboardInterrupt:
        train_seconds = float(time.perf_counter() - train_start)
        pretraining_history = getattr(agent, "last_pretraining_history", None)
        try:
            agent.unfreeze_actor()
        except Exception:
            pass
        checkpoint_path = Path(
            agent.save(
                str(run_dir),
                prefix="of_mpc_pretrained_td3_interrupted",
                include_optim=False,
            )
        )
        loss_paths = save_loss_artifacts(
            run_dir,
            agent,
            expected_actor_epochs=None,
            expected_critic_epochs=None,
            pretraining_history=pretraining_history,
        )
        interrupted_at = {}
        if isinstance(pretraining_history, dict):
            interrupted_at = {
                "last_phase": pretraining_history.get("last_phase"),
                "last_actor_epoch": pretraining_history.get("last_actor_epoch", 0),
                "last_critic_epoch": pretraining_history.get("last_critic_epoch", 0),
            }
        print("OF-MPC TD3 pretraining interrupted. Partial checkpoint and summary were saved.")
        return finish_run(
            "interrupted",
            checkpoint_path,
            train_seconds,
            loss_paths,
            interrupted_at=interrupted_at,
        )

    train_seconds = float(time.perf_counter() - train_start)

    loss_paths = save_loss_artifacts(
        run_dir,
        agent,
        expected_actor_epochs=config.actor_epochs,
        expected_critic_epochs=config.critic_epochs,
        pretraining_history=pretraining_history,
    )
    checkpoint_path = Path(
        agent.save(
            str(run_dir),
            prefix="of_mpc_pretrained_td3",
            include_optim=False,
        )
    )

    return finish_run("completed", checkpoint_path, train_seconds, loss_paths)


def latest_pretrained_checkpoint() -> Path | None:
    root = repo_path("results", "PretrainOFMPC")
    if not root.exists():
        return None
    candidates = list(root.glob("**/of_mpc_pretrained_td3_*.pkl"))
    if not candidates:
        return None
    return max(candidates, key=lambda path: (path.stat().st_mtime, str(path)))


def resolve_pretrained_checkpoint(agent_path: str | None) -> Path:
    if agent_path:
        candidate = resolve_repo_path(agent_path)
        if not candidate.exists():
            raise FileNotFoundError(f"TD3 checkpoint not found: {candidate}")
        return candidate

    latest = latest_pretrained_checkpoint()
    if latest is not None:
        return latest

    fallback = repo_path("Data", "agent_2507171027.pkl")
    if not fallback.exists():
        raise FileNotFoundError(
            "No generated OF-MPC pretraining checkpoint was found, and the default "
            f"fallback checkpoint is missing: {fallback}"
        )
    return fallback


def mode_list(value: str) -> tuple[str, ...]:
    value = value.strip().lower()
    if value == "both":
        return ("nominal", "disturb")
    if value in {"nominal", "disturb"}:
        return (value,)
    raise ValueError("--modes must be one of: both, nominal, disturb")


def baseline_cache_path(cache_dir: Path, mode: str, n_tests: int, set_points_len: int) -> Path:
    return cache_dir / f"mpc_results_{mode}_n{int(n_tests)}_len{int(set_points_len)}.pickle"


def save_rollout_pickle(path: Path, results: tuple[Any, ...], *, mode: str, source: str) -> None:
    payload = {
        "source": source,
        "mode": mode,
        "y": results[0],
        "u": results[1],
        "avg_rewards": results[2],
        "rewards": results[3],
        "xhatdhat": results[4],
        "nFE": results[5],
        "time_in_sub_episodes": results[6],
        "y_sp": results[7],
        "yhat": results[8],
        "delta_y_storage": results[9],
        "delta_u_storage": results[10],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(payload, handle)


def load_or_generate_of_mpc_baseline(
    *,
    config: ComparisonRunConfig,
    setup: PolymerSetup,
    system_data: dict[str, Any],
    of_mpc: OFMPCComponents,
    y_sp_scenario: np.ndarray,
    observer_gain: np.ndarray,
    reward_fn: Any,
    mode: str,
) -> tuple[dict[str, Any], Path, bool]:
    cache_dir = resolve_repo_path(config.baseline_cache_dir, create=True)
    path = baseline_cache_path(cache_dir, mode, config.n_tests, config.set_points_len)
    if path.exists() and not config.force_baseline_refresh:
        with path.open("rb") as handle:
            return pickle.load(handle), path, False

    results = run_mpc(
        make_polymer_system(setup),
        of_mpc.mpc_obj,
        y_sp_scenario,
        config.n_tests,
        config.set_points_len,
        setup.steady_states,
        of_mpc.ic_opt.copy(),
        of_mpc.bnds,
        of_mpc.cons,
        0,
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
        mode=mode,
    )
    payload = {
        "source": "offset_free_mpc",
        "mode": mode,
        "y_mpc": results[0],
        "u_mpc": results[1],
        "avg_rewards": results[2],
        "rewards": results[3],
        "xhatdhat": results[4],
        "nFE": results[5],
        "time_in_sub_episodes": results[6],
        "y_sp": results[7],
        "yhat": results[8],
        "delta_y_storage": results[9],
        "delta_u_storage": results[10],
    }
    with path.open("wb") as handle:
        pickle.dump(payload, handle)
    return payload, path, True


def y_sp_phys_from_scaled(
    y_sp_scaled_dev: np.ndarray,
    *,
    steady_states: dict[str, np.ndarray],
    data_min: np.ndarray,
    data_max: np.ndarray,
    inputs_number: int,
) -> np.ndarray:
    y_ss_scaled = apply_min_max(
        steady_states["y_ss"],
        data_min[inputs_number:],
        data_max[inputs_number:],
    )
    y_sp_scaled = np.asarray(y_sp_scaled_dev, dtype=float) + y_ss_scaled
    return reverse_min_max(y_sp_scaled, data_min[inputs_number:], data_max[inputs_number:])


def trajectory_metrics(
    *,
    y: np.ndarray,
    u: np.ndarray,
    y_sp_scaled_dev: np.ndarray,
    setup: PolymerSetup,
    system_data: dict[str, Any],
    inputs_number: int,
    prefix: str,
) -> dict[str, float]:
    y = np.asarray(y, dtype=float)
    u = np.asarray(u, dtype=float)
    y_sp_phys = y_sp_phys_from_scaled(
        y_sp_scaled_dev,
        steady_states=setup.steady_states,
        data_min=system_data["data_min"],
        data_max=system_data["data_max"],
        inputs_number=inputs_number,
    )
    y_aligned = y[1:] if y.shape[0] == y_sp_phys.shape[0] + 1 else y[: y_sp_phys.shape[0]]
    n = min(y_aligned.shape[0], y_sp_phys.shape[0])
    err = y_aligned[:n] - y_sp_phys[:n]
    rmse = np.sqrt(np.nanmean(err ** 2, axis=0))
    iae = np.nansum(np.abs(err), axis=0)
    input_move = np.diff(u, axis=0) if u.shape[0] > 1 else np.zeros_like(u)
    return {
        f"{prefix}_eta_rmse": float(rmse[0]),
        f"{prefix}_T_rmse": float(rmse[1]),
        f"{prefix}_mean_rmse": float(np.nanmean(rmse)),
        f"{prefix}_eta_iae": float(iae[0]),
        f"{prefix}_T_iae": float(iae[1]),
        f"{prefix}_mean_abs_du": float(np.nanmean(np.abs(input_move))) if input_move.size else 0.0,
    }


def compare_metrics(
    *,
    mode: str,
    rl_results: tuple[Any, ...],
    mpc_payload: dict[str, Any],
    setup: PolymerSetup,
    system_data: dict[str, Any],
    inputs_number: int,
    agent_path: Path,
    baseline_path: Path,
    plot_dir: str,
) -> dict[str, Any]:
    rl_metrics = trajectory_metrics(
        y=rl_results[0],
        u=rl_results[1],
        y_sp_scaled_dev=rl_results[7],
        setup=setup,
        system_data=system_data,
        inputs_number=inputs_number,
        prefix="rl",
    )
    mpc_metrics = trajectory_metrics(
        y=mpc_payload["y_mpc"],
        u=mpc_payload["u_mpc"],
        y_sp_scaled_dev=mpc_payload["y_sp"],
        setup=setup,
        system_data=system_data,
        inputs_number=inputs_number,
        prefix="of_mpc",
    )
    return {
        "mode": mode,
        "agent_path": relative_to_repo(agent_path),
        "baseline_path": relative_to_repo(baseline_path),
        "plot_dir": plot_dir,
        "nFE": int(rl_results[5]),
        "time_in_sub_episodes": int(rl_results[6]),
        "rl_reward_mean": float(np.nanmean(rl_results[3])),
        "of_mpc_reward_mean": float(np.nanmean(mpc_payload["rewards"])),
        **rl_metrics,
        **mpc_metrics,
        "rl_minus_of_mpc_mean_rmse": float(rl_metrics["rl_mean_rmse"] - mpc_metrics["of_mpc_mean_rmse"]),
    }


def run_pretrained_of_mpc_comparison(config: ComparisonRunConfig) -> dict[str, Any]:
    validate_comparison_config(config)
    device = resolve_device(config.device_requested)
    set_seed(config.seed)

    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = resolve_repo_path(config.output_root, create=True)
    run_dir = output_root / run_timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    setup = build_polymer_setup()
    system_data = load_of_mpc_system_data(setup)
    dimensions = compute_td3_dimensions(system_data["A_aug"], system_data["B_aug"], system_data["C_aug"])
    of_mpc = make_of_mpc_components(system_data)
    observer_gain = make_observer_gain(system_data["A_aug"], system_data["C_aug"])
    y_sp_scenario = scaled_setpoint_scenario(
        COMPARISON_SETPOINT_Y_PHYS,
        steady_states=setup.steady_states,
        data_min=system_data["data_min"],
        data_max=system_data["data_max"],
        inputs_number=dimensions.inputs_number,
    )
    reward_config, reward_fn = make_reward_fn_mpc_quadratic(
        Q_diag=Q_REWARD_DIAG,
        R_diag=R_REWARD_DIAG,
    )

    agent_path = resolve_pretrained_checkpoint(config.agent_path)
    records: list[dict[str, Any]] = []
    artifacts: dict[str, Any] = {}

    for mode in config.modes:
        agent = make_td3_agent(
            dimensions,
            buffer_size=300_000,
            device=device,
            actor_hidden=config.actor_layer_sizes,
            critic_hidden=config.critic_layer_sizes,
        )
        agent.load(str(agent_path))

        rl_results = run_rl_pre_trained(
            make_polymer_system(setup),
            agent,
            of_mpc.mpc_obj,
            y_sp_scenario,
            config.n_tests,
            config.set_points_len,
            setup.steady_states,
            0,
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
            system_data["min_max_dict"],
            mode=mode,
        )
        rl_pickle_path = run_dir / f"td3_results_{mode}.pickle"
        save_rollout_pickle(rl_pickle_path, rl_results, mode=mode, source="pretrained_td3")

        mpc_payload, baseline_path, baseline_generated = load_or_generate_of_mpc_baseline(
            config=config,
            setup=setup,
            system_data=system_data,
            of_mpc=of_mpc,
            y_sp_scenario=y_sp_scenario,
            observer_gain=observer_gain,
            reward_fn=reward_fn,
            mode=mode,
        )

        plot_dir = plot_mpc_rl_results_cstr(
            rl_results[7],
            setup.steady_states,
            rl_results[5],
            setup.delta_t,
            rl_results[6],
            mpc_payload["y_mpc"],
            mpc_payload["u_mpc"],
            rl_results[0],
            rl_results[1],
            system_data["data_min"],
            system_data["data_max"],
            directory=run_dir,
            prefix_name=f"pre_train_performance_{mode}",
        )
        record = compare_metrics(
            mode=mode,
            rl_results=rl_results,
            mpc_payload=mpc_payload,
            setup=setup,
            system_data=system_data,
            inputs_number=dimensions.inputs_number,
            agent_path=agent_path,
            baseline_path=baseline_path,
            plot_dir=plot_dir,
        )
        record["rl_results_path"] = relative_to_repo(rl_pickle_path)
        record["baseline_generated"] = bool(baseline_generated)
        records.append(record)
        artifacts[mode] = {
            "rl_results_path": relative_to_repo(rl_pickle_path),
            "baseline_path": relative_to_repo(baseline_path),
            "plot_dir": plot_dir,
            "baseline_generated": bool(baseline_generated),
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
        "td3_dimensions": dimensions,
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
