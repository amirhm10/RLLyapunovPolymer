from __future__ import annotations

import argparse
import csv
import json
import os
import random
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from Simulation.mpc import MpcSolver
from Simulation.system_functions import PolymerCSTR
from TD3Agent.agent import TD3Agent
from utils.path_helpers import repo_path, resolve_repo_path
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
Q_REWARD = np.diag(Q_MPC)
R_REWARD = np.diag(R_MPC)

SETPOINT_Y_PHYS = np.array(
    [
        [2.8, 320.0],
        [5.0, 326.0],
    ],
    dtype=float,
)
U_MIN_PHYS = np.array([71.6, 78.0], dtype=float)
U_MAX_PHYS = np.array([870.0, 670.0], dtype=float)

STATE_DIM_EXPECTED = 13
ACTION_DIM_EXPECTED = 2
ACTOR_LAYER_SIZES = [512, 512, 512, 512, 512]
CRITIC_LAYER_SIZES = [512, 512, 512, 512, 512]


@dataclass(frozen=True)
class Preset:
    mpc_samples: int
    steady_samples: int
    chunk_size: int
    actor_epochs: int
    critic_epochs: int


PRESETS = {
    "smoke": Preset(
        mpc_samples=32,
        steady_samples=8,
        chunk_size=16,
        actor_epochs=1,
        critic_epochs=1,
    ),
    "legacy-full": Preset(
        mpc_samples=4_900_000,
        steady_samples=100_000,
        chunk_size=10_000,
        actor_epochs=1000,
        critic_epochs=500,
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pretrain a TD3 actor/critic from offset-free MPC expert samples.",
    )
    parser.add_argument("--preset", choices=sorted(PRESETS), default="smoke")
    parser.add_argument("--mpc-samples", type=int, default=None)
    parser.add_argument("--steady-samples", type=int, default=None)
    parser.add_argument("--chunk-size", type=int, default=None)
    parser.add_argument("--actor-epochs", type=int, default=None)
    parser.add_argument("--critic-epochs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Use auto to select cuda when available, otherwise cpu.",
    )
    parser.add_argument(
        "--output-root",
        default=os.path.join("results", "PretrainOFMPC"),
        help="Artifact root. Relative paths are resolved from the repository root.",
    )
    return parser.parse_args()


def value_or_default(value: int | None, default: int) -> int:
    return default if value is None else int(value)


def resolve_run_config(args: argparse.Namespace) -> dict[str, Any]:
    preset = PRESETS[args.preset]
    config = {
        "preset": args.preset,
        "mpc_samples": value_or_default(args.mpc_samples, preset.mpc_samples),
        "steady_samples": value_or_default(args.steady_samples, preset.steady_samples),
        "chunk_size": value_or_default(args.chunk_size, preset.chunk_size),
        "actor_epochs": value_or_default(args.actor_epochs, preset.actor_epochs),
        "critic_epochs": value_or_default(args.critic_epochs, preset.critic_epochs),
        "seed": int(args.seed),
        "device_requested": str(args.device),
        "output_root": str(args.output_root),
    }

    if config["mpc_samples"] < 0 or config["steady_samples"] < 0:
        raise ValueError("--mpc-samples and --steady-samples must be nonnegative.")
    if config["mpc_samples"] + config["steady_samples"] <= 0:
        raise ValueError("At least one replay sample is required.")
    if config["chunk_size"] <= 0:
        raise ValueError("--chunk-size must be positive.")
    if config["actor_epochs"] < 0 or config["critic_epochs"] < 0:
        raise ValueError("--actor-epochs and --critic-epochs must be nonnegative.")
    if config["actor_epochs"] + config["critic_epochs"] <= 0:
        raise ValueError("At least one actor or critic pretraining epoch is required.")

    return config


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


def build_polymer_steady_states() -> dict[str, np.ndarray]:
    ad = 2.142e17
    ed = 14897
    ap = 3.816e10
    ep = 3557
    at = 4.50e12
    et = 843
    fi = 0.6
    minus_delta_h_r = -6.99e4
    ha = 1.05e6
    rhocp = 1506
    rhoccpc = 4043
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
    cstr_ss = PolymerCSTR(
        system_params,
        system_design_params,
        steady_inputs,
        delta_t=0.5,
        deviation_form=False,
    )
    return {
        "ss_inputs": steady_inputs.copy(),
        "y_ss": cstr_ss.y_ss.copy(),
    }


def make_agent(
    state_dim: int,
    action_dim: int,
    buffer_size: int,
    device: torch.device,
) -> TD3Agent:
    return TD3Agent(
        state_dim=state_dim,
        action_dim=action_dim,
        actor_hidden=ACTOR_LAYER_SIZES,
        critic_hidden=CRITIC_LAYER_SIZES,
        gamma=0.995,
        actor_lr=1e-4,
        critic_lr=1e-4,
        batch_size=256,
        policy_delay=4,
        max_action=1.0,
        tau=0.005,
        buffer_size=buffer_size,
        device=device,
        mode="mpc",
    )


def jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(jsonable(payload), handle, indent=2, sort_keys=True)
        handle.write("\n")


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


def save_loss_artifacts(run_dir: Path, agent: TD3Agent) -> dict[str, str | None]:
    losses = {
        "actor_losses": [float(v) for v in agent.actor_losses],
        "actor_bc_losses": [float(v) for v in agent.actor_bc_losses],
        "critic_losses": [float(v) for v in agent.critic_losses],
    }

    losses_json = run_dir / "loss_arrays.json"
    write_json(losses_json, losses)

    losses_csv = run_dir / "loss_arrays.csv"
    if any(losses.values()):
        max_len = max(len(values) for values in losses.values())
        with losses_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["index", *losses.keys()])
            writer.writeheader()
            for idx in range(max_len):
                row = {"index": idx}
                for name, values in losses.items():
                    row[name] = values[idx] if idx < len(values) else ""
                writer.writerow(row)
        losses_csv_path: str | None = relative_to_repo(losses_csv)
    else:
        losses_csv_path = None

    return {
        "loss_arrays_json": relative_to_repo(losses_json),
        "loss_arrays_csv": losses_csv_path,
    }


def main() -> None:
    args = parse_args()
    config = resolve_run_config(args)
    device = resolve_device(config["device_requested"])
    config["device_used"] = str(device)
    set_seed(config["seed"])

    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = resolve_repo_path(config["output_root"], create=True)
    run_dir = output_root / run_timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"Writing OF-MPC TD3 pretraining artifacts to: {run_dir}")
    wall_start = time.perf_counter()

    steady_states = build_polymer_steady_states()
    system_data = load_and_prepare_system_data(
        steady_states=steady_states,
        setpoint_y=SETPOINT_Y_PHYS,
        u_min=U_MIN_PHYS,
        u_max=U_MAX_PHYS,
        system_dict_path=os.path.join("Data", "system_dict"),
        augmentation_style="rawlings",
        augmentation_mode="output_disturbance",
    )

    a_aug = system_data["A_aug"]
    b_aug = system_data["B_aug"]
    c_aug = system_data["C_aug"]
    inputs_number = int(b_aug.shape[1])
    outputs_number = int(c_aug.shape[0])
    state_dim = int(a_aug.shape[0] + outputs_number + inputs_number)
    action_dim = int(inputs_number)

    if state_dim != STATE_DIM_EXPECTED or action_dim != ACTION_DIM_EXPECTED:
        raise RuntimeError(
            "Unexpected TD3 dimensions for the OF-MPC pretraining workflow: "
            f"state_dim={state_dim}, action_dim={action_dim}."
        )

    total_samples = int(config["mpc_samples"] + config["steady_samples"])
    agent = make_agent(
        state_dim=state_dim,
        action_dim=action_dim,
        buffer_size=total_samples,
        device=device,
    )

    mpc_obj = MpcSolver(
        a_aug,
        b_aug,
        c_aug,
        Q_out=Q_MPC,
        R_in=R_MPC,
        NP=PREDICT_HORIZON,
        NC=CONTROL_HORIZON,
    )
    bnds = tuple(
        (float(system_data["b_min"][idx]), float(system_data["b_max"][idx]))
        for _ in range(CONTROL_HORIZON)
        for idx in range(inputs_number)
    )
    ic_opt = np.zeros(inputs_number * CONTROL_HORIZON, dtype=float)
    cons = ()

    buffer_start = time.perf_counter()
    if config["mpc_samples"] > 0:
        filling_the_buffer(
            min_max_dict=system_data["min_max_dict"],
            A=a_aug,
            B=b_aug,
            C=c_aug,
            MPC_obj=mpc_obj,
            mpc_pretrain_samples_numbers=config["mpc_samples"],
            Q_penalty=Q_REWARD,
            R_penalty=R_REWARD,
            agent=agent,
            IC_opt=ic_opt,
            bnds=bnds,
            cons=cons,
            chunk_size=config["chunk_size"],
        )
    if config["steady_samples"] > 0:
        add_steady_state_samples(
            min_max_dict=system_data["min_max_dict"],
            A=a_aug,
            B=b_aug,
            C=c_aug,
            MPC_obj=mpc_obj,
            steady_state_samples_numbers=config["steady_samples"],
            Q_penalty=Q_REWARD,
            R_penalty=R_REWARD,
            agent=agent,
            IC_opt=ic_opt,
            bnds=bnds,
            cons=cons,
            chunk_size=config["chunk_size"],
        )
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
        batch_size=min(agent.batch_size, buffer_size),
        shuffle=True,
        drop_last=False,
        pin_memory=(device.type == "cuda"),
    )

    train_start = time.perf_counter()
    agent.pretrain_from_buffer(
        num_actor_epochs=config["actor_epochs"],
        num_critic_epochs=config["critic_epochs"],
        data_loader=data_loader,
        use_target_noise_critic=True,
        log_interval=1,
        mode="mpc",
    )
    train_seconds = float(time.perf_counter() - train_start)

    checkpoint_path = Path(
        agent.save(
            str(run_dir),
            prefix="of_mpc_pretrained_td3",
            include_optim=False,
        )
    )
    loss_paths = save_loss_artifacts(run_dir, agent)

    full_config = {
        "run_timestamp": run_timestamp,
        "method": "td3_pretraining_from_offset_free_mpc",
        "preset_defaults": {name: asdict(preset) for name, preset in PRESETS.items()},
        "run_config": config,
        "controller": {
            "augmentation_style": "rawlings",
            "augmentation_mode": "output_disturbance",
            "setpoint_y_phys": SETPOINT_Y_PHYS,
            "u_min_phys": U_MIN_PHYS,
            "u_max_phys": U_MAX_PHYS,
            "predict_horizon": PREDICT_HORIZON,
            "control_horizon": CONTROL_HORIZON,
            "Q_mpc": Q_MPC,
            "R_mpc": R_MPC,
            "Q_reward": Q_REWARD,
            "R_reward": R_REWARD,
        },
        "td3": {
            "state_dim": state_dim,
            "action_dim": action_dim,
            "actor_hidden": ACTOR_LAYER_SIZES,
            "critic_hidden": CRITIC_LAYER_SIZES,
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
            "steady_states": steady_states,
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
        "buffer_generation_seconds": buffer_seconds,
        "pretraining_seconds": train_seconds,
        "buffer_size": buffer_size,
        "mpc_samples": int(config["mpc_samples"]),
        "steady_samples": int(config["steady_samples"]),
        "state_dim": state_dim,
        "action_dim": action_dim,
        "checkpoint_path": relative_to_repo(checkpoint_path),
        "config_path": relative_to_repo(config_path),
        **loss_paths,
        "reward_stats": array_stats(agent.buffer.rewards[:buffer_size]),
        "action_stats": array_stats(agent.buffer.actions[:buffer_size]),
        "state_stats": array_stats(agent.buffer.states[:buffer_size]),
    }
    summary_path = run_dir / "summary.json"
    write_json(summary_path, summary)

    print("OF-MPC TD3 pretraining complete.")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Config: {config_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
