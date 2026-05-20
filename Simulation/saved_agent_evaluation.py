# Saved-agent evaluation for direct Lyapunov safety-gate RL.
#
# This script does not train. It loads the latest saved cold-start and
# pretrained TD3 agents, keeps the direct Lyapunov safety gate active, and
# compares them against offset-free MPC diagnostics and direct Lyapunov MPC on
# a fixed five-episode disturbance suite.

from __future__ import annotations

import csv
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from pprint import pprint

import numpy as np

try:
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except Exception:
    HAS_MATPLOTLIB = False

import torch

from TD3Agent.agent import TD3Agent
from TD3Agent.reward_functions import make_reward_fn_relative_QR
from Simulation.mpc import MpcSolver, compute_observer_gain
from Simulation.run_rl_lyapunov import run_rl_train
from Simulation.system_functions import PolymerCSTR
from Lyapunov.direct_lyapunov_mpc import (
    build_direct_lyapunov_run_bundle,
    design_direct_lyapunov_mpc_solver,
    make_direct_lyapunov_comparison_record,
    run_direct_output_disturbance_lyapunov_mpc,
    run_offset_free_mpc_with_direct_diagnostics,
    save_direct_lyapunov_debug_artifacts,
)
from Lyapunov.safety_debug import (
    build_safety_filter_run_bundle,
    make_safety_filter_comparison_record,
    save_safety_filter_debug_artifacts,
)
from utils.direct_lyapunov_study import DIRECT_TWO_SETPOINT_Y_PHYS
from utils.path_helpers import repo_path
from utils.scaling_helpers import apply_min_max, reverse_min_max
from utils.td3_helpers import load_and_prepare_system_data


AGENT_SOURCE_MODE = "latest"
COLD_AGENT_PATH = None
PRETRAIN_AGENT_PATH = None
EVAL_N_EPISODES = 5
EVAL_SET_POINTS_LEN = 400
EVAL_SCENARIO_SUITE = "nominal_qi_qs_ha_all_step"
DRY_RUN = "--dry-run" in sys.argv
SAVE_CASE_PLOTS = True
FORCE_FINAL_TEST = False

predict_h = 9
cont_h = 3
rho_lyap = 0.99
lyap_eps = 1e-3
lyap_tol = 1e-10
slack_penalty = 1e6
plant_mode = "disturb"
disturbance_after_step = False
use_target_output_for_tracking = False

u_prev_penalty_weight = 0.1
xs_prev_penalty_weight = 0.1

Ad = 2.142e17
Ed = 14897
Ap = 3.816e10
Ep = 3557
At = 4.50e12
Et = 843
fi = 0.6
m_delta_H_r = -6.99e4
hA = 1.05e6
rhocp = 1506
rhoccpc = 4043
Mm = 104.14
system_params = np.array([Ad, Ed, Ap, Ep, At, Et, fi, m_delta_H_r, hA, rhocp, rhoccpc, Mm])

CIf = 0.5888
CMf = 8.6981
Qi = 108.0
Qs = 459.0
Tf = 330.0
Tcf = 295.0
V = 3000.0
Vc = 3312.4
system_design_params = np.array([CIf, CMf, Qi, Qs, Tf, Tcf, V, Vc])

Qm_ss = 378.0
Qc_ss = 471.6
system_steady_state_inputs = np.array([Qc_ss, Qm_ss])
delta_t = 0.5

steady_states = {"ss_inputs": system_steady_state_inputs.copy()}
cstr_ss = PolymerCSTR(system_params, system_design_params, system_steady_state_inputs, delta_t, deviation_form=False)
steady_states["y_ss"] = cstr_ss.y_ss.copy()

u_min = np.array([71.6, 78.0])
u_max = np.array([870.0, 670.0])
setpoint_y_phys = DIRECT_TWO_SETPOINT_Y_PHYS.copy()

n_tests = EVAL_N_EPISODES
set_points_len = EVAL_SET_POINTS_LEN
TEST_CYCLE = [True] * EVAL_N_EPISODES
warm_start = 0
time_in_sub_episodes = int(setpoint_y_phys.shape[0] * set_points_len)
ACTOR_FREEZE = 0

study_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
study_name = "Compare"
study_root = Path(repo_path()) / "results" / study_name / study_timestamp

system_data = load_and_prepare_system_data(
    steady_states=steady_states,
    setpoint_y=setpoint_y_phys,
    u_min=u_min,
    u_max=u_max,
    system_dict_path=os.path.join("Data", "system_dict"),
    augmentation_style="rawlings",
    augmentation_mode="output_disturbance",
)

A_aug = system_data["A_aug"]
B_aug = system_data["B_aug"]
C_aug = system_data["C_aug"]
data_min = system_data["data_min"]
data_max = system_data["data_max"]
min_max_dict = system_data["min_max_dict"]

inputs_number = int(B_aug.shape[1])
y_sp_scenario = apply_min_max(setpoint_y_phys, data_min[inputs_number:], data_max[inputs_number:]) - apply_min_max(
    steady_states["y_ss"],
    data_min[inputs_number:],
    data_max[inputs_number:],
)

poles = np.array(
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
    ]
)
L = compute_observer_gain(A_aug, C_aug, poles)

set_points_number = int(C_aug.shape[0])
STATE_DIM = int(A_aug.shape[0]) + set_points_number + inputs_number
ACTION_DIM = int(B_aug.shape[1])
ACTOR_LAYER_SIZES = [512, 512, 512, 512, 512]
CRITIC_LAYER_SIZES = [512, 512, 512, 512, 512]
BUFFER_CAPACITY = 40000
ACTOR_LR = 5e-5
CRITIC_LR = 5e-4
SMOOTHING_STD = 0.1
NOISE_CLIP = 0.01
GAMMA = 0.995
TAU = 0.005
MAX_ACTION = 1
POLICY_DELAY = 2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 256
STD_START = 0.0
STD_END = 0.01
STD_DECAY_RATE = 0.99992
STD_DECAY_MODE = "exp"

Qy_diag = np.array([8.0, 6.0])
Su_diag = np.array([1.0, 1.0])
Rdu_diag = np.array([1.0, 1.0])
k_rel = np.array([0.0015, 0.00015])
band_floor_phys = np.array([0.003, 0.035])
gamma_fallback = 3.0
fallback_event_penalty = 10.0
reward_config, reward_fn = make_reward_fn_relative_QR(
    data_min=data_min,
    data_max=data_max,
    n_inputs=inputs_number,
    k_rel=k_rel,
    band_floor_phys=band_floor_phys,
    Q_diag=Qy_diag,
    R_diag=Rdu_diag,
    tau_frac=0.5,
    gamma_out=1.0,
    gamma_in=3.0,
    beta=1.0,
    gate="geom",
    lam_in=3.0,
    bonus_kind="quadratic",
    gamma_fallback=gamma_fallback,
    fallback_event_penalty=fallback_event_penalty,
    R_fallback_diag=Rdu_diag,
    maintenance_band_scale=0.5,
    maintenance_move_weight=0.0,
    jitter_weight=0.0,
    dwell_bonus=0.0,
)

u_ss = apply_min_max(steady_states["ss_inputs"], data_min[:inputs_number], data_max[:inputs_number])
b_min = apply_min_max(u_min, data_min[:inputs_number], data_max[:inputs_number])
b_max = apply_min_max(u_max, data_min[:inputs_number], data_max[:inputs_number])
b1 = (b_min[0] - u_ss[0], b_max[0] - u_ss[0])
b2 = (b_min[1] - u_ss[1], b_max[1] - u_ss[1])
bnds = (b1, b2) * cont_h
IC_opt_template = np.zeros(inputs_number * cont_h)

u_min_scaled = apply_min_max(u_min, data_min[:inputs_number], data_max[:inputs_number])
u_max_scaled = apply_min_max(u_max, data_min[:inputs_number], data_max[:inputs_number])
u_dev_min = u_min_scaled - u_ss
u_dev_max = u_max_scaled - u_ss

LMPC_obj = design_direct_lyapunov_mpc_solver(
    A_aug=A_aug,
    B_aug=B_aug,
    C_aug=C_aug,
    Qy_diag=Qy_diag,
    NP=predict_h,
    NC=cont_h,
    Su_diag=Su_diag,
    u_min=u_dev_min,
    u_max=u_dev_max,
    Rdu_diag=Rdu_diag,
    terminal_set_on=True,
    terminal_alpha_scale=1.0,
)
MPC_obj_offset_free = MpcSolver(
    A_aug,
    B_aug,
    C_aug,
    Q_out=Qy_diag,
    R_in=Rdu_diag,
    NP=predict_h,
    NC=cont_h,
)

nominal_qs = 459.0
nominal_qi = 108.0
nominal_hA = 1.05e6
qi_change = 0.95
qs_change = 1.05
ha_change = 0.92

direct_target_config = {
    "u_ref_weight": float(u_prev_penalty_weight),
    "x_ref_weight": float(xs_prev_penalty_weight),
}


def make_td3_agent() -> TD3Agent:
    return TD3Agent(
        state_dim=STATE_DIM,
        action_dim=ACTION_DIM,
        actor_hidden=ACTOR_LAYER_SIZES,
        critic_hidden=CRITIC_LAYER_SIZES,
        gamma=GAMMA,
        actor_lr=ACTOR_LR,
        critic_lr=CRITIC_LR,
        batch_size=BATCH_SIZE,
        policy_delay=POLICY_DELAY,
        target_policy_smoothing_noise_std=SMOOTHING_STD,
        noise_clip=NOISE_CLIP,
        max_action=MAX_ACTION,
        tau=TAU,
        std_start=STD_START,
        std_end=STD_END,
        std_decay_rate=STD_DECAY_RATE,
        std_decay_mode=STD_DECAY_MODE,
        buffer_size=BUFFER_CAPACITY,
        device=DEVICE,
        actor_freeze=ACTOR_FREEZE,
    )


def latest_trained_agent_path(study: str) -> Path:
    root = Path(repo_path()) / "results" / study
    candidates = [
        path for path in root.glob("**/trained_agent_*.pkl")
        if "mpc_only" not in {part.lower() for part in path.parts}
    ]
    if not candidates:
        raise FileNotFoundError(f"No non-mpc_only trained_agent_*.pkl files found under {root}")
    return max(candidates, key=lambda path: (path.stat().st_mtime, str(path)))


def resolve_agent_paths() -> tuple[Path, Path]:
    if AGENT_SOURCE_MODE != "latest":
        if COLD_AGENT_PATH is None or PRETRAIN_AGENT_PATH is None:
            raise ValueError("Manual agent mode requires both COLD_AGENT_PATH and PRETRAIN_AGENT_PATH.")
        return Path(COLD_AGENT_PATH), Path(PRETRAIN_AGENT_PATH)

    cold_path = Path(COLD_AGENT_PATH) if COLD_AGENT_PATH else latest_trained_agent_path("ColdStart")
    pretrain_path = Path(PRETRAIN_AGENT_PATH) if PRETRAIN_AGENT_PATH else latest_trained_agent_path("Pretrain")
    return cold_path, pretrain_path


def load_agent(path: Path) -> TD3Agent:
    agent = make_td3_agent()
    agent.load(str(path))
    return agent


def build_eval_disturbance_profile() -> tuple[list[dict], dict[str, np.ndarray]]:
    episode_steps = time_in_sub_episodes
    scenarios = [
        {"name": "nominal", "qi": nominal_qi, "qs": nominal_qs, "ha": nominal_hA},
        {"name": "qi_step", "qi": nominal_qi * 0.95, "qs": nominal_qs, "ha": nominal_hA},
        {"name": "qs_step", "qi": nominal_qi, "qs": nominal_qs * 1.05, "ha": nominal_hA},
        {"name": "ha_step", "qi": nominal_qi, "qs": nominal_qs, "ha": nominal_hA * 0.92},
        {"name": "all_step", "qi": nominal_qi * 0.95, "qs": nominal_qs * 1.05, "ha": nominal_hA * 0.92},
    ]
    if len(scenarios) != EVAL_N_EPISODES:
        raise ValueError(f"EVAL_N_EPISODES={EVAL_N_EPISODES} must match scenario count {len(scenarios)}")

    profile = {
        "qi": np.concatenate([np.full(episode_steps, item["qi"], dtype=float) for item in scenarios]),
        "qs": np.concatenate([np.full(episode_steps, item["qs"], dtype=float) for item in scenarios]),
        "ha": np.concatenate([np.full(episode_steps, item["ha"], dtype=float) for item in scenarios]),
    }
    return scenarios, profile


def make_system() -> PolymerCSTR:
    return PolymerCSTR(
        system_params,
        system_design_params,
        system_steady_state_inputs,
        delta_t,
        deviation_form=False,
    )


def jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def y_sp_phys_from_bundle(bundle: dict) -> np.ndarray:
    y_sp = np.asarray(bundle.get("y_sp_steps", bundle.get("y_sp")), dtype=float)
    y_ss_scaled = apply_min_max(steady_states["y_ss"], data_min[inputs_number:], data_max[inputs_number:])
    y_sp_scaled = y_sp + y_ss_scaled
    return reverse_min_max(y_sp_scaled, data_min[inputs_number:], data_max[inputs_number:])


def aligned_step_outputs(bundle: dict) -> tuple[np.ndarray, np.ndarray]:
    """Return post-step outputs aligned with the per-control-step setpoint array."""
    y_system = np.asarray(bundle["y_system"], dtype=float)
    y_sp_phys = y_sp_phys_from_bundle(bundle)
    n_sp = int(y_sp_phys.shape[0])
    if y_system.shape[0] == n_sp + 1:
        y_aligned = y_system[1:]
    else:
        y_aligned = y_system[:n_sp]
    n = min(y_aligned.shape[0], y_sp_phys.shape[0])
    return y_aligned[:n], y_sp_phys[:n]


def output_error_metrics(bundle: dict) -> dict:
    y_system, y_sp_phys = aligned_step_outputs(bundle)
    n = min(y_system.shape[0], y_sp_phys.shape[0])
    err = y_system[:n] - y_sp_phys[:n]
    rmse = np.sqrt(np.nanmean(err ** 2, axis=0))

    episode_tail = min(50, time_in_sub_episodes)
    tail_chunks = []
    for start in range(0, n, time_in_sub_episodes):
        stop = min(start + time_in_sub_episodes, n)
        if stop > start:
            tail_chunks.append(err[max(start, stop - episode_tail):stop])
    tail_err = np.vstack(tail_chunks) if tail_chunks else err[-episode_tail:]
    final_err = err[min(n - 1, time_in_sub_episodes - 1)::time_in_sub_episodes]
    if final_err.size == 0:
        final_err = err[-1:]

    return {
        "eta_rmse": float(rmse[0]),
        "T_rmse": float(rmse[1]),
        "output_rmse_mean": float(np.nanmean(rmse)),
        "tail_eta_abs_mean": float(np.nanmean(np.abs(tail_err[:, 0]))),
        "tail_T_abs_mean": float(np.nanmean(np.abs(tail_err[:, 1]))),
        "tail_abs_mean": float(np.nanmean(np.abs(tail_err))),
        "final_eta_abs_mean": float(np.nanmean(np.abs(final_err[:, 0]))),
        "final_T_abs_mean": float(np.nanmean(np.abs(final_err[:, 1]))),
        "final_abs_mean": float(np.nanmean(np.abs(final_err))),
    }


def counts_by_episode(flags: np.ndarray) -> list[float]:
    values = np.asarray(flags, dtype=float).reshape(-1)
    counts = []
    for start in range(0, values.size, time_in_sub_episodes):
        stop = min(start + time_in_sub_episodes, values.size)
        counts.append(float(np.nansum(values[start:stop])))
    return counts[:EVAL_N_EPISODES]


def actual_fallback_flags(bundle: dict) -> np.ndarray:
    if "fallback_verified_flags" in bundle:
        return np.asarray(bundle["fallback_verified_flags"], dtype=float).reshape(-1)
    return np.zeros(int(bundle.get("nFE", 0)), dtype=float)


def intervention_flags(bundle: dict) -> np.ndarray:
    n = int(bundle.get("nFE", 0))
    flags = np.zeros(n, dtype=float)
    for key in ("projection_active_flags", "fallback_verified_flags", "constrained_mpc_applied_flags"):
        if key in bundle:
            values = np.asarray(bundle[key], dtype=float).reshape(-1)
            use = min(n, values.size)
            flags[:use] = np.maximum(flags[:use], np.nan_to_num(values[:use], nan=0.0))
    return flags


def would_be_activation_flags(bundle: dict) -> np.ndarray:
    n = int(bundle.get("nFE", 0))
    flags = np.zeros(n, dtype=float)
    for key in ("diagnostic_unsafe_flags", "diagnostic_unstable_flags"):
        if key in bundle:
            values = np.asarray(bundle[key], dtype=float).reshape(-1)
            use = min(n, values.size)
            flags[:use] = np.maximum(flags[:use], np.nan_to_num(values[:use], nan=0.0))
    return flags


def make_unified_record(case_name: str, controller_type: str, bundle: dict, record: dict, debug_dir: Path | str) -> dict:
    metrics = output_error_metrics(bundle)
    fallback_counts = counts_by_episode(actual_fallback_flags(bundle))
    intervention_counts = counts_by_episode(intervention_flags(bundle))
    would_be_counts = counts_by_episode(would_be_activation_flags(bundle))
    n_steps = int(record.get("n_steps") or bundle.get("nFE") or 0)

    unified = {
        "case_name": case_name,
        "controller_type": controller_type,
        "scenario_suite": EVAL_SCENARIO_SUITE,
        "n_eval_episodes": EVAL_N_EPISODES,
        "n_steps": n_steps,
        "reward_mean": record.get("reward_mean"),
        "reward_sum": record.get("reward_sum"),
        "fallback_rate": record.get("fallback_rate"),
        "actual_intervention_rate": record.get("actual_intervention_rate"),
        "mpc_only_would_be_activation_rate": (
            record.get("diagnostic_safety_active_rate", record.get("diagnostic_unsafe_rate"))
            if case_name == "mpc_only"
            else record.get("diagnostic_unsafe_rate")
        ),
        "fallback_count_total": float(np.nansum(fallback_counts)),
        "intervention_count_total": float(np.nansum(intervention_counts)),
        "would_be_activation_count_total": float(np.nansum(would_be_counts)),
        "wall_clock_seconds": record.get("wall_clock_seconds"),
        "wall_clock_seconds_per_episode": record.get("wall_clock_seconds_per_episode"),
        "wall_clock_seconds_per_step": record.get("wall_clock_seconds_per_step"),
        "wall_clock_steps_per_second": record.get("wall_clock_steps_per_second"),
        "debug_dir": str(debug_dir),
    }
    unified.update(metrics)
    return unified


def plot_bar(records: list[dict], key: str, title: str, ylabel: str, output_path: Path) -> None:
    labels = [row["case_name"] for row in records]
    values = [row.get(key, np.nan) for row in records]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(labels, values, color=["#2b6cb0", "#2f855a", "#975a16", "#744210"])
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=20)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_tracking(bundles: dict[str, dict], output_path: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    output_names = ["eta", "T"]
    first_bundle = next(iter(bundles.values()))
    _, y_sp_phys = aligned_step_outputs(first_bundle)
    t = np.arange(y_sp_phys.shape[0])
    for idx, ax in enumerate(axes):
        ax.plot(t, y_sp_phys[:, idx], color="black", linewidth=2.0, linestyle="--", label="setpoint")
        for case_name, bundle in bundles.items():
            y, y_sp_case = aligned_step_outputs(bundle)
            n = min(y.shape[0], y_sp_case.shape[0])
            ax.plot(np.arange(n), y[:n, idx], linewidth=1.2, label=case_name)
        for boundary in range(time_in_sub_episodes, EVAL_N_EPISODES * time_in_sub_episodes, time_in_sub_episodes):
            ax.axvline(boundary, color="0.75", linewidth=0.8)
        ax.set_ylabel(output_names[idx])
        ax.grid(alpha=0.25)
    axes[-1].set_xlabel("control step")
    axes[0].legend(ncol=2, fontsize=8)
    fig.suptitle("Saved-agent evaluation output tracking")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_inputs(bundles: dict[str, dict], output_path: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    input_names = ["Qc", "Qm"]
    for idx, ax in enumerate(axes):
        for case_name, bundle in bundles.items():
            u = np.asarray(bundle["u_applied_phys"], dtype=float)
            ax.plot(np.arange(u.shape[0]), u[:, idx], linewidth=1.2, label=case_name)
        for boundary in range(time_in_sub_episodes, EVAL_N_EPISODES * time_in_sub_episodes, time_in_sub_episodes):
            ax.axvline(boundary, color="0.75", linewidth=0.8)
        ax.set_ylabel(input_names[idx])
        ax.grid(alpha=0.25)
    axes[-1].set_xlabel("control step")
    axes[0].legend(ncol=2, fontsize=8)
    fig.suptitle("Saved-agent evaluation input trajectories")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_episode_counts(bundles: dict[str, dict], output_path: Path) -> None:
    labels = [f"E{idx + 1}" for idx in range(EVAL_N_EPISODES)]
    x = np.arange(EVAL_N_EPISODES)
    width = 0.8 / max(len(bundles), 1)
    fig, ax = plt.subplots(figsize=(11, 5))
    for idx, (case_name, bundle) in enumerate(bundles.items()):
        flags = would_be_activation_flags(bundle) if case_name == "mpc_only" else actual_fallback_flags(bundle)
        counts = counts_by_episode(flags)
        offset = (idx - (len(bundles) - 1) / 2) * width
        ax.bar(x + offset, counts, width=width, label=case_name)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("count per episode")
    ax.set_title("Fallback/intervention counts; MPC-only uses would-be gate activation")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def make_comparison_plots(records: list[dict], bundles: dict[str, dict], figures_dir: Path) -> list[str]:
    if not HAS_MATPLOTLIB:
        print("matplotlib is not available; skipping comparison plots.")
        return []
    figures_dir.mkdir(parents=True, exist_ok=True)
    plot_paths = []
    specs = [
        ("output_rmse_mean", "Output RMSE summary", "mean output RMSE", "comparison_output_rmse.png"),
        ("wall_clock_seconds", "Runtime summary", "wall-clock seconds", "comparison_runtime.png"),
        ("tail_abs_mean", "Tail offset summary", "mean absolute tail offset", "comparison_tail_offset.png"),
        ("would_be_activation_count_total", "Would-be/diagnostic activation summary", "count", "comparison_would_be_activation.png"),
    ]
    for key, title, ylabel, filename in specs:
        path = figures_dir / filename
        plot_bar(records, key, title, ylabel, path)
        plot_paths.append(str(path))
    path = figures_dir / "comparison_output_tracking.png"
    plot_tracking(bundles, path)
    plot_paths.append(str(path))
    path = figures_dir / "comparison_input_trajectories.png"
    plot_inputs(bundles, path)
    plot_paths.append(str(path))
    path = figures_dir / "comparison_fallback_intervention_counts.png"
    plot_episode_counts(bundles, path)
    plot_paths.append(str(path))
    return plot_paths


def run_rl_saved_agent_case(case_name: str, agent_path: Path, scenarios: list[dict], profile: dict[str, np.ndarray]):
    case_config = {
        "study_name": study_name,
        "case_name": case_name,
        "controller_mode": "saved_rl_direct_safety_gate",
        "agent_path": str(agent_path),
        "scenario_suite": EVAL_SCENARIO_SUITE,
        "scenarios": scenarios,
        "n_tests": n_tests,
        "set_points_len": set_points_len,
        "test_cycle": TEST_CYCLE,
        "force_final_test": FORCE_FINAL_TEST,
        "training_phase_config": None,
        "reward_config": reward_config,
        "gamma": GAMMA,
        "rho_lyap": rho_lyap,
        "lyap_eps": lyap_eps,
        "fallback_event_penalty": fallback_event_penalty,
        "direct_target_config": dict(direct_target_config),
    }
    agent = load_agent(agent_path)
    cstr_case = make_system()
    timer_start = time.perf_counter()
    results_case = run_rl_train(
        system=cstr_case,
        y_sp_scenario=y_sp_scenario,
        n_tests=n_tests,
        set_points_len=set_points_len,
        steady_states=steady_states,
        min_max_dict=min_max_dict,
        agent=agent,
        MPC_obj=LMPC_obj,
        L=L,
        data_min=data_min,
        data_max=data_max,
        warm_start=warm_start,
        test_cycle=TEST_CYCLE,
        nominal_qi=nominal_qi,
        nominal_qs=nominal_qs,
        nominal_ha=nominal_hA,
        qi_change=qi_change,
        qs_change=qs_change,
        ha_change=ha_change,
        reward_fn=reward_fn,
        mode=plant_mode,
        rho_lyap=rho_lyap,
        lyap_eps=lyap_eps,
        lyap_tol=lyap_tol,
        seed=0,
        use_lyap=True,
        IC_opt=IC_opt_template.copy(),
        bnds=bnds,
        cons=(),
        reuse_mpc_solution_as_ic=False,
        reset_system_on_entry=True,
        projection_backend="direct_accept_or_fallback",
        first_step_contraction_on=True,
        direct_target_mode="bounded",
        direct_target_config=direct_target_config,
        direct_tracking_use_target_output=False,
        diagnostic_lmpc_obj=LMPC_obj,
        disturbance_after_step=disturbance_after_step,
        training_phase_config=None,
        force_final_test=FORCE_FINAL_TEST,
        disturbance_profile=profile,
    )
    wall_clock_seconds = float(time.perf_counter() - timer_start)
    n_steps = int(results_case[5])
    episode_len = int(results_case[6])
    n_episodes = int(np.ceil(n_steps / float(episode_len))) if episode_len > 0 else 0
    timing = {
        "wall_clock_seconds": wall_clock_seconds,
        "wall_clock_seconds_per_episode": None if n_episodes <= 0 else wall_clock_seconds / float(n_episodes),
        "wall_clock_seconds_per_step": None if n_steps <= 0 else wall_clock_seconds / float(n_steps),
        "wall_clock_steps_per_second": None if wall_clock_seconds <= 0.0 else n_steps / wall_clock_seconds,
        "wall_clock_n_steps": n_steps,
        "wall_clock_n_episodes": n_episodes,
    }
    case_config.update(timing)
    bundle = build_safety_filter_run_bundle(
        source=case_name,
        results=results_case,
        steady_states=steady_states,
        config=case_config,
        min_max_dict=min_max_dict,
        data_min=data_min,
        data_max=data_max,
        extra={"reward_config": reward_config, "timing": timing, "scenarios": scenarios},
    )
    debug_dir = save_safety_filter_debug_artifacts(
        bundle,
        directory=str(study_root),
        prefix_name=case_name,
        save_plots=SAVE_CASE_PLOTS,
    )
    record = make_safety_filter_comparison_record(case_name, bundle, debug_dir)
    record.update(timing)
    return bundle, Path(debug_dir), record


def run_mpc_only_case(scenarios: list[dict], profile: dict[str, np.ndarray]):
    case_name = "mpc_only"
    case_config = {
        "study_name": study_name,
        "case_name": case_name,
        "controller_mode": "offset_free_mpc_with_direct_diagnostics",
        "scenario_suite": EVAL_SCENARIO_SUITE,
        "scenarios": scenarios,
        "n_tests": n_tests,
        "set_points_len": set_points_len,
        "force_final_test": FORCE_FINAL_TEST,
        "direct_target_config": dict(direct_target_config),
    }
    cstr_case = make_system()
    timer_start = time.perf_counter()
    results_case = run_offset_free_mpc_with_direct_diagnostics(
        system=cstr_case,
        MPC_obj=MPC_obj_offset_free,
        diagnostic_LMPC_obj=LMPC_obj,
        y_sp_scenario=y_sp_scenario,
        n_tests=n_tests,
        set_points_len=set_points_len,
        steady_states=steady_states,
        IC_opt=IC_opt_template.copy(),
        bnds=bnds,
        L=L,
        data_min=data_min,
        data_max=data_max,
        test_cycle=TEST_CYCLE,
        reward_fn=reward_fn,
        nominal_qi=nominal_qi,
        nominal_qs=nominal_qs,
        nominal_ha=nominal_hA,
        qi_change=qi_change,
        qs_change=qs_change,
        ha_change=ha_change,
        target_mode="bounded",
        target_config=direct_target_config,
        target_H=None,
        mode=plant_mode,
        disturbance_after_step=disturbance_after_step,
        use_target_output_for_tracking=use_target_output_for_tracking,
        rho_lyap=rho_lyap,
        lyap_eps=lyap_eps,
        first_step_contraction_on=True,
        reset_system_on_entry=True,
        solver_options={"warm_start": True},
        force_final_test=FORCE_FINAL_TEST,
        disturbance_profile=profile,
    )
    wall_clock_seconds = float(time.perf_counter() - timer_start)
    n_steps = int(results_case["nFE"])
    episode_len = int(results_case["time_in_sub_episodes"])
    n_episodes = int(np.ceil(n_steps / float(episode_len))) if episode_len > 0 else 0
    timing = {
        "wall_clock_seconds": wall_clock_seconds,
        "wall_clock_seconds_per_episode": None if n_episodes <= 0 else wall_clock_seconds / float(n_episodes),
        "wall_clock_seconds_per_step": None if n_steps <= 0 else wall_clock_seconds / float(n_steps),
        "wall_clock_steps_per_second": None if wall_clock_seconds <= 0.0 else n_steps / wall_clock_seconds,
        "wall_clock_n_steps": n_steps,
        "wall_clock_n_episodes": n_episodes,
    }
    case_config.update(timing)
    bundle = build_direct_lyapunov_run_bundle(
        source=case_name,
        results=results_case,
        steady_states=steady_states,
        config=case_config,
        data_min=data_min,
        data_max=data_max,
        extra={"reward_config": reward_config, "min_max_dict": min_max_dict, "timing": timing, "scenarios": scenarios},
    )
    debug_dir = save_direct_lyapunov_debug_artifacts(
        bundle,
        directory=str(study_root),
        prefix_name=case_name,
        save_plots=SAVE_CASE_PLOTS,
    )
    record = make_direct_lyapunov_comparison_record(case_name, bundle, debug_dir)
    record.update(timing)
    return bundle, Path(debug_dir), record


def run_direct_lmpc_case(scenarios: list[dict], profile: dict[str, np.ndarray]):
    case_name = "direct_lmpc"
    case_config = {
        "study_name": study_name,
        "case_name": case_name,
        "controller_mode": "direct_lyapunov_mpc",
        "scenario_suite": EVAL_SCENARIO_SUITE,
        "scenarios": scenarios,
        "n_tests": n_tests,
        "set_points_len": set_points_len,
        "force_final_test": FORCE_FINAL_TEST,
        "direct_target_config": dict(direct_target_config),
    }
    cstr_case = make_system()
    timer_start = time.perf_counter()
    results_case = run_direct_output_disturbance_lyapunov_mpc(
        system=cstr_case,
        LMPC_obj=LMPC_obj,
        y_sp_scenario=y_sp_scenario,
        n_tests=n_tests,
        set_points_len=set_points_len,
        steady_states=steady_states,
        IC_opt=IC_opt_template.copy(),
        bnds=bnds,
        L=L,
        data_min=data_min,
        data_max=data_max,
        test_cycle=TEST_CYCLE,
        reward_fn=reward_fn,
        nominal_qi=nominal_qi,
        nominal_qs=nominal_qs,
        nominal_ha=nominal_hA,
        qi_change=qi_change,
        qs_change=qs_change,
        ha_change=ha_change,
        target_mode="bounded",
        lyapunov_mode="hard",
        target_config=direct_target_config,
        target_H=None,
        mode=plant_mode,
        disturbance_after_step=disturbance_after_step,
        use_target_output_for_tracking=use_target_output_for_tracking,
        skip_terminal_if_alpha_small=True,
        alpha_terminal_min=1e-8,
        use_target_on_solver_fail=False,
        rho_lyap=rho_lyap,
        lyap_eps=lyap_eps,
        slack_penalty=slack_penalty,
        first_step_contraction_on=True,
        reset_system_on_entry=True,
        solver_options={"warm_start": True},
        force_final_test=FORCE_FINAL_TEST,
        disturbance_profile=profile,
    )
    wall_clock_seconds = float(time.perf_counter() - timer_start)
    n_steps = int(results_case["nFE"])
    episode_len = int(results_case["time_in_sub_episodes"])
    n_episodes = int(np.ceil(n_steps / float(episode_len))) if episode_len > 0 else 0
    timing = {
        "wall_clock_seconds": wall_clock_seconds,
        "wall_clock_seconds_per_episode": None if n_episodes <= 0 else wall_clock_seconds / float(n_episodes),
        "wall_clock_seconds_per_step": None if n_steps <= 0 else wall_clock_seconds / float(n_steps),
        "wall_clock_steps_per_second": None if wall_clock_seconds <= 0.0 else n_steps / wall_clock_seconds,
        "wall_clock_n_steps": n_steps,
        "wall_clock_n_episodes": n_episodes,
    }
    case_config.update(timing)
    bundle = build_direct_lyapunov_run_bundle(
        source=case_name,
        results=results_case,
        steady_states=steady_states,
        config=case_config,
        data_min=data_min,
        data_max=data_max,
        extra={"reward_config": reward_config, "min_max_dict": min_max_dict, "timing": timing, "scenarios": scenarios},
    )
    debug_dir = save_direct_lyapunov_debug_artifacts(
        bundle,
        directory=str(study_root),
        prefix_name=case_name,
        save_plots=SAVE_CASE_PLOTS,
    )
    record = make_direct_lyapunov_comparison_record(case_name, bundle, debug_dir)
    record.update(timing)
    return bundle, Path(debug_dir), record


def main() -> None:
    cold_agent_path, pretrain_agent_path = resolve_agent_paths()
    scenarios, disturbance_profile = build_eval_disturbance_profile()
    planned = {
        "study_root": study_root,
        "cold_agent_path": cold_agent_path,
        "pretrain_agent_path": pretrain_agent_path,
        "scenario_suite": EVAL_SCENARIO_SUITE,
        "scenarios": scenarios,
        "n_eval_episodes": EVAL_N_EPISODES,
        "set_points_len": EVAL_SET_POINTS_LEN,
        "time_in_sub_episodes": time_in_sub_episodes,
        "n_steps": int(len(disturbance_profile["qi"])),
        "controllers": ["cold_saved_rl", "pretrained_saved_rl", "mpc_only", "direct_lmpc"],
    }
    if DRY_RUN:
        print("Saved-agent evaluation dry run:")
        pprint(jsonable(planned))
        return

    study_root.mkdir(parents=True, exist_ok=True)
    print(f"Saving saved-agent evaluation artifacts under: {study_root}")
    print("Using agents:")
    print(f"  cold: {cold_agent_path}")
    print(f"  pretrained: {pretrain_agent_path}")

    bundles = {}
    records = []
    summary_records = []

    for case_name, path in (
        ("cold_saved_rl", cold_agent_path),
        ("pretrained_saved_rl", pretrain_agent_path),
    ):
        print(f"Running {case_name}")
        bundle, debug_dir, record = run_rl_saved_agent_case(case_name, path, scenarios, disturbance_profile)
        bundles[case_name] = bundle
        records.append(record)
        summary_records.append(make_unified_record(case_name, "saved_rl_safety_gate", bundle, record, debug_dir))
        pprint(summary_records[-1])

    print("Running mpc_only")
    bundle, debug_dir, record = run_mpc_only_case(scenarios, disturbance_profile)
    bundles["mpc_only"] = bundle
    records.append(record)
    summary_records.append(make_unified_record("mpc_only", "offset_free_mpc_diagnostic", bundle, record, debug_dir))
    pprint(summary_records[-1])

    print("Running direct_lmpc")
    bundle, debug_dir, record = run_direct_lmpc_case(scenarios, disturbance_profile)
    bundles["direct_lmpc"] = bundle
    records.append(record)
    summary_records.append(make_unified_record("direct_lmpc", "direct_lyapunov_mpc", bundle, record, debug_dir))
    pprint(summary_records[-1])

    comparison_csv = study_root / "comparison_table.csv"
    write_csv(comparison_csv, summary_records)

    raw_records_csv = study_root / "raw_comparison_records.csv"
    write_csv(raw_records_csv, records)

    scenarios_csv = study_root / "scenario_table.csv"
    write_csv(scenarios_csv, scenarios)

    figures_dir = study_root / "figures"
    plot_paths = make_comparison_plots(summary_records, bundles, figures_dir)

    summary = {
        **planned,
        "completed_at": datetime.now().isoformat(timespec="seconds"),
        "comparison_table": comparison_csv,
        "raw_comparison_records": raw_records_csv,
        "scenario_table": scenarios_csv,
        "plot_paths": plot_paths,
    }
    with (study_root / "evaluation_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(jsonable(summary), handle, indent=2)

    print("Saved-agent evaluation complete.")
    print(f"Comparison table: {comparison_csv}")


if __name__ == "__main__":
    main()
