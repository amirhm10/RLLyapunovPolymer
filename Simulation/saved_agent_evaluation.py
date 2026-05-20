"""Reusable helpers for saved-agent Lyapunov safety-gate evaluation.

The root script owns the experiment setup, parameter values, solver
construction, and run orchestration. This module only provides callable helper
functions/classes that can be reused by root entrypoints or future scripts.
"""

from __future__ import annotations

import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except Exception:
    HAS_MATPLOTLIB = False

from TD3Agent.agent import TD3Agent
from Simulation.run_rl_lyapunov import run_rl_train
from Simulation.system_functions import PolymerCSTR
from Lyapunov.direct_lyapunov_mpc import (
    build_direct_lyapunov_run_bundle,
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
from utils.path_helpers import repo_path
from utils.scaling_helpers import apply_min_max, reverse_min_max


@dataclass
class TD3AgentConfig:
    state_dim: int
    action_dim: int
    actor_hidden: list[int]
    critic_hidden: list[int]
    gamma: float
    actor_lr: float
    critic_lr: float
    batch_size: int
    policy_delay: int
    target_policy_smoothing_noise_std: float
    noise_clip: float
    max_action: float
    tau: float
    std_start: float
    std_end: float
    std_decay_rate: float
    std_decay_mode: str
    buffer_size: int
    device: Any
    actor_freeze: int


@dataclass
class SavedAgentEvalContext:
    study_name: str
    study_root: Path
    scenario_suite: str
    n_tests: int
    set_points_len: int
    test_cycle: list[bool]
    warm_start: int
    time_in_sub_episodes: int
    force_final_test: bool
    save_case_plots: bool
    td3_agent_config: TD3AgentConfig
    system_params: np.ndarray
    system_design_params: np.ndarray
    system_steady_state_inputs: np.ndarray
    delta_t: float
    steady_states: dict
    min_max_dict: dict
    data_min: np.ndarray
    data_max: np.ndarray
    inputs_number: int
    y_sp_scenario: np.ndarray
    L: np.ndarray
    LMPC_obj: Any
    MPC_obj_offset_free: Any
    reward_fn: Any
    reward_config: dict
    gamma: float
    rho_lyap: float
    lyap_eps: float
    lyap_tol: float
    slack_penalty: float
    fallback_event_penalty: float
    plant_mode: str
    disturbance_after_step: bool
    use_target_output_for_tracking: bool
    IC_opt_template: np.ndarray
    bnds: tuple
    direct_target_config: dict
    nominal_qi: float
    nominal_qs: float
    nominal_hA: float
    qi_change: float
    qs_change: float
    ha_change: float


def build_td3_agent(config: TD3AgentConfig) -> TD3Agent:
    return TD3Agent(
        state_dim=config.state_dim,
        action_dim=config.action_dim,
        actor_hidden=config.actor_hidden,
        critic_hidden=config.critic_hidden,
        gamma=config.gamma,
        actor_lr=config.actor_lr,
        critic_lr=config.critic_lr,
        batch_size=config.batch_size,
        policy_delay=config.policy_delay,
        target_policy_smoothing_noise_std=config.target_policy_smoothing_noise_std,
        noise_clip=config.noise_clip,
        max_action=config.max_action,
        tau=config.tau,
        std_start=config.std_start,
        std_end=config.std_end,
        std_decay_rate=config.std_decay_rate,
        std_decay_mode=config.std_decay_mode,
        buffer_size=config.buffer_size,
        device=config.device,
        actor_freeze=config.actor_freeze,
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


def resolve_agent_paths(
    *,
    agent_source_mode: str,
    cold_agent_path: str | Path | None,
    pretrain_agent_path: str | Path | None,
) -> tuple[Path, Path]:
    if agent_source_mode != "latest":
        if cold_agent_path is None or pretrain_agent_path is None:
            raise ValueError("Manual agent mode requires both cold and pretrained agent paths.")
        return Path(cold_agent_path), Path(pretrain_agent_path)

    cold_path = Path(cold_agent_path) if cold_agent_path else latest_trained_agent_path("ColdStart")
    pretrain_path = Path(pretrain_agent_path) if pretrain_agent_path else latest_trained_agent_path("Pretrain")
    return cold_path, pretrain_path


def load_agent(path: Path, config: TD3AgentConfig) -> TD3Agent:
    agent = build_td3_agent(config)
    agent.load(str(path))
    return agent


def build_eval_disturbance_profile(
    *,
    n_eval_episodes: int,
    episode_steps: int,
    nominal_qi: float,
    nominal_qs: float,
    nominal_ha: float,
) -> tuple[list[dict], dict[str, np.ndarray]]:
    scenarios = [
        {"name": "nominal", "qi": nominal_qi, "qs": nominal_qs, "ha": nominal_ha},
        {"name": "qi_step", "qi": nominal_qi * 0.95, "qs": nominal_qs, "ha": nominal_ha},
        {"name": "qs_step", "qi": nominal_qi, "qs": nominal_qs * 1.05, "ha": nominal_ha},
        {"name": "ha_step", "qi": nominal_qi, "qs": nominal_qs, "ha": nominal_ha * 0.92},
        {"name": "all_step", "qi": nominal_qi * 0.95, "qs": nominal_qs * 1.05, "ha": nominal_ha * 0.92},
    ]
    if len(scenarios) != int(n_eval_episodes):
        raise ValueError(f"n_eval_episodes={n_eval_episodes} must match scenario count {len(scenarios)}")

    profile = {
        "qi": np.concatenate([np.full(episode_steps, item["qi"], dtype=float) for item in scenarios]),
        "qs": np.concatenate([np.full(episode_steps, item["qs"], dtype=float) for item in scenarios]),
        "ha": np.concatenate([np.full(episode_steps, item["ha"], dtype=float) for item in scenarios]),
    }
    return scenarios, profile


def make_system(ctx: SavedAgentEvalContext) -> PolymerCSTR:
    return PolymerCSTR(
        ctx.system_params,
        ctx.system_design_params,
        ctx.system_steady_state_inputs,
        ctx.delta_t,
        deviation_form=False,
    )


def jsonable(value: Any) -> Any:
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


def y_sp_phys_from_bundle(ctx: SavedAgentEvalContext, bundle: dict) -> np.ndarray:
    y_sp = np.asarray(bundle.get("y_sp_steps", bundle.get("y_sp")), dtype=float)
    y_ss_scaled = apply_min_max(
        ctx.steady_states["y_ss"],
        ctx.data_min[ctx.inputs_number:],
        ctx.data_max[ctx.inputs_number:],
    )
    y_sp_scaled = y_sp + y_ss_scaled
    return reverse_min_max(y_sp_scaled, ctx.data_min[ctx.inputs_number:], ctx.data_max[ctx.inputs_number:])


def aligned_step_outputs(ctx: SavedAgentEvalContext, bundle: dict) -> tuple[np.ndarray, np.ndarray]:
    y_system = np.asarray(bundle["y_system"], dtype=float)
    y_sp_phys = y_sp_phys_from_bundle(ctx, bundle)
    n_sp = int(y_sp_phys.shape[0])
    y_aligned = y_system[1:] if y_system.shape[0] == n_sp + 1 else y_system[:n_sp]
    n = min(y_aligned.shape[0], y_sp_phys.shape[0])
    return y_aligned[:n], y_sp_phys[:n]


def output_error_metrics(ctx: SavedAgentEvalContext, bundle: dict) -> dict:
    y_system, y_sp_phys = aligned_step_outputs(ctx, bundle)
    n = min(y_system.shape[0], y_sp_phys.shape[0])
    err = y_system[:n] - y_sp_phys[:n]
    rmse = np.sqrt(np.nanmean(err ** 2, axis=0))

    episode_tail = min(50, ctx.time_in_sub_episodes)
    tail_chunks = []
    for start in range(0, n, ctx.time_in_sub_episodes):
        stop = min(start + ctx.time_in_sub_episodes, n)
        if stop > start:
            tail_chunks.append(err[max(start, stop - episode_tail):stop])
    tail_err = np.vstack(tail_chunks) if tail_chunks else err[-episode_tail:]
    final_err = err[min(n - 1, ctx.time_in_sub_episodes - 1)::ctx.time_in_sub_episodes]
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


def counts_by_episode(ctx: SavedAgentEvalContext, flags: np.ndarray) -> list[float]:
    values = np.asarray(flags, dtype=float).reshape(-1)
    counts = []
    for start in range(0, values.size, ctx.time_in_sub_episodes):
        stop = min(start + ctx.time_in_sub_episodes, values.size)
        counts.append(float(np.nansum(values[start:stop])))
    return counts[:ctx.n_tests]


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


def make_unified_record(
    ctx: SavedAgentEvalContext,
    case_name: str,
    controller_type: str,
    bundle: dict,
    record: dict,
    debug_dir: Path | str,
) -> dict:
    metrics = output_error_metrics(ctx, bundle)
    fallback_counts = counts_by_episode(ctx, actual_fallback_flags(bundle))
    intervention_counts = counts_by_episode(ctx, intervention_flags(bundle))
    would_be_counts = counts_by_episode(ctx, would_be_activation_flags(bundle))
    n_steps = int(record.get("n_steps") or bundle.get("nFE") or 0)

    unified = {
        "case_name": case_name,
        "controller_type": controller_type,
        "scenario_suite": ctx.scenario_suite,
        "n_eval_episodes": ctx.n_tests,
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


def plot_tracking(ctx: SavedAgentEvalContext, bundles: dict[str, dict], output_path: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    output_names = ["eta", "T"]
    first_bundle = next(iter(bundles.values()))
    _, y_sp_phys = aligned_step_outputs(ctx, first_bundle)
    t = np.arange(y_sp_phys.shape[0])
    for idx, ax in enumerate(axes):
        ax.plot(t, y_sp_phys[:, idx], color="black", linewidth=2.0, linestyle="--", label="setpoint")
        for case_name, bundle in bundles.items():
            y, y_sp_case = aligned_step_outputs(ctx, bundle)
            n = min(y.shape[0], y_sp_case.shape[0])
            ax.plot(np.arange(n), y[:n, idx], linewidth=1.2, label=case_name)
        for boundary in range(ctx.time_in_sub_episodes, ctx.n_tests * ctx.time_in_sub_episodes, ctx.time_in_sub_episodes):
            ax.axvline(boundary, color="0.75", linewidth=0.8)
        ax.set_ylabel(output_names[idx])
        ax.grid(alpha=0.25)
    axes[-1].set_xlabel("control step")
    axes[0].legend(ncol=2, fontsize=8)
    fig.suptitle("Saved-agent evaluation output tracking")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_inputs(ctx: SavedAgentEvalContext, bundles: dict[str, dict], output_path: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    input_names = ["Qc", "Qm"]
    for idx, ax in enumerate(axes):
        for case_name, bundle in bundles.items():
            u = np.asarray(bundle["u_applied_phys"], dtype=float)
            ax.plot(np.arange(u.shape[0]), u[:, idx], linewidth=1.2, label=case_name)
        for boundary in range(ctx.time_in_sub_episodes, ctx.n_tests * ctx.time_in_sub_episodes, ctx.time_in_sub_episodes):
            ax.axvline(boundary, color="0.75", linewidth=0.8)
        ax.set_ylabel(input_names[idx])
        ax.grid(alpha=0.25)
    axes[-1].set_xlabel("control step")
    axes[0].legend(ncol=2, fontsize=8)
    fig.suptitle("Saved-agent evaluation input trajectories")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_episode_counts(ctx: SavedAgentEvalContext, bundles: dict[str, dict], output_path: Path) -> None:
    labels = [f"E{idx + 1}" for idx in range(ctx.n_tests)]
    x = np.arange(ctx.n_tests)
    width = 0.8 / max(len(bundles), 1)
    fig, ax = plt.subplots(figsize=(11, 5))
    for idx, (case_name, bundle) in enumerate(bundles.items()):
        flags = would_be_activation_flags(bundle) if case_name == "mpc_only" else actual_fallback_flags(bundle)
        counts = counts_by_episode(ctx, flags)
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


def make_comparison_plots(
    ctx: SavedAgentEvalContext,
    records: list[dict],
    bundles: dict[str, dict],
    figures_dir: Path,
) -> list[str]:
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
    plot_tracking(ctx, bundles, path)
    plot_paths.append(str(path))
    path = figures_dir / "comparison_input_trajectories.png"
    plot_inputs(ctx, bundles, path)
    plot_paths.append(str(path))
    path = figures_dir / "comparison_fallback_intervention_counts.png"
    plot_episode_counts(ctx, bundles, path)
    plot_paths.append(str(path))
    return plot_paths


def _timing_metadata(start_time: float, n_steps: int, episode_len: int) -> dict:
    wall_clock_seconds = float(time.perf_counter() - start_time)
    n_episodes = int(np.ceil(n_steps / float(episode_len))) if episode_len > 0 else 0
    return {
        "wall_clock_seconds": wall_clock_seconds,
        "wall_clock_seconds_per_episode": None if n_episodes <= 0 else wall_clock_seconds / float(n_episodes),
        "wall_clock_seconds_per_step": None if n_steps <= 0 else wall_clock_seconds / float(n_steps),
        "wall_clock_steps_per_second": None if wall_clock_seconds <= 0.0 else n_steps / wall_clock_seconds,
        "wall_clock_n_steps": int(n_steps),
        "wall_clock_n_episodes": int(n_episodes),
    }


def run_rl_saved_agent_case(
    ctx: SavedAgentEvalContext,
    *,
    case_name: str,
    agent_path: Path,
    scenarios: list[dict],
    profile: dict[str, np.ndarray],
):
    case_config = {
        "study_name": ctx.study_name,
        "case_name": case_name,
        "controller_mode": "saved_rl_direct_safety_gate",
        "agent_path": str(agent_path),
        "scenario_suite": ctx.scenario_suite,
        "scenarios": scenarios,
        "n_tests": ctx.n_tests,
        "set_points_len": ctx.set_points_len,
        "test_cycle": ctx.test_cycle,
        "force_final_test": ctx.force_final_test,
        "training_phase_config": None,
        "reward_config": ctx.reward_config,
        "gamma": ctx.gamma,
        "rho_lyap": ctx.rho_lyap,
        "lyap_eps": ctx.lyap_eps,
        "fallback_event_penalty": ctx.fallback_event_penalty,
        "direct_target_config": dict(ctx.direct_target_config),
    }
    agent = load_agent(agent_path, ctx.td3_agent_config)
    cstr_case = make_system(ctx)
    timer_start = time.perf_counter()
    results_case = run_rl_train(
        system=cstr_case,
        y_sp_scenario=ctx.y_sp_scenario,
        n_tests=ctx.n_tests,
        set_points_len=ctx.set_points_len,
        steady_states=ctx.steady_states,
        min_max_dict=ctx.min_max_dict,
        agent=agent,
        MPC_obj=ctx.LMPC_obj,
        L=ctx.L,
        data_min=ctx.data_min,
        data_max=ctx.data_max,
        warm_start=ctx.warm_start,
        test_cycle=ctx.test_cycle,
        nominal_qi=ctx.nominal_qi,
        nominal_qs=ctx.nominal_qs,
        nominal_ha=ctx.nominal_hA,
        qi_change=ctx.qi_change,
        qs_change=ctx.qs_change,
        ha_change=ctx.ha_change,
        reward_fn=ctx.reward_fn,
        mode=ctx.plant_mode,
        rho_lyap=ctx.rho_lyap,
        lyap_eps=ctx.lyap_eps,
        lyap_tol=ctx.lyap_tol,
        seed=0,
        use_lyap=True,
        IC_opt=ctx.IC_opt_template.copy(),
        bnds=ctx.bnds,
        cons=(),
        reuse_mpc_solution_as_ic=False,
        reset_system_on_entry=True,
        projection_backend="direct_accept_or_fallback",
        first_step_contraction_on=True,
        direct_target_mode="bounded",
        direct_target_config=ctx.direct_target_config,
        direct_tracking_use_target_output=False,
        diagnostic_lmpc_obj=ctx.LMPC_obj,
        disturbance_after_step=ctx.disturbance_after_step,
        training_phase_config=None,
        force_final_test=ctx.force_final_test,
        disturbance_profile=profile,
    )
    timing = _timing_metadata(timer_start, int(results_case[5]), int(results_case[6]))
    case_config.update(timing)
    bundle = build_safety_filter_run_bundle(
        source=case_name,
        results=results_case,
        steady_states=ctx.steady_states,
        config=case_config,
        min_max_dict=ctx.min_max_dict,
        data_min=ctx.data_min,
        data_max=ctx.data_max,
        extra={"reward_config": ctx.reward_config, "timing": timing, "scenarios": scenarios},
    )
    debug_dir = save_safety_filter_debug_artifacts(
        bundle,
        directory=str(ctx.study_root),
        prefix_name=case_name,
        save_plots=ctx.save_case_plots,
    )
    record = make_safety_filter_comparison_record(case_name, bundle, debug_dir)
    record.update(timing)
    return bundle, Path(debug_dir), record


def run_mpc_only_case(ctx: SavedAgentEvalContext, *, scenarios: list[dict], profile: dict[str, np.ndarray]):
    case_name = "mpc_only"
    case_config = {
        "study_name": ctx.study_name,
        "case_name": case_name,
        "controller_mode": "offset_free_mpc_with_direct_diagnostics",
        "scenario_suite": ctx.scenario_suite,
        "scenarios": scenarios,
        "n_tests": ctx.n_tests,
        "set_points_len": ctx.set_points_len,
        "force_final_test": ctx.force_final_test,
        "direct_target_config": dict(ctx.direct_target_config),
    }
    cstr_case = make_system(ctx)
    timer_start = time.perf_counter()
    results_case = run_offset_free_mpc_with_direct_diagnostics(
        system=cstr_case,
        MPC_obj=ctx.MPC_obj_offset_free,
        diagnostic_LMPC_obj=ctx.LMPC_obj,
        y_sp_scenario=ctx.y_sp_scenario,
        n_tests=ctx.n_tests,
        set_points_len=ctx.set_points_len,
        steady_states=ctx.steady_states,
        IC_opt=ctx.IC_opt_template.copy(),
        bnds=ctx.bnds,
        L=ctx.L,
        data_min=ctx.data_min,
        data_max=ctx.data_max,
        test_cycle=ctx.test_cycle,
        reward_fn=ctx.reward_fn,
        nominal_qi=ctx.nominal_qi,
        nominal_qs=ctx.nominal_qs,
        nominal_ha=ctx.nominal_hA,
        qi_change=ctx.qi_change,
        qs_change=ctx.qs_change,
        ha_change=ctx.ha_change,
        target_mode="bounded",
        target_config=ctx.direct_target_config,
        target_H=None,
        mode=ctx.plant_mode,
        disturbance_after_step=ctx.disturbance_after_step,
        use_target_output_for_tracking=ctx.use_target_output_for_tracking,
        rho_lyap=ctx.rho_lyap,
        lyap_eps=ctx.lyap_eps,
        first_step_contraction_on=True,
        reset_system_on_entry=True,
        solver_options={"warm_start": True},
        force_final_test=ctx.force_final_test,
        disturbance_profile=profile,
    )
    timing = _timing_metadata(timer_start, int(results_case["nFE"]), int(results_case["time_in_sub_episodes"]))
    case_config.update(timing)
    bundle = build_direct_lyapunov_run_bundle(
        source=case_name,
        results=results_case,
        steady_states=ctx.steady_states,
        config=case_config,
        data_min=ctx.data_min,
        data_max=ctx.data_max,
        extra={"reward_config": ctx.reward_config, "min_max_dict": ctx.min_max_dict, "timing": timing, "scenarios": scenarios},
    )
    debug_dir = save_direct_lyapunov_debug_artifacts(
        bundle,
        directory=str(ctx.study_root),
        prefix_name=case_name,
        save_plots=ctx.save_case_plots,
    )
    record = make_direct_lyapunov_comparison_record(case_name, bundle, debug_dir)
    record.update(timing)
    return bundle, Path(debug_dir), record


def run_direct_lmpc_case(ctx: SavedAgentEvalContext, *, scenarios: list[dict], profile: dict[str, np.ndarray]):
    case_name = "direct_lmpc"
    case_config = {
        "study_name": ctx.study_name,
        "case_name": case_name,
        "controller_mode": "direct_lyapunov_mpc",
        "scenario_suite": ctx.scenario_suite,
        "scenarios": scenarios,
        "n_tests": ctx.n_tests,
        "set_points_len": ctx.set_points_len,
        "force_final_test": ctx.force_final_test,
        "direct_target_config": dict(ctx.direct_target_config),
    }
    cstr_case = make_system(ctx)
    timer_start = time.perf_counter()
    results_case = run_direct_output_disturbance_lyapunov_mpc(
        system=cstr_case,
        LMPC_obj=ctx.LMPC_obj,
        y_sp_scenario=ctx.y_sp_scenario,
        n_tests=ctx.n_tests,
        set_points_len=ctx.set_points_len,
        steady_states=ctx.steady_states,
        IC_opt=ctx.IC_opt_template.copy(),
        bnds=ctx.bnds,
        L=ctx.L,
        data_min=ctx.data_min,
        data_max=ctx.data_max,
        test_cycle=ctx.test_cycle,
        reward_fn=ctx.reward_fn,
        nominal_qi=ctx.nominal_qi,
        nominal_qs=ctx.nominal_qs,
        nominal_ha=ctx.nominal_hA,
        qi_change=ctx.qi_change,
        qs_change=ctx.qs_change,
        ha_change=ctx.ha_change,
        target_mode="bounded",
        lyapunov_mode="hard",
        target_config=ctx.direct_target_config,
        target_H=None,
        mode=ctx.plant_mode,
        disturbance_after_step=ctx.disturbance_after_step,
        use_target_output_for_tracking=ctx.use_target_output_for_tracking,
        skip_terminal_if_alpha_small=True,
        alpha_terminal_min=1e-8,
        use_target_on_solver_fail=False,
        rho_lyap=ctx.rho_lyap,
        lyap_eps=ctx.lyap_eps,
        slack_penalty=ctx.slack_penalty,
        first_step_contraction_on=True,
        reset_system_on_entry=True,
        solver_options={"warm_start": True},
        force_final_test=ctx.force_final_test,
        disturbance_profile=profile,
    )
    timing = _timing_metadata(timer_start, int(results_case["nFE"]), int(results_case["time_in_sub_episodes"]))
    case_config.update(timing)
    bundle = build_direct_lyapunov_run_bundle(
        source=case_name,
        results=results_case,
        steady_states=ctx.steady_states,
        config=case_config,
        data_min=ctx.data_min,
        data_max=ctx.data_max,
        extra={"reward_config": ctx.reward_config, "min_max_dict": ctx.min_max_dict, "timing": timing, "scenarios": scenarios},
    )
    debug_dir = save_direct_lyapunov_debug_artifacts(
        bundle,
        directory=str(ctx.study_root),
        prefix_name=case_name,
        save_plots=ctx.save_case_plots,
    )
    record = make_direct_lyapunov_comparison_record(case_name, bundle, debug_dir)
    record.update(timing)
    return bundle, Path(debug_dir), record
