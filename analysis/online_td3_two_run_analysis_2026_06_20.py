from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = REPO_ROOT / "results"
REPORT_ROOT = REPO_ROOT / "report"
FIG_ROOT = REPORT_ROOT / "figures" / "2026-06-20_online_td3_two_run"
REPORT_PATH = REPORT_ROOT / "online_td3_two_run_analysis_2026-06-20.md"


@dataclass(frozen=True)
class CaseSpec:
    key: str
    label: str
    short_label: str
    result_root: str
    family: str
    safety_gate: bool
    pretrained: bool
    color: str


CASES = [
    CaseSpec(
        key="cold_gate",
        label="Cold start + gate",
        short_label="Cold gate",
        result_root="OnlineTD3_ColdStart_SafetyGate",
        family="Cold start",
        safety_gate=True,
        pretrained=False,
        color="#3b6ea8",
    ),
    CaseSpec(
        key="cold_nogate",
        label="Cold start no gate",
        short_label="Cold no gate",
        result_root="OnlineTD3_ColdStart_NoSafetyGate",
        family="Cold start",
        safety_gate=False,
        pretrained=False,
        color="#7a9e4f",
    ),
    CaseSpec(
        key="of_gate",
        label="OF-MPC pretrained + gate",
        short_label="OF gate",
        result_root="OnlineTD3_OFMPCPretrained_SafetyGate",
        family="OF-MPC pretrained",
        safety_gate=True,
        pretrained=True,
        color="#c46a3a",
    ),
    CaseSpec(
        key="of_nogate",
        label="OF-MPC pretrained no gate",
        short_label="OF no gate",
        result_root="OnlineTD3_OFMPCPretrained_NoSafetyGate",
        family="OF-MPC pretrained",
        safety_gate=False,
        pretrained=True,
        color="#6f5aa8",
    ),
]


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def rel(path: Path) -> str:
    return str(path.relative_to(REPO_ROOT)).replace("\\", "/")


def is_completed_run(path: Path) -> bool:
    if not path.is_dir() or path.name.startswith("diagnostic"):
        return False
    required = ["record.json", "run_summary.json", "summary.json", "episode_table.csv", "arrays.npz"]
    if any(not (path / name).exists() for name in required):
        return False
    try:
        record = load_json(path / "record.json")
    except Exception:
        return False
    n_steps = int(record.get("wall_clock_n_steps") or record.get("n_steps") or 0)
    return n_steps >= 240_000


def select_latest_completed_runs(spec: CaseSpec, n: int = 2) -> list[Path]:
    root = RESULTS_ROOT / spec.result_root
    candidates = [path for path in root.iterdir() if is_completed_run(path)]
    if len(candidates) < n:
        raise FileNotFoundError(f"Expected at least {n} completed runs under {root}, found {len(candidates)}")
    latest = sorted(candidates, key=lambda path: path.stat().st_mtime, reverse=True)[:n]
    return sorted(latest, key=lambda path: path.stat().st_mtime)


def training_config(run_summary: dict[str, Any]) -> dict[str, Any]:
    return ((run_summary.get("config") or {}).get("training_phase_config") or {})


def reward_config(run_summary: dict[str, Any]) -> dict[str, Any]:
    return ((run_summary.get("config") or {}).get("reward_config") or {})


def case_gate_activity(spec: CaseSpec, record: dict[str, Any]) -> float:
    if spec.safety_gate:
        return float(record.get("actual_intervention_rate", np.nan))
    return float(record.get("diagnostic_unsafe_rate", np.nan))


def summarize_run(spec: CaseSpec, path: Path, execution: int) -> dict[str, Any]:
    record = load_json(path / "record.json")
    run_summary = load_json(path / "run_summary.json")
    config = run_summary.get("config") or {}
    cfg = training_config(run_summary)
    n_steps = int(record.get("wall_clock_n_steps") or record.get("n_steps") or 0)
    target_failures = int(record.get("n_target_failures") or 0)
    diagnostic_unsafe_rate = float(record.get("diagnostic_unsafe_rate", np.nan))
    candidate_pass_rate = (
        float(record.get("accepted_rate", np.nan))
        if spec.safety_gate
        else float(1.0 - diagnostic_unsafe_rate)
    )
    return {
        "case_key": spec.key,
        "case": spec.label,
        "short_label": spec.short_label,
        "family": spec.family,
        "safety_gate": spec.safety_gate,
        "pretrained": spec.pretrained,
        "execution": execution,
        "run_id": path.name,
        "run_dir": rel(path),
        "mtime": pd.Timestamp.fromtimestamp(path.stat().st_mtime).isoformat(),
        "seed": config.get("seed"),
        "episodes": int(record.get("wall_clock_n_episodes") or 0),
        "steps": n_steps,
        "reward_no_penalty": float(record.get("reward_no_penalty_mean", np.nan)),
        "training_reward": float(record.get("reward_mean", np.nan)),
        "fallback_penalty": float(record.get("fallback_penalty_mean", np.nan)),
        "output_rmse_mean": float(record.get("output_rmse_mean", np.nan)),
        "eta_rmse": float(record.get("output0_rmse", np.nan)),
        "T_rmse": float(record.get("output1_rmse", np.nan)),
        "candidate_pass_rate": candidate_pass_rate,
        "accepted_rate": float(record.get("accepted_rate", np.nan)),
        "actual_intervention_rate": float(record.get("actual_intervention_rate", np.nan)),
        "fallback_rate": float(record.get("fallback_rate", np.nan)),
        "diagnostic_unsafe_rate": diagnostic_unsafe_rate,
        "gate_activity_rate": case_gate_activity(spec, record),
        "target_failures": target_failures,
        "target_failure_rate": target_failures / n_steps if n_steps else np.nan,
        "executed_action_gap_inf_mean": float(record.get("executed_action_gap_inf_mean", np.nan)),
        "executed_action_gap_inf_max": float(record.get("executed_action_gap_inf_max", np.nan)),
        "wall_clock_hours": float(record.get("wall_clock_seconds", np.nan)) / 3600.0,
        "projection_backend": config.get("projection_backend"),
        "rl_observation_mode": config.get("rl_observation_mode"),
        "teacher_episodes": cfg.get("behavior_clone_teacher_episodes"),
        "bc_update_mode": cfg.get("bc_update_mode"),
        "bc_exploration_std": cfg.get("bc_exploration_std"),
        "bc_exploration_space": cfg.get("bc_exploration_space"),
        "handoff_episodes": cfg.get("handoff_episodes"),
        "handoff_exploration_std_end": cfg.get("handoff_exploration_std_end"),
        "handoff_exploration_space": cfg.get("handoff_exploration_space"),
        "full_rl_exploration_std_start": cfg.get("full_rl_exploration_std_start"),
        "full_rl_exploration_std_end": cfg.get("full_rl_exploration_std_end"),
        "full_rl_exploration_space": cfg.get("full_rl_exploration_space"),
        "reward_Q_diag": reward_config(run_summary).get("Q_diag"),
        "reward_R_diag": reward_config(run_summary).get("R_diag"),
    }


def phase_name(episode: int, teacher_episodes: int, handoff_episodes: int) -> str:
    if episode <= teacher_episodes:
        return "teacher critic"
    if episode <= teacher_episodes + handoff_episodes:
        return "handoff"
    return "full RL"


def summarize_phase(
    spec: CaseSpec,
    path: Path,
    execution: int,
    ep: pd.DataFrame,
    cfg: dict[str, Any],
) -> list[dict[str, Any]]:
    teacher_eps = int(cfg.get("behavior_clone_teacher_episodes") or 0)
    handoff_eps = int(cfg.get("handoff_episodes") or 0)
    work = ep.copy()
    work["phase"] = [phase_name(int(e), teacher_eps, handoff_eps) for e in work["episode"]]
    rows = []
    for phase in ["teacher critic", "handoff", "full RL", "last 100 episodes"]:
        if phase == "last 100 episodes":
            sub = work[work["episode"] > int(work["episode"].max()) - 100]
        else:
            sub = work[work["phase"] == phase]
        if sub.empty:
            continue
        n_steps = float(sub["n_steps"].sum())
        reward_no_penalty = float(sub["reward_no_penalty_sum"].sum() / n_steps)
        training_reward = float(sub["reward_sum"].sum() / n_steps)
        actual_intervention_rate = float(sub["actual_intervention_count"].sum() / n_steps)
        fallback_rate = float(sub["fallback_count"].sum() / n_steps)
        diagnostic_unsafe_rate = float(sub["diagnostic_unsafe_count"].sum() / n_steps)
        target_failure_rate = float(sub["target_failure_count"].sum() / n_steps)
        rows.append(
            {
                "case_key": spec.key,
                "case": spec.label,
                "short_label": spec.short_label,
                "execution": execution,
                "run_id": path.name,
                "phase": phase,
                "episodes": int(len(sub)),
                "steps": int(n_steps),
                "reward_no_penalty": reward_no_penalty,
                "training_reward": training_reward,
                "fallback_penalty": float(sub["fallback_penalty_sum"].sum() / n_steps),
                "output_rmse_mean": float(sub["output_rmse_mean"].mean()),
                "eta_rmse": float(sub["output0_rmse"].mean()),
                "T_rmse": float(sub["output1_rmse"].mean()),
                "actual_intervention_rate": actual_intervention_rate,
                "fallback_rate": fallback_rate,
                "diagnostic_unsafe_rate": diagnostic_unsafe_rate,
                "gate_activity_rate": actual_intervention_rate if spec.safety_gate else diagnostic_unsafe_rate,
                "target_failure_rate": target_failure_rate,
                "max_executed_action_gap_inf": float(sub["max_executed_action_gap_inf"].max()),
            }
        )
    return rows


def load_episode(spec: CaseSpec, path: Path, execution: int) -> pd.DataFrame:
    ep = pd.read_csv(path / "episode_table.csv")
    run_summary = load_json(path / "run_summary.json")
    cfg = training_config(run_summary)
    teacher_eps = int(cfg.get("behavior_clone_teacher_episodes") or 0)
    handoff_eps = int(cfg.get("handoff_episodes") or 0)
    ep.insert(0, "case_key", spec.key)
    ep.insert(1, "case", spec.label)
    ep.insert(2, "short_label", spec.short_label)
    ep.insert(3, "execution", execution)
    ep.insert(4, "run_id", path.name)
    ep["phase"] = [phase_name(int(e), teacher_eps, handoff_eps) for e in ep["episode"]]
    ep["gate_activity_rate"] = (
        ep["actual_intervention_rate"] if spec.safety_gate else ep["diagnostic_unsafe_rate"]
    )
    return ep


def aggregate_metrics(run_df: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "reward_no_penalty",
        "training_reward",
        "fallback_penalty",
        "output_rmse_mean",
        "eta_rmse",
        "T_rmse",
        "candidate_pass_rate",
        "actual_intervention_rate",
        "fallback_rate",
        "diagnostic_unsafe_rate",
        "gate_activity_rate",
        "target_failure_rate",
        "executed_action_gap_inf_mean",
        "executed_action_gap_inf_max",
    ]
    rows = []
    for key, sub in run_df.groupby("case_key", sort=False):
        first = sub.iloc[0]
        row = {
            "case_key": key,
            "case": first["case"],
            "short_label": first["short_label"],
            "family": first["family"],
            "safety_gate": bool(first["safety_gate"]),
            "pretrained": bool(first["pretrained"]),
            "n_runs": int(len(sub)),
            "run_ids": ", ".join(sub["run_id"].astype(str).tolist()),
            "seeds": ", ".join(str(v) for v in sub["seed"].tolist()),
        }
        for metric in metrics:
            vals = pd.to_numeric(sub[metric], errors="coerce")
            row[f"{metric}_mean"] = float(vals.mean())
            row[f"{metric}_std"] = float(vals.std(ddof=0))
            row[f"{metric}_min"] = float(vals.min())
            row[f"{metric}_max"] = float(vals.max())
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate_phase_metrics(phase_df: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "reward_no_penalty",
        "training_reward",
        "fallback_penalty",
        "output_rmse_mean",
        "eta_rmse",
        "T_rmse",
        "actual_intervention_rate",
        "fallback_rate",
        "diagnostic_unsafe_rate",
        "gate_activity_rate",
        "target_failure_rate",
        "max_executed_action_gap_inf",
    ]
    rows = []
    phase_order = ["teacher critic", "handoff", "full RL", "last 100 episodes"]
    for (case_key, phase), sub in phase_df.groupby(["case_key", "phase"], sort=False):
        first = sub.iloc[0]
        row = {
            "case_key": case_key,
            "case": first["case"],
            "short_label": first["short_label"],
            "phase": phase,
            "phase_order": phase_order.index(phase),
            "n_runs": int(len(sub)),
            "episodes_mean": float(sub["episodes"].mean()),
            "steps_mean": float(sub["steps"].mean()),
        }
        for metric in metrics:
            vals = pd.to_numeric(sub[metric], errors="coerce")
            row[f"{metric}_mean"] = float(vals.mean())
            row[f"{metric}_std"] = float(vals.std(ddof=0))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["case_key", "phase_order"]).reset_index(drop=True)


def duplicate_checks(selected: dict[str, list[Path]]) -> pd.DataFrame:
    rows = []
    compare_arrays = [
        "y_system",
        "u_applied_phys",
        "reward_no_penalty",
        "rewards",
        "actual_intervention_flags",
        "diagnostic_unsafe_flags",
    ]
    ignore_record_keys = {
        "debug_dir",
        "trained_agent_path",
        "wall_clock_seconds",
        "wall_clock_seconds_per_episode",
        "wall_clock_seconds_per_step",
        "wall_clock_steps_per_second",
    }
    for spec in CASES:
        paths = selected[spec.key]
        run_summaries = [load_json(path / "run_summary.json") for path in paths]
        seeds = [(summary.get("config") or {}).get("seed") for summary in run_summaries]
        ep_equal = pd.read_csv(paths[0] / "episode_table.csv").equals(
            pd.read_csv(paths[1] / "episode_table.csv")
        )
        rec0 = load_json(paths[0] / "record.json")
        rec1 = load_json(paths[1] / "record.json")
        record_diffs = [
            key
            for key in sorted(set(rec0) & set(rec1))
            if key not in ignore_record_keys and rec0.get(key) != rec1.get(key)
        ]
        max_abs = 0.0
        arrays_equal = True
        with np.load(paths[0] / "arrays.npz", allow_pickle=True) as arr0, np.load(
            paths[1] / "arrays.npz", allow_pickle=True
        ) as arr1:
            for key in compare_arrays:
                if key not in arr0.files or key not in arr1.files:
                    arrays_equal = False
                    continue
                equal = np.array_equal(arr0[key], arr1[key])
                arrays_equal = arrays_equal and bool(equal)
                if arr0[key].shape == arr1[key].shape:
                    diff = float(np.nanmax(np.abs(arr0[key] - arr1[key])))
                    max_abs = max(max_abs, diff)
        rows.append(
            {
                "case_key": spec.key,
                "case": spec.label,
                "run_id_1": paths[0].name,
                "run_id_2": paths[1].name,
                "seed_1": seeds[0],
                "seed_2": seeds[1],
                "same_seed": seeds[0] == seeds[1],
                "episode_tables_equal": bool(ep_equal),
                "selected_arrays_equal": bool(arrays_equal),
                "selected_arrays_max_abs_diff": max_abs,
                "record_metric_diff_count_excluding_paths_and_wallclock": len(record_diffs),
                "record_metric_diff_keys": ", ".join(record_diffs[:20]),
            }
        )
    return pd.DataFrame(rows)


def format_mean_std(mean: float, std: float, decimals: int = 3, percent: bool = False) -> str:
    if not np.isfinite(mean):
        return ""
    scale = 100.0 if percent else 1.0
    suffix = "%" if percent else ""
    return f"{mean * scale:.{decimals}f} +/- {std * scale:.{decimals}f}{suffix}"


def pct_change(new: float, old: float, lower_better: bool = False) -> float:
    if old == 0 or not np.isfinite(old) or not np.isfinite(new):
        return float("nan")
    if lower_better:
        return (old - new) / abs(old)
    return (new - old) / abs(old)


def md_table(df: pd.DataFrame, columns: list[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] + ["---:" for _ in columns[1:]]) + " |"
    rows = [header, sep]
    for _, row in df.iterrows():
        rows.append("| " + " | ".join("" if pd.isna(row[col]) else str(row[col]) for col in columns) + " |")
    return "\n".join(rows)


def write_csvs(
    run_df: pd.DataFrame,
    aggregate_df: pd.DataFrame,
    phase_df: pd.DataFrame,
    phase_agg_df: pd.DataFrame,
    episode_df: pd.DataFrame,
    dup_df: pd.DataFrame,
) -> None:
    FIG_ROOT.mkdir(parents=True, exist_ok=True)
    run_df.to_csv(FIG_ROOT / "run_metrics.csv", index=False)
    aggregate_df.to_csv(FIG_ROOT / "aggregate_metrics.csv", index=False)
    phase_df.to_csv(FIG_ROOT / "phase_metrics.csv", index=False)
    phase_agg_df.to_csv(FIG_ROOT / "phase_aggregate_metrics.csv", index=False)
    episode_df.to_csv(FIG_ROOT / "episode_metrics.csv", index=False)
    dup_df.to_csv(FIG_ROOT / "duplicate_checks.csv", index=False)


def add_phase_background(ax: plt.Axes) -> None:
    ax.axvspan(1, 10, color="#e8e0cf", alpha=0.35, linewidth=0)
    ax.axvspan(11, 20, color="#d9e7e5", alpha=0.35, linewidth=0)
    ax.axvline(10.5, color="#777777", linewidth=0.8, alpha=0.45)
    ax.axvline(20.5, color="#777777", linewidth=0.8, alpha=0.45)


def save_aggregate_bars(aggregate_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.5), constrained_layout=True)
    metrics = [
        ("reward_no_penalty", "Reward no penalty", False),
        ("output_rmse_mean", "Mean output RMSE", False),
        ("gate_activity_rate", "Intervention or diagnostic unsafe", True),
        ("target_failure_rate", "Target failure rate", True),
    ]
    x = np.arange(len(aggregate_df))
    colors = [spec.color for spec in CASES]
    labels = aggregate_df["short_label"].tolist()
    for ax, (metric, title, percent) in zip(axes.flat, metrics):
        means = aggregate_df[f"{metric}_mean"].to_numpy(dtype=float)
        stds = aggregate_df[f"{metric}_std"].to_numpy(dtype=float)
        if percent:
            means = means * 100.0
            stds = stds * 100.0
        ax.bar(x, means, yerr=stds, color=colors, capsize=3)
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.set_ylabel("% of steps" if percent else "mean value")
        ax.grid(axis="y", alpha=0.25)
    fig.savefig(FIG_ROOT / "aggregate_bar_metrics.png", dpi=180)
    plt.close(fig)


def save_tradeoff_scatter(aggregate_df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 6.2), constrained_layout=True)
    for spec in CASES:
        row = aggregate_df[aggregate_df["case_key"] == spec.key].iloc[0]
        size = 90 + 1500 * float(row["gate_activity_rate_mean"])
        marker = "o" if spec.safety_gate else "s"
        ax.scatter(
            row["output_rmse_mean_mean"],
            row["reward_no_penalty_mean"],
            s=size,
            color=spec.color,
            marker=marker,
            edgecolor="#222222",
            linewidth=0.8,
            label=spec.short_label,
        )
        ax.annotate(
            spec.short_label,
            (row["output_rmse_mean_mean"], row["reward_no_penalty_mean"]),
            xytext=(6, 5),
            textcoords="offset points",
            fontsize=9,
        )
    ax.set_xlabel("Mean output RMSE, lower is better")
    ax.set_ylabel("Reward no penalty, higher is better")
    ax.set_title("Reward-tracking-safety tradeoff")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.savefig(FIG_ROOT / "reward_tracking_safety_tradeoff.png", dpi=180)
    plt.close(fig)


def save_episode_trends(episode_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(12.5, 10), sharex=True, constrained_layout=True)
    metrics = [
        ("reward_no_penalty_mean", "Reward no penalty"),
        ("output_rmse_mean", "Output RMSE"),
        ("gate_activity_rate", "Intervention or diagnostic unsafe rate"),
    ]
    for ax, (metric, ylabel) in zip(axes, metrics):
        add_phase_background(ax)
        for spec in CASES:
            sub = episode_df[episode_df["case_key"] == spec.key]
            grouped = sub.groupby("episode")[metric].agg(["mean", "std"]).reset_index()
            grouped["std"] = grouped["std"].fillna(0.0)
            grouped["mean_smooth"] = grouped["mean"].rolling(10, min_periods=1).mean()
            grouped["std_smooth"] = grouped["std"].rolling(10, min_periods=1).mean()
            x = grouped["episode"].to_numpy(dtype=float)
            y = grouped["mean_smooth"].to_numpy(dtype=float)
            s = grouped["std_smooth"].to_numpy(dtype=float)
            ax.plot(x, y, color=spec.color, linewidth=1.5, label=spec.short_label)
            if np.nanmax(s) > 0:
                ax.fill_between(x, y - s, y + s, color=spec.color, alpha=0.15, linewidth=0)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
    axes[-1].set_xlabel("Episode")
    axes[0].legend(ncol=4, fontsize=8)
    fig.savefig(FIG_ROOT / "episode_trends_mean.png", dpi=180)
    plt.close(fig)


def save_phase_bars(phase_agg_df: pd.DataFrame) -> None:
    phase_order = ["teacher critic", "handoff", "full RL", "last 100 episodes"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2), constrained_layout=True)
    metrics = [
        ("reward_no_penalty", "Reward no penalty"),
        ("output_rmse_mean", "Output RMSE"),
        ("gate_activity_rate", "Intervention or diagnostic unsafe"),
    ]
    width = 0.18
    x = np.arange(len(phase_order))
    for case_idx, spec in enumerate(CASES):
        sub = (
            phase_agg_df[phase_agg_df["case_key"] == spec.key]
            .set_index("phase")
            .reindex(phase_order)
        )
        offset = (case_idx - 1.5) * width
        for ax, (metric, title) in zip(axes, metrics):
            ax.bar(
                x + offset,
                sub[f"{metric}_mean"],
                yerr=sub[f"{metric}_std"],
                width=width,
                color=spec.color,
                capsize=2,
                label=spec.short_label,
            )
            ax.set_title(title)
            ax.set_xticks(x)
            ax.set_xticklabels(["teacher", "handoff", "full RL", "last 100"], rotation=20, ha="right")
            ax.grid(axis="y", alpha=0.25)
    axes[0].legend(fontsize=8)
    fig.savefig(FIG_ROOT / "phase_metric_bars.png", dpi=180)
    plt.close(fig)


def save_final_episode_tracking(selected: dict[str, list[Path]]) -> None:
    fig, axes = plt.subplots(len(CASES), 2, figsize=(13.2, 10.8), sharex=True, constrained_layout=True)
    for row, spec in enumerate(CASES):
        path = selected[spec.key][-1]
        ep = pd.read_csv(path / "episode_table.csv").iloc[-1]
        start = int(ep["step_start"])
        stop = int(ep["step_stop_exclusive"])
        with np.load(path / "arrays.npz", allow_pickle=True) as arrays:
            y = np.asarray(arrays["y_system"], dtype=float)[:-1, :][start:stop]
            y_sp = np.asarray(arrays["y_sp_phys_store"], dtype=float)[start:stop]
        t = np.arange(stop - start)
        for col, name in enumerate(["eta", "T"]):
            ax = axes[row, col]
            ax.plot(t, y[:, col], color=spec.color, linewidth=1.2, label="output")
            ax.plot(t, y_sp[:, col], color="#222222", linestyle="--", linewidth=0.9, label="setpoint")
            ax.set_title(f"{spec.short_label}: {name}")
            ax.grid(alpha=0.25)
            if row == len(CASES) - 1:
                ax.set_xlabel("Step in final episode")
            if col == 0:
                ax.set_ylabel("Physical output")
            if row == 0 and col == 1:
                ax.legend(fontsize=8)
    fig.savefig(FIG_ROOT / "final_episode_tracking_representative.png", dpi=180)
    plt.close(fig)


def save_final_episode_inputs(selected: dict[str, list[Path]]) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(12.5, 7), sharex=True, constrained_layout=True)
    for spec in CASES:
        path = selected[spec.key][-1]
        ep = pd.read_csv(path / "episode_table.csv").iloc[-1]
        start = int(ep["step_start"])
        stop = int(ep["step_stop_exclusive"])
        with np.load(path / "arrays.npz", allow_pickle=True) as arrays:
            u = np.asarray(arrays["u_applied_phys"], dtype=float)[start:stop]
        t = np.arange(stop - start)
        axes[0].plot(t, u[:, 0], color=spec.color, linewidth=1.1, label=spec.short_label)
        axes[1].plot(t, u[:, 1], color=spec.color, linewidth=1.1, label=spec.short_label)
    axes[0].set_ylabel("Qc")
    axes[1].set_ylabel("Qm")
    axes[1].set_xlabel("Step in final episode")
    for ax in axes:
        ax.grid(alpha=0.25)
    axes[0].legend(ncol=4, fontsize=8)
    fig.savefig(FIG_ROOT / "final_episode_inputs_representative.png", dpi=180)
    plt.close(fig)


def make_report_tables(
    run_df: pd.DataFrame,
    aggregate_df: pd.DataFrame,
    phase_agg_df: pd.DataFrame,
    dup_df: pd.DataFrame,
) -> dict[str, str]:
    data_rows = []
    for _, row in aggregate_df.iterrows():
        subset = run_df[run_df["case_key"] == row["case_key"]]
        data_rows.append(
            {
                "case": row["case"],
                "run_ids": row["run_ids"],
                "seed": row["seeds"],
                "episodes": int(subset["episodes"].iloc[0]),
                "steps": int(subset["steps"].iloc[0]),
            }
        )
    data_table = md_table(pd.DataFrame(data_rows), ["case", "run_ids", "seed", "episodes", "steps"])

    perf_rows = []
    for _, row in aggregate_df.iterrows():
        perf_rows.append(
            {
                "case": row["case"],
                "reward_no_penalty": format_mean_std(
                    row["reward_no_penalty_mean"], row["reward_no_penalty_std"], decimals=3
                ),
                "training_reward": format_mean_std(
                    row["training_reward_mean"], row["training_reward_std"], decimals=3
                ),
                "output_rmse": format_mean_std(
                    row["output_rmse_mean_mean"], row["output_rmse_mean_std"], decimals=3
                ),
                "eta_rmse": format_mean_std(row["eta_rmse_mean"], row["eta_rmse_std"], decimals=3),
                "T_rmse": format_mean_std(row["T_rmse_mean"], row["T_rmse_std"], decimals=3),
            }
        )
    perf_table = md_table(
        pd.DataFrame(perf_rows),
        ["case", "reward_no_penalty", "training_reward", "output_rmse", "eta_rmse", "T_rmse"],
    )

    safety_rows = []
    for _, row in aggregate_df.iterrows():
        safety_rows.append(
            {
                "case": row["case"],
                "candidate_pass": format_mean_std(
                    row["candidate_pass_rate_mean"], row["candidate_pass_rate_std"], decimals=2, percent=True
                ),
                "activity": format_mean_std(
                    row["gate_activity_rate_mean"], row["gate_activity_rate_std"], decimals=2, percent=True
                ),
                "actual_intervention": format_mean_std(
                    row["actual_intervention_rate_mean"],
                    row["actual_intervention_rate_std"],
                    decimals=2,
                    percent=True,
                ),
                "fallback": format_mean_std(
                    row["fallback_rate_mean"], row["fallback_rate_std"], decimals=2, percent=True
                ),
                "target_failure": format_mean_std(
                    row["target_failure_rate_mean"], row["target_failure_rate_std"], decimals=2, percent=True
                ),
            }
        )
    safety_table = md_table(
        pd.DataFrame(safety_rows),
        ["case", "candidate_pass", "activity", "actual_intervention", "fallback", "target_failure"],
    )

    last100_rows = []
    last100 = phase_agg_df[phase_agg_df["phase"] == "last 100 episodes"]
    for _, row in last100.iterrows():
        last100_rows.append(
            {
                "case": row["case"],
                "reward_no_penalty": format_mean_std(
                    row["reward_no_penalty_mean"], row["reward_no_penalty_std"], decimals=3
                ),
                "output_rmse": format_mean_std(
                    row["output_rmse_mean_mean"], row["output_rmse_mean_std"], decimals=3
                ),
                "activity": format_mean_std(
                    row["gate_activity_rate_mean"], row["gate_activity_rate_std"], decimals=2, percent=True
                ),
                "fallback": format_mean_std(
                    row["fallback_rate_mean"], row["fallback_rate_std"], decimals=2, percent=True
                ),
            }
        )
    last100_table = md_table(
        pd.DataFrame(last100_rows),
        ["case", "reward_no_penalty", "output_rmse", "activity", "fallback"],
    )

    phase_rows = []
    phase_subset = phase_agg_df[phase_agg_df["phase"].isin(["teacher critic", "handoff", "full RL"])]
    for _, row in phase_subset.iterrows():
        phase_rows.append(
            {
                "case": row["case"],
                "phase": row["phase"],
                "reward_no_penalty": f"{row['reward_no_penalty_mean']:.3f}",
                "output_rmse": f"{row['output_rmse_mean_mean']:.3f}",
                "activity": f"{100.0 * row['gate_activity_rate_mean']:.2f}%",
            }
        )
    phase_table = md_table(
        pd.DataFrame(phase_rows),
        ["case", "phase", "reward_no_penalty", "output_rmse", "activity"],
    )

    dup_rows = []
    for _, row in dup_df.iterrows():
        dup_rows.append(
            {
                "case": row["case"],
                "same_seed": str(bool(row["same_seed"])),
                "episode_equal": str(bool(row["episode_tables_equal"])),
                "arrays_equal": str(bool(row["selected_arrays_equal"])),
                "max_abs_diff": f"{row['selected_arrays_max_abs_diff']:.1e}",
            }
        )
    dup_table = md_table(
        pd.DataFrame(dup_rows),
        ["case", "same_seed", "episode_equal", "arrays_equal", "max_abs_diff"],
    )

    return {
        "data": data_table,
        "performance": perf_table,
        "safety": safety_table,
        "last100": last100_table,
        "phase": phase_table,
        "duplicates": dup_table,
    }


def write_report(
    run_df: pd.DataFrame,
    aggregate_df: pd.DataFrame,
    phase_agg_df: pd.DataFrame,
    dup_df: pd.DataFrame,
) -> None:
    tables = make_report_tables(run_df, aggregate_df, phase_agg_df, dup_df)
    row = aggregate_df.set_index("case_key")
    cold_gate = row.loc["cold_gate"]
    cold_nogate = row.loc["cold_nogate"]
    of_gate = row.loc["of_gate"]
    of_nogate = row.loc["of_nogate"]
    of_vs_cold_gate_rmse = pct_change(
        of_gate["output_rmse_mean_mean"], cold_gate["output_rmse_mean_mean"], lower_better=True
    )
    of_vs_cold_nogate_rmse = pct_change(
        of_nogate["output_rmse_mean_mean"], cold_nogate["output_rmse_mean_mean"], lower_better=True
    )
    of_vs_cold_gate_activity = pct_change(
        of_gate["gate_activity_rate_mean"], cold_gate["gate_activity_rate_mean"], lower_better=True
    )
    of_vs_cold_nogate_activity = pct_change(
        of_nogate["gate_activity_rate_mean"], cold_nogate["gate_activity_rate_mean"], lower_better=True
    )

    text = f"""# Two-Run Online TD3 Analysis

Date: 2026-06-20

## Objective

This report compares the two latest completed executions of each active online
TD3 disturbance runner:

- cold start with the GART-LMPC safety gate
- cold start without active safety intervention
- OF-MPC-pretrained with the GART-LMPC safety gate
- OF-MPC-pretrained without active safety intervention

The comparison focuses on three questions: reward, tracking, and intervention
burden. It also checks whether the two executions are independent stochastic
runs or deterministic repeats.

## Data Used

{tables["data"]}

Full metric exports are stored under
`report/figures/2026-06-20_online_td3_two_run/`.

## Reproducibility Check

{tables["duplicates"]}

The two executions per runner used the same configured seed, `123`. Their
episode tables and selected trajectory arrays are identical for every case.
Therefore the two executions are useful as deterministic reproducibility checks,
but they are not independent seed replicates. The aggregate tables below report
mean +/- standard deviation across the two executions, but the zero standard
deviation should not be interpreted as statistical robustness.

## Method

All selected runs use the current active online schedule:

$$
N_{{\\mathrm{{teacher}}}} = 10,
\\qquad
\\text{{teacher update}} = \\text{{critic TD only}},
\\qquad
N_{{\\mathrm{{handoff}}}} = 10.
$$

The teacher behavior is noisy GART-LMPC in scaled input-deviation coordinates.
The handoff and full-RL exploration are also applied in `input_dev` coordinates.
The online reward uses

$$
r_k = r_{{\\mathrm{{track/move}}, k}}
      - r_{{\\mathrm{{fallback/event}}, k}},
$$

where `reward_no_penalty` is the tracking and move-quality component before
safety fallback penalties. Because gate runs can receive fallback/event
penalties while no-gate runs do not, `reward_no_penalty` is the fairer
cross-controller control-performance comparison. The `training_reward` column is
still reported because it is what TD3 actually optimizes online.

For safety-gate runs, `activity` means actual intervention rate: the fraction
of steps where the executed action differs from the TD3 candidate because of
fallback or hold-previous logic. For no-gate runs, `activity` means diagnostic
unsafe rate: the fraction of candidate actions that would have failed the GART
diagnostic gate, while the candidate was still executed.

Important timing note: these online runs were completed before the later
OF-MPC offline pretraining reward-label change. They evaluate the current online
runner configuration, not a regenerated OF-MPC-pretrained checkpoint.

## Reward And Tracking

{tables["performance"]}

![Aggregate metrics](figures/2026-06-20_online_td3_two_run/aggregate_bar_metrics.png)

![Reward tracking safety tradeoff](figures/2026-06-20_online_td3_two_run/reward_tracking_safety_tradeoff.png)

The nominal rankings are close but not identical. By `reward_no_penalty`, the
order is:

1. OF-MPC pretrained no gate
2. cold start no gate
3. OF-MPC pretrained + gate
4. cold start + gate

By mean output RMSE, the order is:

1. OF-MPC pretrained no gate
2. OF-MPC pretrained + gate
3. cold start no gate
4. cold start + gate

This mismatch is useful: the OF-MPC gate run tracks better than cold no-gate,
but the reward still reflects move usage and gate-compatible behavior, not only
output RMSE.

OF-MPC pretraining improves the active-gate tracking RMSE by
{100.0 * of_vs_cold_gate_rmse:.1f}% relative to cold start with gate. It improves
the no-gate tracking RMSE by {100.0 * of_vs_cold_nogate_rmse:.1f}% relative to
cold start no gate. This supports the methodological value of starting online
TD3 from an MPC-shaped policy rather than relying only on cold-start online
learning.

## Intervention And Safety Burden

{tables["safety"]}

The safety story is different from the nominal reward story. The no-gate cases
achieve better nominal tracking, but they also execute actions that the
diagnostic GART gate marks unsafe. The cold-start no-gate case has the largest
diagnostic unsafe rate, while OF-MPC pretraining reduces that diagnostic burden
by {100.0 * of_vs_cold_nogate_activity:.1f}%.

Among active-gate runs, OF-MPC pretraining reduces the intervention burden by
{100.0 * of_vs_cold_gate_activity:.1f}% relative to cold start with gate. This
is the strongest evidence in this batch for the combined methodology: pretraining
moves the policy closer to the safe controller manifold, and the gate handles
the remaining unsafe candidates.

## Phase Behavior

{tables["phase"]}

![Phase metrics](figures/2026-06-20_online_td3_two_run/phase_metric_bars.png)

The teacher phase is almost identical across cold-start and pretrained cases
because behavior is supplied by GART-LMPC and actor BC is disabled. Differences
emerge during handoff and full RL. The no-gate cold-start run improves reward
and tracking in full RL, but its diagnostic unsafe rate also grows. The OF-MPC
pretrained no-gate run keeps the best full-RL tracking while reducing the
diagnostic unsafe load.

## Late-Training Behavior

{tables["last100"]}

![Episode trends](figures/2026-06-20_online_td3_two_run/episode_trends_mean.png)

The last 100 episodes show the same ranking as the full-run averages. The
OF-MPC no-gate run is the best nominal tracker, but it is not intervention-safe.
The OF-MPC gate run is the strongest active-gate result and has a lower late
intervention rate than the cold-start gate run.

## Representative Final-Episode Trajectories

Because the two executions per case are trajectory-identical, the plots below
use the newest execution for each runner as the representative final episode.

![Final episode tracking](figures/2026-06-20_online_td3_two_run/final_episode_tracking_representative.png)

![Final episode inputs](figures/2026-06-20_online_td3_two_run/final_episode_inputs_representative.png)

## Interpretation

The result supports a balanced methodological claim rather than a simple
"safety gate always improves tracking" claim.

- The no-gate controllers are useful nominal-performance upper bounds, but they
  execute candidate actions that the diagnostic GART gate rejects.
- The active safety gate reduces risk by replacing unsafe candidates, but the
  replacement produces a reward and tracking cost, especially for cold start.
- OF-MPC pretraining improves both nominal performance and safety compatibility.
  It gives the best no-gate tracker and the best active-gate tracker.
- The strongest argument for the method is therefore the combination:
  pretraining reduces how often the gate must intervene, and the gate remains as
  a certification layer for the residual unsafe actions.

## Risks And Consistency Checks

- The two executions per runner are same-seed deterministic repeats, not
  independent seeds.
- Existing OF-MPC-pretrained checkpoints were not regenerated after the offline
  reward-label change, so this report should not be used to evaluate that later
  change.
- `training_reward` is not directly fair across gate and no-gate cases because
  gate cases include fallback/event penalties. Use `reward_no_penalty` for the
  main control-performance comparison.
- No-gate `activity` is diagnostic-only. It does not mean the action was
  replaced.

## Recommended Next Experiment

Run the same four active online runners with distinct seeds, for example
`123`, `124`, and `125`, while preserving the current schedule and probe-style
input exploration. The confirming result would be:

- OF-MPC pretrained + gate keeps lower RMSE than cold start + gate.
- OF-MPC pretrained + gate keeps lower intervention and fallback rates than
  cold start + gate.
- OF-MPC pretrained no gate keeps a lower diagnostic unsafe rate than cold start
  no gate.
- The qualitative ranking remains stable across independent seeds.

After regenerating OF-MPC-pretrained checkpoints with the aligned offline reward
labels, repeat this same report. That will separate the benefit of online
configuration changes from the benefit of the corrected offline critic reward
labels.
"""
    REPORT_PATH.write_text(text, encoding="utf-8")


def main() -> None:
    FIG_ROOT.mkdir(parents=True, exist_ok=True)
    selected = {spec.key: select_latest_completed_runs(spec, n=2) for spec in CASES}
    run_rows = []
    phase_rows = []
    episode_frames = []
    for spec in CASES:
        for execution, path in enumerate(selected[spec.key], start=1):
            run_rows.append(summarize_run(spec, path, execution))
            run_summary = load_json(path / "run_summary.json")
            cfg = training_config(run_summary)
            ep = load_episode(spec, path, execution)
            phase_rows.extend(summarize_phase(spec, path, execution, ep, cfg))
            episode_frames.append(ep)
    run_df = pd.DataFrame(run_rows)
    aggregate_df = aggregate_metrics(run_df)
    phase_df = pd.DataFrame(phase_rows)
    phase_agg_df = aggregate_phase_metrics(phase_df)
    episode_df = pd.concat(episode_frames, ignore_index=True)
    dup_df = duplicate_checks(selected)

    write_csvs(run_df, aggregate_df, phase_df, phase_agg_df, episode_df, dup_df)
    save_aggregate_bars(aggregate_df)
    save_tradeoff_scatter(aggregate_df)
    save_episode_trends(episode_df)
    save_phase_bars(phase_agg_df)
    save_final_episode_tracking(selected)
    save_final_episode_inputs(selected)
    write_report(run_df, aggregate_df, phase_agg_df, dup_df)
    print(f"Wrote {REPORT_PATH.relative_to(REPO_ROOT)}")
    print(f"Wrote figures and CSVs under {FIG_ROOT.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
