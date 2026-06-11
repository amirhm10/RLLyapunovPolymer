"""Analyze the June 10 disturbance-only TD3/LMPC/OF-MPC runner batch.

The script intentionally reads the saved result bundles instead of relying on
console output. It writes reproducible tables and figures used by the
corresponding Markdown report.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT_DIR = ROOT / "report" / "figures" / "2026-06-10_online_disturbance_8_runner_analysis"

RUNNERS = [
    ("LMPC pretrained + gate", "OnlineTD3_LMPCPretrained_SafetyGate", "online_gate"),
    ("OF-MPC pretrained + gate", "OnlineTD3_OFMPCPretrained_SafetyGate", "online_gate"),
    ("LMPC pretrained no gate", "OnlineTD3_LMPCPretrained_NoSafetyGate", "online_no_gate"),
    ("OF-MPC pretrained no gate", "OnlineTD3_OFMPCPretrained_NoSafetyGate", "online_no_gate"),
    ("Cold start + gate", "OnlineTD3_ColdStart_SafetyGate", "online_gate"),
    ("Cold start no gate", "OnlineTD3_ColdStart_NoSafetyGate", "online_no_gate"),
    ("Direct LMPC baseline", "DirectLMPCDisturbance", "baseline"),
    ("OF-MPC baseline", "OffsetFreeMPCDisturbance", "baseline"),
]

HISTORICAL_DIRECT_LYAP = [
    ("2026-05-20 full bounded MPC monitor", "20260520_165653", "mpc_only"),
    ("2026-05-20 full bounded Lyap", "20260520_165653", "lyap_mix_u0p1_x0p1_lex"),
    ("2026-05-23 short bounded MPC monitor", "20260523_011436", "mpc_only"),
    ("2026-06-05 governed MPC monitor A", "20260605_143910", "mpc_only"),
    ("2026-06-05 governed MPC monitor B", "20260605_144901", "mpc_only"),
    ("2026-06-05 governed MPC monitor C", "20260605_155254", "mpc_only"),
    ("2026-06-06 governed Direct LMPC", "20260606_020549", "lyap_governed_reference"),
    ("2026-06-06 governed MPC monitor", "20260606_020549", "mpc_only"),
]


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _latest_complete_run(runner_dir: Path, min_steps: int = 200_000) -> Path:
    candidates: list[tuple[str, Path]] = []
    for child in runner_dir.iterdir():
        if not child.is_dir():
            continue
        summary_path = child / "summary.json"
        arrays_path = child / "arrays.npz"
        episode_path = child / "episode_table.csv"
        if not (summary_path.exists() and arrays_path.exists() and episode_path.exists()):
            continue
        try:
            summary = _read_json(summary_path)
        except Exception:
            continue
        if int(summary.get("n_steps", 0)) >= min_steps:
            candidates.append((child.name, child))
    if not candidates:
        raise FileNotFoundError(f"No complete run found in {runner_dir}")
    return sorted(candidates, key=lambda item: item[0])[-1][1]


def _output_errors(arrays: np.lib.npyio.NpzFile) -> np.ndarray:
    if "y_minus_y_sp_phys_store" in arrays.files:
        return np.asarray(arrays["y_minus_y_sp_phys_store"], dtype=float)
    if "y_system" in arrays.files and "y_sp_phys_store" in arrays.files:
        return np.asarray(arrays["y_system"][1:], dtype=float) - np.asarray(
            arrays["y_sp_phys_store"], dtype=float
        )
    raise KeyError("Could not find physical output error arrays")


def _safe_mean(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return float("nan")
    return float(np.nanmean(values))


def _sum_column(df: pd.DataFrame, name: str) -> float:
    if name not in df.columns:
        return 0.0
    return float(df[name].fillna(0.0).sum())


def _mean_column(df: pd.DataFrame, name: str) -> float:
    if name not in df.columns:
        return float("nan")
    return float(df[name].mean())


def _episode_end_column(episode: pd.DataFrame) -> str:
    if "step_stop_exclusive" in episode.columns:
        return "step_stop_exclusive"
    if "step_end_exclusive" in episode.columns:
        return "step_end_exclusive"
    raise KeyError("Episode table has no exclusive end-step column")


def _enrich_episode_table(
    episode: pd.DataFrame,
    err: np.ndarray,
    reward_no_penalty: np.ndarray,
    rewards: np.ndarray,
    fallback_penalty: np.ndarray,
) -> pd.DataFrame:
    episode = episode.copy()
    end_col = _episode_end_column(episode)

    if "reward_no_penalty_mean" not in episode.columns:
        episode["reward_no_penalty_mean"] = [
            float(np.mean(reward_no_penalty[int(start) : int(stop)]))
            for start, stop in zip(episode["step_start"], episode[end_col])
        ]
    if "reward_mean" not in episode.columns:
        episode["reward_mean"] = [
            float(np.mean(rewards[int(start) : int(stop)]))
            for start, stop in zip(episode["step_start"], episode[end_col])
        ]
    if "fallback_penalty_mean" not in episode.columns:
        episode["fallback_penalty_mean"] = [
            float(np.mean(fallback_penalty[int(start) : int(stop)]))
            for start, stop in zip(episode["step_start"], episode[end_col])
        ]
    if "output_rmse_mean" not in episode.columns:
        episode["output_rmse_mean"] = [
            float(np.mean(np.sqrt(np.mean(err[int(start) : int(stop)] ** 2, axis=0))))
            for start, stop in zip(episode["step_start"], episode[end_col])
        ]
    if "output0_rmse" not in episode.columns or "output1_rmse" not in episode.columns:
        rmse_by_episode = [
            np.sqrt(np.mean(err[int(start) : int(stop)] ** 2, axis=0))
            for start, stop in zip(episode["step_start"], episode[end_col])
        ]
        episode["output0_rmse"] = [float(row[0]) for row in rmse_by_episode]
        episode["output1_rmse"] = [float(row[1]) for row in rmse_by_episode]
    for count_col, rate_col in [
        ("diagnostic_unsafe_count", "diagnostic_unsafe_rate"),
        ("actual_intervention_count", "actual_intervention_rate"),
        ("fallback_count", "fallback_rate"),
        ("fallback_verified_count", "fallback_verified_rate"),
        ("solver_fail_hold_prev_count", "solver_fail_hold_prev_rate"),
    ]:
        if rate_col not in episode.columns:
            if count_col in episode.columns:
                episode[rate_col] = episode[count_col] / episode["n_steps"].replace(0, np.nan)
            else:
                episode[rate_col] = 0.0
        if count_col not in episode.columns:
            episode[count_col] = 0
    for optional_count in [
        "fallback_unverified_count",
        "target_fail_hold_prev_count",
        "target_failure_count",
    ]:
        if optional_count not in episode.columns:
            episode[optional_count] = 0
    return episode


def collect_latest_metrics() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, str]]:
    rows: list[dict[str, Any]] = []
    phase_rows: list[dict[str, Any]] = []
    block_rows: list[dict[str, Any]] = []
    manifest: dict[str, str] = {}

    for label, dirname, family in RUNNERS:
        run_dir = _latest_complete_run(RESULTS / dirname)
        manifest[label] = str(run_dir.relative_to(ROOT))
        summary = _read_json(run_dir / "summary.json")
        run_summary = _read_json(run_dir / "run_summary.json")
        episode = pd.read_csv(run_dir / "episode_table.csv")
        arrays = np.load(run_dir / "arrays.npz", allow_pickle=True)
        err = _output_errors(arrays)

        output_rmse = np.sqrt(np.mean(err**2, axis=0))
        output_mae = np.mean(np.abs(err), axis=0)
        reward_no_penalty = np.asarray(arrays["reward_no_penalty"], dtype=float)
        rewards = np.asarray(arrays["rewards"], dtype=float)
        if "fallback_penalty" in arrays.files:
            fallback_penalty = np.asarray(arrays["fallback_penalty"], dtype=float)
        else:
            fallback_penalty = np.zeros_like(rewards)
        episode = _enrich_episode_table(episode, err, reward_no_penalty, rewards, fallback_penalty)

        tail = episode.tail(50)
        last = episode.tail(1)
        best_idx = episode["reward_no_penalty_mean"].idxmax()
        best = episode.loc[best_idx]

        safety_gate_active = bool(run_summary.get("config", {}).get("safety_gate_active", False))
        actual_intervention_rate = float(summary.get("actual_intervention_rate", np.nan))
        row = {
            "case": label,
            "family": family,
            "run_dir": str(run_dir.relative_to(ROOT)),
            "n_steps": int(summary.get("n_steps", len(reward_no_penalty))),
            "n_episodes": int(run_summary.get("config", {}).get("n_episodes", len(episode))),
            "reward_mean": _safe_mean(rewards),
            "reward_no_penalty_mean": _safe_mean(reward_no_penalty),
            "fallback_penalty_mean": _safe_mean(fallback_penalty),
            "reward_penalty_gap": _safe_mean(reward_no_penalty) - _safe_mean(rewards),
            "output0_rmse": float(output_rmse[0]),
            "output1_rmse": float(output_rmse[1]),
            "output_rmse_mean": float(np.mean(output_rmse)),
            "output0_mae": float(output_mae[0]),
            "output1_mae": float(output_mae[1]),
            "output_mae_mean": float(np.mean(output_mae)),
            "output_max_abs_error": float(np.max(np.abs(err))),
            "tail50_reward_no_penalty": _mean_column(tail, "reward_no_penalty_mean"),
            "tail50_reward": _mean_column(tail, "reward_mean"),
            "tail50_output_rmse_mean": _mean_column(tail, "output_rmse_mean"),
            "tail50_actual_intervention_rate": _mean_column(tail, "actual_intervention_rate"),
            "tail50_diagnostic_unsafe_rate": _mean_column(tail, "diagnostic_unsafe_rate"),
            "last_reward_no_penalty": float(last["reward_no_penalty_mean"].iloc[0]),
            "last_output_rmse_mean": float(last["output_rmse_mean"].iloc[0]),
            "last_actual_intervention_rate": float(last.get("actual_intervention_rate", pd.Series([0.0])).iloc[0]),
            "last_diagnostic_unsafe_rate": float(last.get("diagnostic_unsafe_rate", pd.Series([0.0])).iloc[0]),
            "best_episode": int(best["episode"]),
            "best_reward_no_penalty": float(best["reward_no_penalty_mean"]),
            "best_output_rmse_mean": float(best["output_rmse_mean"]),
            "actual_intervention_rate": actual_intervention_rate,
            "actual_gate_intervention_rate": actual_intervention_rate if safety_gate_active else 0.0,
            "diagnostic_unsafe_rate": float(summary.get("diagnostic_unsafe_rate", np.nan)),
            "fallback_rate": float(summary.get("fallback_rate", np.nan)),
            "accepted_rate": float(summary.get("accepted_rate", np.nan)),
            "fallback_count": int(summary.get("n_fallback", _sum_column(episode, "fallback_count"))),
            "fallback_verified_count": int(
                summary.get("n_fallback_mpc_verified", _sum_column(episode, "fallback_verified_count"))
            ),
            "solver_fail_hold_prev_count": int(
                summary.get("n_solver_fail_hold_prev", _sum_column(episode, "solver_fail_hold_prev_count"))
            ),
            "target_fail_hold_prev_count": int(
                summary.get("n_target_fail_hold_prev", _sum_column(episode, "target_fail_hold_prev_count"))
            ),
            "teacher_source": run_summary.get("config", {}).get("teacher_source", ""),
            "safety_gate": safety_gate_active,
            "controller": run_summary.get("controller", run_summary.get("config", {}).get("controller_mode", "")),
            "wall_clock_seconds": float(summary.get("wall_clock_seconds", np.nan)),
        }
        rows.append(row)

        phase_defs = [
            ("BC", episode["episode"].between(1, 20)),
            ("handoff", episode["episode"].between(21, 25)),
            ("full TD3", episode["episode"].between(26, 300)),
            ("tail 50", episode["episode"].between(max(1, int(episode["episode"].max()) - 49), int(episode["episode"].max()))),
        ]
        for phase, mask in phase_defs:
            part = episode.loc[mask]
            if part.empty:
                continue
            phase_rows.append(
                {
                    "case": label,
                    "phase": phase,
                    "n_episodes": int(len(part)),
                    "reward_mean": _mean_column(part, "reward_mean"),
                    "reward_no_penalty_mean": _mean_column(part, "reward_no_penalty_mean"),
                    "fallback_penalty_mean": _mean_column(part, "fallback_penalty_mean"),
                    "output_rmse_mean": _mean_column(part, "output_rmse_mean"),
                    "actual_intervention_rate": _mean_column(part, "actual_intervention_rate"),
                    "diagnostic_unsafe_rate": _mean_column(part, "diagnostic_unsafe_rate"),
                    "fallback_count": int(_sum_column(part, "fallback_count")),
                    "fallback_verified_count": int(_sum_column(part, "fallback_verified_count")),
                    "solver_fail_hold_prev_count": int(_sum_column(part, "solver_fail_hold_prev_count")),
                }
            )

        step_index = np.arange(len(reward_no_penalty))
        episode_index = step_index // 800 + 1
        block_index = (step_index % 800) // 400
        for tail_only, tail_label in [(False, "all episodes"), (True, "tail 50")]:
            base_mask = episode_index >= (len(episode) - 49) if tail_only else np.ones_like(step_index, dtype=bool)
            for block_id, block_label in [(0, "S1 high [4.5, 324]"), (1, "S2 low [3.4, 321]")]:
                mask = base_mask & (block_index == block_id)
                if not np.any(mask):
                    continue
                err_block = err[mask]
                rmse = np.sqrt(np.mean(err_block**2, axis=0))
                block_rows.append(
                    {
                        "case": label,
                        "window": tail_label,
                        "block": block_label,
                        "n_steps": int(np.sum(mask)),
                        "reward_no_penalty_mean": float(np.mean(reward_no_penalty[mask])),
                        "reward_mean": float(np.mean(rewards[mask])),
                        "output0_rmse": float(rmse[0]),
                        "output1_rmse": float(rmse[1]),
                        "output_rmse_mean": float(np.mean(rmse)),
                        "actual_intervention_rate": float(np.mean(arrays["actual_intervention_flags"][mask]))
                        if "actual_intervention_flags" in arrays.files
                        else 0.0,
                        "diagnostic_unsafe_rate": float(np.mean(arrays["diagnostic_unsafe_flags"][mask]))
                        if "diagnostic_unsafe_flags" in arrays.files
                        else 0.0,
                    }
                )

    return pd.DataFrame(rows), pd.DataFrame(phase_rows), pd.DataFrame(block_rows), manifest


def collect_historical_context(current_metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    direct_root = RESULTS / "directLyap"
    for label, run_id, case_name in HISTORICAL_DIRECT_LYAP:
        path = direct_root / run_id / "comparison_table.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        match = df.loc[df["case_name"] == case_name]
        if match.empty:
            continue
        row = match.iloc[0].to_dict()
        rows.append(
            {
                "label": label,
                "run_dir": str((direct_root / run_id).relative_to(ROOT)),
                "case_name": case_name,
                "n_steps": int(row.get("n_steps", np.nan)),
                "target_mode": row.get("target_mode", ""),
                "lyapunov_mode": row.get("lyapunov_mode", ""),
                "reward_mean_logged": float(row.get("reward_mean", np.nan)),
                "output0_rmse": float(row.get("output0_rmse", np.nan)),
                "output1_rmse": float(row.get("output1_rmse", np.nan)),
                "output_rmse_mean": float(row.get("output_rmse_mean", np.nan)),
                "diagnostic_unsafe_rate": float(row.get("diagnostic_unsafe_rate", np.nan)),
                "actual_intervention_rate": float(row.get("actual_intervention_rate", np.nan)),
                "target_success_rate": float(row.get("target_success_rate", np.nan)),
                "governor_active_rate": float(row.get("governor_active_rate", np.nan)),
            }
        )

    current_lookup = {
        "current Direct LMPC baseline": "Direct LMPC baseline",
        "current OF-MPC baseline": "OF-MPC baseline",
    }
    for label, case in current_lookup.items():
        row = current_metrics.loc[current_metrics["case"] == case].iloc[0]
        rows.append(
            {
                "label": label,
                "run_dir": row["run_dir"],
                "case_name": case,
                "n_steps": int(row["n_steps"]),
                "target_mode": "governed_reference",
                "lyapunov_mode": "current_runner",
                "reward_mean_logged": float(row["reward_mean"]),
                "output0_rmse": float(row["output0_rmse"]),
                "output1_rmse": float(row["output1_rmse"]),
                "output_rmse_mean": float(row["output_rmse_mean"]),
                "diagnostic_unsafe_rate": float(row["diagnostic_unsafe_rate"]),
                "actual_intervention_rate": float(row["actual_intervention_rate"]),
                "target_success_rate": np.nan,
                "governor_active_rate": np.nan,
            }
        )
    return pd.DataFrame(rows)


def _moving_average(values: pd.Series, window: int = 5) -> pd.Series:
    return values.rolling(window=window, min_periods=1, center=True).mean()


def _load_enriched_episode_from_relpath(rel: str) -> pd.DataFrame:
    run_dir = ROOT / rel
    episode = pd.read_csv(run_dir / "episode_table.csv")
    arrays = np.load(run_dir / "arrays.npz", allow_pickle=True)
    err = _output_errors(arrays)
    reward_no_penalty = np.asarray(arrays["reward_no_penalty"], dtype=float)
    rewards = np.asarray(arrays["rewards"], dtype=float)
    if "fallback_penalty" in arrays.files:
        fallback_penalty = np.asarray(arrays["fallback_penalty"], dtype=float)
    else:
        fallback_penalty = np.zeros_like(rewards)
    return _enrich_episode_table(episode, err, reward_no_penalty, rewards, fallback_penalty)


def _style_axes(ax: plt.Axes, ylabel: str | None = None) -> None:
    ax.grid(True, axis="y", alpha=0.25, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if ylabel:
        ax.set_ylabel(ylabel)


def plot_overview(metrics: pd.DataFrame) -> None:
    order = metrics.sort_values("tail50_reward_no_penalty", ascending=True)
    colors = np.where(order["safety_gate"], "#31688e", "#35b779")
    colors = np.where(order["family"].eq("baseline"), "#7b3294", colors)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.3), constrained_layout=True)
    axes[0].barh(order["case"], order["tail50_reward_no_penalty"], color=colors)
    axes[0].set_title("Late training control reward")
    axes[0].set_xlabel("Tail-50 mean reward_no_penalty")
    _style_axes(axes[0])

    rmse_order = metrics.sort_values("tail50_output_rmse_mean", ascending=False)
    colors2 = np.where(rmse_order["safety_gate"], "#31688e", "#35b779")
    colors2 = np.where(rmse_order["family"].eq("baseline"), "#7b3294", colors2)
    axes[1].barh(rmse_order["case"], rmse_order["tail50_output_rmse_mean"], color=colors2)
    axes[1].set_title("Late training physical tracking")
    axes[1].set_xlabel("Tail-50 mean output RMSE")
    _style_axes(axes[1])
    fig.suptitle("Latest 8 disturbance-only runners: late-phase performance")
    fig.savefig(OUT_DIR / "latest_tail_performance_overview.png", dpi=220)
    plt.close(fig)


def plot_safety_activity(metrics: pd.DataFrame) -> None:
    order = metrics.copy()
    x = np.arange(len(order))
    width = 0.28
    fig, ax1 = plt.subplots(figsize=(12.5, 5.4), constrained_layout=True)
    ax1.bar(
        x - width / 2,
        100 * order["actual_gate_intervention_rate"].fillna(0),
        width,
        label="actual gate intervention",
        color="#31688e",
    )
    ax1.bar(x + width / 2, 100 * order["diagnostic_unsafe_rate"].fillna(0), width, label="diagnostic would activate", color="#35b779")
    ax1.set_xticks(x)
    ax1.set_xticklabels(order["case"], rotation=35, ha="right")
    ax1.set_ylabel("Rate (% of steps)")
    ax1.set_title("Actual gate interventions versus monitor-only Lyapunov failures")
    _style_axes(ax1)
    ax1.legend(loc="upper left")

    ax2 = ax1.twinx()
    ax2.plot(x, order["fallback_penalty_mean"], color="#b35806", marker="o", linewidth=2, label="fallback penalty mean")
    ax2.set_ylabel("Mean fallback penalty in logged reward")
    ax2.spines["top"].set_visible(False)
    ax2.legend(loc="upper right")
    fig.savefig(OUT_DIR / "safety_activity_and_penalty.png", dpi=220)
    plt.close(fig)


def plot_episode_trends(
    manifest: dict[str, str],
    metric: str,
    outfile: str,
    ylabel: str,
    ylim: tuple[float, float] | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 5.6), constrained_layout=True)
    palette = {
        "LMPC pretrained + gate": "#31688e",
        "OF-MPC pretrained + gate": "#1f77b4",
        "Cold start + gate": "#5e3c99",
        "LMPC pretrained no gate": "#35b779",
        "OF-MPC pretrained no gate": "#2ca25f",
        "Cold start no gate": "#8c6d31",
        "Direct LMPC baseline": "#7b3294",
        "OF-MPC baseline": "#c2a5cf",
    }
    for label, rel in manifest.items():
        if not (ROOT / rel / "episode_table.csv").exists():
            continue
        df = _load_enriched_episode_from_relpath(rel)
        y = _moving_average(df[metric], 5)
        lw = 2.1 if "baseline" not in label.lower() else 1.5
        alpha = 0.92 if "baseline" not in label.lower() else 0.65
        ax.plot(df["episode"], y, label=label, color=palette.get(label), linewidth=lw, alpha=alpha)
    ax.set_xlabel("Episode")
    ax.set_ylabel(ylabel)
    if ylim is not None:
        ax.set_ylim(*ylim)
    _style_axes(ax)
    ax.legend(ncol=2, fontsize=8.5, frameon=False)
    fig.savefig(OUT_DIR / outfile, dpi=220)
    plt.close(fig)


def plot_last_episode_outputs(manifest: dict[str, str]) -> None:
    selected = [
        "LMPC pretrained + gate",
        "OF-MPC pretrained + gate",
        "OF-MPC pretrained no gate",
        "Cold start no gate",
        "Direct LMPC baseline",
        "OF-MPC baseline",
    ]
    palette = {
        "LMPC pretrained + gate": "#31688e",
        "OF-MPC pretrained + gate": "#1f77b4",
        "OF-MPC pretrained no gate": "#2ca25f",
        "Cold start no gate": "#8c6d31",
        "Direct LMPC baseline": "#7b3294",
        "OF-MPC baseline": "#c2a5cf",
    }
    fig, axes = plt.subplots(2, 1, figsize=(12, 6.4), sharex=True, constrained_layout=True)
    for label in selected:
        rel = manifest[label]
        arrays = np.load(ROOT / rel / "arrays.npz", allow_pickle=True)
        y = np.asarray(arrays["y_system"][1:], dtype=float)
        n = len(y)
        start = max(0, n - 800)
        t = np.arange(n - start)
        if "y_sp_phys_store" in arrays.files:
            sp = np.asarray(arrays["y_sp_phys_store"], dtype=float)
        elif "y_sp_steps" in arrays.files:
            sp = np.asarray(arrays["y_sp_steps"], dtype=float)
        else:
            sp = y * np.nan
        axes[0].plot(t, y[start:, 0], label=label, color=palette.get(label), linewidth=1.8, alpha=0.9)
        axes[1].plot(t, y[start:, 1], label=label, color=palette.get(label), linewidth=1.8, alpha=0.9)
    # Use one representative setpoint trace. All runners use the same schedule.
    first = np.load(ROOT / manifest[selected[0]] / "arrays.npz", allow_pickle=True)
    sp = np.asarray(first["y_sp_phys_store"], dtype=float)
    start = max(0, len(sp) - 800)
    t = np.arange(len(sp) - start)
    axes[0].plot(t, sp[start:, 0], color="black", linestyle="--", linewidth=1.5, label="setpoint")
    axes[1].plot(t, sp[start:, 1], color="black", linestyle="--", linewidth=1.5, label="setpoint")
    axes[0].set_ylabel("Output 0")
    axes[1].set_ylabel("Output 1")
    axes[1].set_xlabel("Last episode step")
    for ax in axes:
        _style_axes(ax)
    axes[0].legend(ncol=3, fontsize=8, frameon=False)
    fig.suptitle("Last episode output tracking in physical units")
    fig.savefig(OUT_DIR / "last_episode_output_tracking.png", dpi=220)
    plt.close(fig)


def plot_historical_context(historical: pd.DataFrame) -> None:
    display = historical.copy()
    display["short_label"] = display["label"].str.replace("2026-", "", regex=False)
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 6.1), constrained_layout=True)
    axes[0].barh(display["short_label"], 100 * display["diagnostic_unsafe_rate"], color="#35b779")
    axes[0].set_xlabel("Diagnostic unsafe rate (%)")
    axes[0].set_title("Monitor activation changed across selector/gate runs")
    _style_axes(axes[0])
    axes[1].barh(display["short_label"], display["output_rmse_mean"], color="#7b3294")
    axes[1].set_xlabel("Output RMSE mean")
    axes[1].set_title("Tracking comparison should use physical RMSE")
    _style_axes(axes[1])
    fig.savefig(OUT_DIR / "historical_selector_monitor_context.png", dpi=220)
    plt.close(fig)


def plot_fallback_breakdown(metrics: pd.DataFrame) -> None:
    gates = metrics.loc[metrics["safety_gate"]].copy()
    if gates.empty:
        return
    x = np.arange(len(gates))
    fig, ax = plt.subplots(figsize=(9.5, 4.8), constrained_layout=True)
    ax.bar(x, gates["fallback_verified_count"], label="verified Direct LMPC fallback", color="#31688e")
    ax.bar(
        x,
        gates["solver_fail_hold_prev_count"],
        bottom=gates["fallback_verified_count"],
        label="hold previous after solver issue",
        color="#b35806",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(gates["case"], rotation=25, ha="right")
    ax.set_ylabel("Count over 240000 steps")
    ax.set_title("What the console 'fallback / hold-prev' count contains")
    _style_axes(ax)
    ax.legend(frameon=False)
    fig.savefig(OUT_DIR / "safety_gate_fallback_breakdown.png", dpi=220)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    metrics, phases, blocks, manifest = collect_latest_metrics()
    historical = collect_historical_context(metrics)

    metrics.to_csv(OUT_DIR / "latest_metrics.csv", index=False)
    phases.to_csv(OUT_DIR / "online_phase_metrics.csv", index=False)
    blocks.to_csv(OUT_DIR / "setpoint_block_metrics.csv", index=False)
    historical.to_csv(OUT_DIR / "historical_selector_context.csv", index=False)
    with (OUT_DIR / "run_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    plot_overview(metrics)
    plot_safety_activity(metrics)
    plot_episode_trends(manifest, "reward_no_penalty_mean", "episode_reward_no_penalty_trends.png", "5-episode smoothed reward_no_penalty")
    plot_episode_trends(
        manifest,
        "reward_no_penalty_mean",
        "episode_reward_no_penalty_trends_zoom.png",
        "5-episode smoothed reward_no_penalty",
        ylim=(-160, 5),
    )
    plot_episode_trends(manifest, "output_rmse_mean", "episode_output_rmse_trends.png", "5-episode smoothed output RMSE")
    plot_episode_trends(
        manifest,
        "output_rmse_mean",
        "episode_output_rmse_trends_zoom.png",
        "5-episode smoothed output RMSE",
        ylim=(0, 1.7),
    )
    plot_last_episode_outputs(manifest)
    plot_historical_context(historical)
    plot_fallback_breakdown(metrics)

    print(f"Wrote analysis tables and figures to {OUT_DIR.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
