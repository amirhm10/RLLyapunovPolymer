"""Analyze the four pretrained online TD3 critic-reset disturbance runs."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "report" / "online_pretrained_critic_reset_analysis_2026-06-12.md"
FIG_DIR = ROOT / "report" / "figures" / "2026-06-12_online_pretrained_critic_reset_analysis"
TABLE_DIR = ROOT / "report" / "tables" / "2026-06-12_online_pretrained_critic_reset_analysis"


CASES = [
    ("LMPC pretrained + gate", "OnlineTD3_LMPCPretrained_SafetyGate", True),
    ("LMPC pretrained no gate", "OnlineTD3_LMPCPretrained_NoSafetyGate", False),
    ("OF-MPC pretrained + gate", "OnlineTD3_OFMPCPretrained_SafetyGate", True),
    ("OF-MPC pretrained no gate", "OnlineTD3_OFMPCPretrained_NoSafetyGate", False),
]

BATCHES = {
    "old_noise": {
        "label": "bounded mixed, no reset, BC std 0.02",
        "runs": {
            "LMPC pretrained + gate": "20260611_000544",
            "LMPC pretrained no gate": "20260611_000541",
            "OF-MPC pretrained + gate": "20260611_000552",
            "OF-MPC pretrained no gate": "20260611_000548",
        },
    },
    "low_noise": {
        "label": "bounded mixed, no reset, BC std 0",
        "runs": {
            "LMPC pretrained + gate": "20260612_011534",
            "LMPC pretrained no gate": "20260612_011530",
            "OF-MPC pretrained + gate": "20260612_011542",
            "OF-MPC pretrained no gate": "20260612_011538",
        },
    },
    "critic_reset": {
        "label": "bounded mixed, critic reset, BC std 1e-4",
        "runs": {
            "LMPC pretrained + gate": "20260612_130549",
            "LMPC pretrained no gate": "20260612_130546",
            "OF-MPC pretrained + gate": "20260612_130557",
            "OF-MPC pretrained no gate": "20260612_130553",
        },
    },
}

CASE_TO_ROOT = {case: root for case, root, _gate in CASES}
CASE_TO_GATE = {case: gate for case, _root, gate in CASES}
CASE_ORDER = [case for case, _root, _gate in CASES]
PHASE_ORDER = ["BC", "handoff", "early full", "mid full", "tail 50"]


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def rel_report(path: Path) -> str:
    return path.relative_to(REPORT.parent).as_posix()


def fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "-"
    try:
        value = float(value)
    except (TypeError, ValueError):
        return str(value)
    if np.isnan(value):
        return "-"
    if abs(value) >= 1000:
        return f"{value:,.0f}"
    if 0 < abs(value) < 1e-4:
        return f"{value:.2e}"
    return f"{value:.{digits}f}"


def md_table(rows: list[dict[str, Any]], columns: list[tuple[str, str]]) -> str:
    lines = [
        "| " + " | ".join(title for _key, title in columns) + " |",
        "| " + " | ".join([":---"] + ["---:"] * (len(columns) - 1)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(key, "")) for key, _title in columns) + " |")
    return "\n".join(lines)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def stop_col(episode: pd.DataFrame) -> str:
    if "step_stop_exclusive" in episode.columns:
        return "step_stop_exclusive"
    if "step_end_exclusive" in episode.columns:
        return "step_end_exclusive"
    raise KeyError("No episode stop column found.")


def safe_mean(values: Any) -> float:
    arr = np.asarray(values, dtype=float).reshape(-1)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return float("nan")
    return float(np.mean(finite))


def safe_max(values: Any) -> float:
    arr = np.asarray(values, dtype=float).reshape(-1)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return float("nan")
    return float(np.max(finite))


def arr_or_fill(arrays: np.lib.npyio.NpzFile, name: str, n: int, fill: float = 0.0) -> np.ndarray:
    if name in arrays.files:
        return np.asarray(arrays[name], dtype=float).reshape(-1)[:n]
    return np.full(n, fill, dtype=float)


def phase_slices(episode: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    n_ep = int(episode["episode"].max())
    return [
        ("BC", episode.loc[episode["episode"].between(1, 20)]),
        ("handoff", episode.loc[episode["episode"].between(21, 25)]),
        ("early full", episode.loc[episode["episode"].between(26, 75)]),
        ("mid full", episode.loc[episode["episode"].between(76, 250)]),
        ("tail 50", episode.loc[episode["episode"].between(max(1, n_ep - 49), n_ep)]),
    ]


def load_run(case: str, batch: str, run_name: str) -> tuple[dict[str, Any], pd.DataFrame, dict[str, Any]]:
    run_dir = ROOT / "results" / CASE_TO_ROOT[case] / run_name
    arrays = np.load(run_dir / "arrays.npz", allow_pickle=True)
    episode = pd.read_csv(run_dir / "episode_table.csv")
    summary = read_json(run_dir / "summary.json")
    run_summary = read_json(run_dir / "run_summary.json")
    cfg = run_summary.get("config", {})
    phase_cfg = cfg.get("training_phase_config", {})
    end = stop_col(episode)
    n = int(episode[end].max())

    reward_no_penalty = arr_or_fill(arrays, "reward_no_penalty", n)
    reward = arr_or_fill(arrays, "rewards", n)
    fallback_penalty = arr_or_fill(arrays, "fallback_penalty", n)
    actual_intervention = arr_or_fill(arrays, "actual_intervention_flags", n)
    diagnostic_unsafe = arr_or_fill(arrays, "diagnostic_unsafe_flags", n)
    accepted = arr_or_fill(arrays, "accepted_flags", n)
    fallback_verified = arr_or_fill(arrays, "fallback_verified_flags", n)
    teacher_gap = arr_or_fill(arrays, "bc_teacher_gap_inf", n, fill=np.nan)
    handoff_gap = arr_or_fill(arrays, "handoff_candidate_gap_inf", n, fill=np.nan)
    executed_gap = arr_or_fill(arrays, "executed_action_gap_inf", n, fill=np.nan)

    u_phys = np.asarray(arrays["u_applied_phys"], dtype=float)
    du = np.diff(u_phys, axis=0) if u_phys.ndim == 2 and u_phys.shape[0] > 1 else np.empty((0,))

    worst = episode.loc[episode["reward_no_penalty_mean"].idxmin()]
    row = {
        "case": case,
        "batch": batch,
        "batch_label": BATCHES[batch]["label"],
        "run_dir": rel(run_dir),
        "n_steps": n,
        "n_episodes": int(episode["episode"].max()),
        "episode_len": int(episode["n_steps"].iloc[0]),
        "initial_agent": Path(str(cfg.get("initial_agent_path") or "")).name,
        "teacher_source": cfg.get("teacher_source"),
        "safety_gate": bool(cfg.get("safety_gate_active", CASE_TO_GATE[case])),
        "target_selector_variant": cfg.get("target_selector_variant"),
        "rho_lyap": cfg.get("rho_lyap"),
        "lyap_eps": cfg.get("lyap_eps"),
        "pretrained_critic_reset": run_summary.get("pretrained_critic_reset", cfg.get("pretrained_critic_reset")),
        "critic_loaded_from_checkpoint": cfg.get("critic_loaded_from_checkpoint"),
        "bc_noise": phase_cfg.get("bc_behavior_noise"),
        "bc_std": phase_cfg.get("bc_exploration_std"),
        "handoff_std_end": phase_cfg.get("handoff_exploration_std_end"),
        "full_std_start": phase_cfg.get("full_rl_exploration_std_start"),
        "mean_reward_no_penalty": safe_mean(reward_no_penalty),
        "mean_training_reward": safe_mean(reward),
        "mean_fallback_penalty": safe_mean(fallback_penalty),
        "summary_output_rmse_mean": float(summary.get("output_rmse_mean", np.nan)),
        "tail50_reward_no_penalty": float(episode.tail(50)["reward_no_penalty_mean"].mean()),
        "tail50_training_reward": float(episode.tail(50)["reward_mean"].mean()),
        "tail50_output_rmse_mean": float(episode.tail(50)["output_rmse_mean"].mean()),
        "actual_intervention_rate": safe_mean(actual_intervention),
        "diagnostic_unsafe_rate": safe_mean(diagnostic_unsafe),
        "accepted_rate": safe_mean(accepted),
        "fallback_verified_rate": safe_mean(fallback_verified),
        "mean_abs_du_phys": safe_mean(np.abs(du)) if du.size else float("nan"),
        "max_abs_du_phys": safe_max(np.abs(du)) if du.size else float("nan"),
        "teacher_gap_mean": safe_mean(teacher_gap),
        "handoff_gap_mean": safe_mean(handoff_gap),
        "executed_gap_mean": safe_mean(executed_gap),
        "worst_episode": int(worst["episode"]),
        "worst_reward_no_penalty": float(worst["reward_no_penalty_mean"]),
        "worst_output_rmse_mean": float(worst["output_rmse_mean"]),
    }

    phase_rows: list[dict[str, Any]] = []
    for phase, part in phase_slices(episode):
        if part.empty:
            continue
        start = int(part["step_start"].iloc[0])
        finish = int(part[end].iloc[-1])
        phase_rows.append(
            {
                "case": case,
                "batch": batch,
                "phase": phase,
                "n_episodes": int(len(part)),
                "reward_no_penalty": float(part["reward_no_penalty_mean"].mean()),
                "training_reward": float(part["reward_mean"].mean()),
                "fallback_penalty": float(part["fallback_penalty_mean"].mean()),
                "output_rmse": float(part["output_rmse_mean"].mean()),
                "actual_intervention_rate": float(part["actual_intervention_rate"].mean()),
                "diagnostic_unsafe_rate": float(part["diagnostic_unsafe_rate"].mean()),
                "teacher_gap": safe_mean(teacher_gap[start:finish]),
                "handoff_gap": safe_mean(handoff_gap[start:finish]),
                "executed_gap": safe_mean(executed_gap[start:finish]),
            }
        )

    meta = {
        "run_dir": run_dir,
        "arrays": arrays,
        "episode": episode,
        "summary": summary,
        "run_summary": run_summary,
        "config": cfg,
    }
    return row, pd.DataFrame(phase_rows), meta


def collect() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    phase_frames: list[pd.DataFrame] = []
    meta: dict[str, dict[str, Any]] = {}
    for case in CASE_ORDER:
        for batch, batch_info in BATCHES.items():
            run_name = batch_info["runs"][case]
            row, phase_df, case_meta = load_run(case, batch, run_name)
            rows.append(row)
            phase_frames.append(phase_df)
            meta[f"{case}::{batch}"] = case_meta

    metrics = pd.DataFrame(rows)
    phases = pd.concat(phase_frames, ignore_index=True)
    deltas: list[dict[str, Any]] = []
    for case in CASE_ORDER:
        reset = metrics.loc[(metrics["case"] == case) & (metrics["batch"] == "critic_reset")].iloc[0]
        for reference in ("low_noise", "old_noise"):
            ref = metrics.loc[(metrics["case"] == case) & (metrics["batch"] == reference)].iloc[0]
            deltas.append(
                {
                    "case": case,
                    "comparison": f"critic_reset - {reference}",
                    "delta_mean_reward_no_penalty": reset["mean_reward_no_penalty"] - ref["mean_reward_no_penalty"],
                    "delta_tail50_reward_no_penalty": reset["tail50_reward_no_penalty"] - ref["tail50_reward_no_penalty"],
                    "delta_tail50_rmse": reset["tail50_output_rmse_mean"] - ref["tail50_output_rmse_mean"],
                    "delta_actual_intervention_pp": 100.0
                    * (reset["actual_intervention_rate"] - ref["actual_intervention_rate"]),
                    "delta_diagnostic_unsafe_pp": 100.0
                    * (reset["diagnostic_unsafe_rate"] - ref["diagnostic_unsafe_rate"]),
                    "delta_mean_abs_du_phys": reset["mean_abs_du_phys"] - ref["mean_abs_du_phys"],
                }
            )
            for phase in PHASE_ORDER:
                r_phase = phases.loc[
                    (phases["case"] == case)
                    & (phases["batch"] == "critic_reset")
                    & (phases["phase"] == phase)
                ].iloc[0]
                ref_phase = phases.loc[
                    (phases["case"] == case)
                    & (phases["batch"] == reference)
                    & (phases["phase"] == phase)
                ].iloc[0]
                deltas[-1][f"delta_{phase}_reward"] = (
                    r_phase["reward_no_penalty"] - ref_phase["reward_no_penalty"]
                )
                deltas[-1][f"delta_{phase}_rmse"] = r_phase["output_rmse"] - ref_phase["output_rmse"]
    return metrics, phases, pd.DataFrame(deltas), meta


def moving_average(values: np.ndarray, window: int = 5) -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size == 0:
        return arr
    return pd.Series(arr).rolling(window=window, min_periods=1).mean().to_numpy()


def make_figures(metrics: pd.DataFrame, phases: pd.DataFrame, deltas: pd.DataFrame, meta: dict[str, dict[str, Any]]) -> dict[str, Path]:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    colors = {"old_noise": "#6f6f6f", "low_noise": "#c94f4f", "critic_reset": "#1f77b4"}
    labels = {"old_noise": "old noise/no reset", "low_noise": "zero BC/no reset", "critic_reset": "critic reset"}

    fig, axes = plt.subplots(2, 2, figsize=(12.2, 7.4), sharex=True)
    for ax, case in zip(axes.flat, CASE_ORDER):
        for batch in ("old_noise", "low_noise", "critic_reset"):
            episode = meta[f"{case}::{batch}"]["episode"]
            y = moving_average(episode["reward_no_penalty_mean"].to_numpy(), window=5)
            ax.plot(episode["episode"], y, label=labels[batch], color=colors[batch], linewidth=1.7)
        ax.axvspan(1, 20, color="#e8f0fb", alpha=0.45, linewidth=0)
        ax.axvspan(21, 25, color="#fff0cc", alpha=0.55, linewidth=0)
        ax.set_title(case)
        ax.set_yscale("symlog", linthresh=10)
        ax.grid(True, alpha=0.25)
    axes[1, 0].set_xlabel("Episode")
    axes[1, 1].set_xlabel("Episode")
    axes[0, 0].set_ylabel("Reward no penalty, 5-episode mean")
    axes[1, 0].set_ylabel("Reward no penalty, 5-episode mean")
    axes[0, 0].legend(loc="lower right", fontsize=8)
    fig.suptitle("Pretrained online TD3 reward traces", y=0.995)
    fig.tight_layout()
    paths["reward_traces"] = FIG_DIR / "pretrained_reset_reward_traces.png"
    fig.savefig(paths["reward_traces"], dpi=180)
    plt.close(fig)

    phase_delta = (
        deltas.loc[deltas["comparison"] == "critic_reset - low_noise"]
        .set_index("case")[[f"delta_{phase}_reward" for phase in PHASE_ORDER]]
        .loc[CASE_ORDER]
    )
    phase_delta.columns = PHASE_ORDER
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    values = phase_delta.to_numpy(dtype=float)
    vmax = np.nanmax(np.abs(values))
    im = ax.imshow(values, aspect="auto", cmap="RdBu", vmin=-vmax, vmax=vmax)
    ax.set_xticks(np.arange(len(PHASE_ORDER)), PHASE_ORDER, rotation=25, ha="right")
    ax.set_yticks(np.arange(len(CASE_ORDER)), CASE_ORDER)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            ax.text(j, i, fmt(values[i, j], 1), ha="center", va="center", fontsize=8)
    ax.set_title("Critic-reset minus no-reset low-noise reward by phase")
    cb = fig.colorbar(im, ax=ax)
    cb.set_label("Delta reward no penalty")
    fig.tight_layout()
    paths["phase_delta"] = FIG_DIR / "pretrained_reset_phase_delta_heatmap.png"
    fig.savefig(paths["phase_delta"], dpi=180)
    plt.close(fig)

    reset = metrics.loc[metrics["batch"] == "critic_reset"].set_index("case").loc[CASE_ORDER]
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.2))
    x = np.arange(len(CASE_ORDER))
    axes[0].bar(x, reset["tail50_reward_no_penalty"], color="#1f77b4")
    axes[0].set_xticks(x, CASE_ORDER, rotation=35, ha="right")
    axes[0].set_ylabel("Tail50 reward no penalty")
    axes[0].grid(True, axis="y", alpha=0.25)
    axes[1].bar(x, reset["tail50_output_rmse_mean"], color="#3f8f6b")
    axes[1].set_xticks(x, CASE_ORDER, rotation=35, ha="right")
    axes[1].set_ylabel("Tail50 output RMSE")
    axes[1].grid(True, axis="y", alpha=0.25)
    fig.suptitle("Tail performance for critic-reset batch")
    fig.tight_layout()
    paths["tail_summary"] = FIG_DIR / "pretrained_reset_tail_summary.png"
    fig.savefig(paths["tail_summary"], dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(12.2, 7.0), sharex=True)
    for ax, case in zip(axes.flat, CASE_ORDER):
        episode = meta[f"{case}::critic_reset"]["episode"]
        local = episode.loc[episode["episode"].between(18, 35)]
        ax.plot(local["episode"], local["reward_no_penalty_mean"], marker="o", label="reward", color="#1f77b4")
        ax2 = ax.twinx()
        ax2.plot(local["episode"], local["output_rmse_mean"], marker="s", label="RMSE", color="#c94f4f")
        ax.axvspan(21, 25, color="#fff0cc", alpha=0.65, linewidth=0)
        ax.set_title(case)
        ax.set_yscale("symlog", linthresh=10)
        ax.grid(True, alpha=0.25)
        if ax in axes[:, 0]:
            ax.set_ylabel("Reward no penalty")
        if ax in axes[:, 1]:
            ax2.set_ylabel("Output RMSE")
    axes[1, 0].set_xlabel("Episode")
    axes[1, 1].set_xlabel("Episode")
    fig.suptitle("Critic-reset handoff zoom", y=0.995)
    fig.tight_layout()
    paths["handoff_zoom"] = FIG_DIR / "pretrained_reset_handoff_zoom.png"
    fig.savefig(paths["handoff_zoom"], dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.2))
    width = 0.28
    x = np.arange(len(CASE_ORDER))
    for idx, batch in enumerate(("old_noise", "low_noise", "critic_reset")):
        part = metrics.loc[metrics["batch"] == batch].set_index("case").loc[CASE_ORDER]
        offset = (idx - 1) * width
        axes[0].bar(x + offset, 100.0 * part["actual_intervention_rate"], width, label=labels[batch], color=colors[batch])
        axes[1].bar(x + offset, 100.0 * part["diagnostic_unsafe_rate"], width, label=labels[batch], color=colors[batch])
    for ax in axes:
        ax.set_xticks(x, CASE_ORDER, rotation=35, ha="right")
        ax.grid(True, axis="y", alpha=0.25)
    axes[0].set_ylabel("Actual safety intervention %")
    axes[1].set_ylabel("Diagnostic would-activate %")
    axes[0].set_title("Safety-gate runners")
    axes[1].set_title("No-gate monitor signal")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    paths["safety"] = FIG_DIR / "pretrained_reset_safety_diagnostics.png"
    fig.savefig(paths["safety"], dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(4, 2, figsize=(12.2, 9.5), sharex=True)
    for row_idx, case in enumerate(CASE_ORDER):
        arrays = meta[f"{case}::critic_reset"]["arrays"]
        y = np.asarray(arrays["y_system"], dtype=float)[:-1]
        sp = np.asarray(arrays["y_sp_phys_store"], dtype=float)
        n = min(y.shape[0], sp.shape[0])
        start = max(0, n - 8000)
        t = np.arange(start, n)
        for j, ylabel in enumerate(("Output 1", "Output 2")):
            ax = axes[row_idx, j]
            ax.plot(t, y[start:n, j], linewidth=1.0, label="plant")
            ax.plot(t, sp[start:n, j], linestyle="--", linewidth=1.2, label="setpoint")
            ax.set_title(f"{case} - {ylabel}")
            ax.grid(True, alpha=0.25)
            if row_idx == 0 and j == 0:
                ax.legend(fontsize=8)
    axes[-1, 0].set_xlabel("Step")
    axes[-1, 1].set_xlabel("Step")
    fig.suptitle("Critic-reset tail tracking snapshots, final 10 episodes", y=0.995)
    fig.tight_layout()
    paths["tracking"] = FIG_DIR / "pretrained_reset_tail_tracking.png"
    fig.savefig(paths["tracking"], dpi=180)
    plt.close(fig)

    return paths


def make_tables(metrics: pd.DataFrame, phases: pd.DataFrame, deltas: pd.DataFrame) -> dict[str, Path]:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    paths = {
        "metrics": TABLE_DIR / "pretrained_reset_metrics.csv",
        "phases": TABLE_DIR / "pretrained_reset_phase_metrics.csv",
        "deltas": TABLE_DIR / "pretrained_reset_deltas.csv",
    }
    metrics.to_csv(paths["metrics"], index=False)
    phases.to_csv(paths["phases"], index=False)
    deltas.to_csv(paths["deltas"], index=False)
    return paths


def report_text(metrics: pd.DataFrame, phases: pd.DataFrame, deltas: pd.DataFrame, figs: dict[str, Path], tables: dict[str, Path]) -> str:
    reset = metrics.loc[metrics["batch"] == "critic_reset"].set_index("case").loc[CASE_ORDER]
    low_delta = deltas.loc[deltas["comparison"] == "critic_reset - low_noise"].set_index("case").loc[CASE_ORDER]
    old_delta = deltas.loc[deltas["comparison"] == "critic_reset - old_noise"].set_index("case").loc[CASE_ORDER]

    data_rows = []
    for case in CASE_ORDER:
        r = reset.loc[case]
        data_rows.append(
            {
                "Case": case,
                "Run": str(r["run_dir"]).split("/")[-1],
                "Agent": r["initial_agent"],
                "Teacher": r["teacher_source"],
                "Gate": "active" if r["safety_gate"] else "monitor only",
            }
        )

    perf_rows = []
    for case in CASE_ORDER:
        r = reset.loc[case]
        perf_rows.append(
            {
                "Case": case,
                "Mean Rnp": fmt(r["mean_reward_no_penalty"]),
                "Tail Rnp": fmt(r["tail50_reward_no_penalty"]),
                "Tail RMSE": fmt(r["tail50_output_rmse_mean"]),
                "Gate %": fmt(100.0 * r["actual_intervention_rate"]),
                "Diag %": fmt(100.0 * r["diagnostic_unsafe_rate"]),
                "Mean abs dU": fmt(r["mean_abs_du_phys"]),
            }
        )

    delta_low_rows = []
    for case in CASE_ORDER:
        d = low_delta.loc[case]
        delta_low_rows.append(
            {
                "Case": case,
                "Mean Rnp": fmt(d["delta_mean_reward_no_penalty"]),
                "Early": fmt(d["delta_early full_reward"]),
                "Tail Rnp": fmt(d["delta_tail50_reward_no_penalty"]),
                "Tail RMSE": fmt(d["delta_tail50_rmse"]),
                "Gate pp": fmt(d["delta_actual_intervention_pp"]),
                "Diag pp": fmt(d["delta_diagnostic_unsafe_pp"]),
            }
        )

    phase_rows = []
    reset_phase = phases.loc[phases["batch"] == "critic_reset"]
    of_gate_handoff = reset_phase.loc[
        (reset_phase["case"] == "OF-MPC pretrained + gate") & (reset_phase["phase"] == "handoff"),
        "reward_no_penalty",
    ].iloc[0]
    for case in CASE_ORDER:
        row = {"Case": case}
        for phase in ("BC", "handoff", "early full", "tail 50"):
            value = reset_phase.loc[
                (reset_phase["case"] == case) & (reset_phase["phase"] == phase),
                "reward_no_penalty",
            ].iloc[0]
            row[phase] = fmt(value)
        phase_rows.append(row)

    handoff_rows = []
    for case in CASE_ORDER:
        r = reset.loc[case]
        phase = "handoff" if 21 <= int(r["worst_episode"]) <= 25 else "other"
        handoff_rows.append(
            {
                "Case": case,
                "Worst ep": int(r["worst_episode"]),
                "Phase": phase,
                "Reward": fmt(r["worst_reward_no_penalty"]),
                "RMSE": fmt(r["worst_output_rmse_mean"]),
            }
        )

    old_delta_rows = []
    for case in CASE_ORDER:
        d = old_delta.loc[case]
        old_delta_rows.append(
            {
                "Case": case,
                "Mean Rnp": fmt(d["delta_mean_reward_no_penalty"]),
                "Tail Rnp": fmt(d["delta_tail50_reward_no_penalty"]),
                "Tail RMSE": fmt(d["delta_tail50_rmse"]),
                "Gate pp": fmt(d["delta_actual_intervention_pp"]),
                "Diag pp": fmt(d["delta_diagnostic_unsafe_pp"]),
            }
        )

    config = reset.iloc[0]
    lmpc_gate = reset.loc["LMPC pretrained + gate"]
    lmpc_nogate = reset.loc["LMPC pretrained no gate"]
    of_gate = reset.loc["OF-MPC pretrained + gate"]
    of_nogate = reset.loc["OF-MPC pretrained no gate"]

    return f"""# Pretrained Online TD3 Critic-Reset Batch Analysis

Date: 2026-06-12

## Question

Four pretrained online TD3 disturbance runners were rerun after two targeted
changes: the online BC phase for pretrained agents now uses tiny teacher-action
noise, and the pretrained critic is reset before online training. This report
analyzes whether the new runs support keeping the pretrained actor while
discarding the offline critic.

Short answer: the critic reset is strongly supported for the LMPC-pretrained
runs and it removes the catastrophic early-full-RL collapse in all four
pretrained cases. The remaining weakness is not the full-RL phase; it is a
localized handoff transient in the OF-MPC-pretrained cases, especially episode
23. The evidence therefore points toward keeping critic reset, but making the
handoff more conservative when the critic is fresh.

## Paper-Consistency Frame

The framing follows the practical process-control style of Hamedi et al. (2026):
MPC and OF-MPC remain the engineering reference points, RL is introduced as a
policy-improvement mechanism, and unsafe online exploration is treated as a
deployment limitation rather than a generic machine-learning inconvenience. It
also follows the close-comparator logic of Khodaverdian et al. (2025): the RL
actor proposes a control action, while a Lyapunov-based supervisory layer has
final authority when the safety gate is active. The distinction here is that the
certificate is computed for the identified output-disturbance model and a
bounded Direct LMPC target, so the result should be stated as model-based
practical one-step contraction rather than global nonlinear asymptotic
stability to the raw setpoint.

## Data Used

| Case | Run | Agent | Teacher | Gate |
| :--- | ---: | ---: | ---: | ---: |
""" + "\n".join(
        f"| {row['Case']} | {row['Run']} | {row['Agent']} | {row['Teacher']} | {row['Gate']} |"
        for row in data_rows
    ) + f"""

All four current runs used:

- disturbance-only plant mode, 300 episodes, and 400-step setpoint blocks;
- bounded mixed Direct LMPC target selector, `bounded_mixed_u0p1_x0p1`;
- `rho_lyap={fmt(config['rho_lyap'])}`, `lyap_eps={fmt(config['lyap_eps'])}`;
- pretrained online BC std `0.0001` with Gaussian teacher-action perturbation;
- pretrained actor loaded from checkpoint and critic reset before online TD3.

The comparison batch `low_noise` is the immediately preceding pretrained
low-noise batch with no critic reset and BC std `0`. The older `old_noise` batch
is retained as context because it used moderate BC exploration (`0.02`) and no
critic reset. The LMPC comparison to `old_noise` is partly confounded because
the LMPC checkpoint changed between the old and current batches.

## Method Reconstruction

The controller uses the identified output-disturbance state-space model in
scaled deviation coordinates,

$$
\\begin{{aligned}}
\\hat z_{{k+1}} &= A_a \\hat z_k + B_a u_k + L\\left(y_k-C_a\\hat z_k\\right), \\\\
\\hat y_k &= C_a\\hat z_k,
\\end{{aligned}}
$$

where $\\hat z_k=[\\hat x_k^\\top,\\hat d_k^\\top]^\\top$ is the estimated
augmented state. The TD3 actor observes

$$
s_k = \\left[
\\mathrm{{scale}}_{{[-1,1]}}(\\hat z_k)^\\top,
\\mathrm{{scale}}_{{[-1,1]}}(y_{{sp,k}})^\\top,
\\mathrm{{scale}}_{{[-1,1]}}(u_{{k-1}})^\\top
\\right]^\\top .
$$

The actor output $a_k\\in[-1,1]^{{n_u}}$ is mapped to the admissible input
deviation interval as

$$
u_k^\\pi = u_{{\\min}} + \\frac{{a_k+1}}{{2}}(u_{{\\max}}-u_{{\\min}}).
$$

The online TD3 critic update uses the standard clipped double-Q target,

$$
\\begin{{aligned}}
\\tilde a_{{k+1}} &=
\\mathrm{{clip}}\\left(
\\pi_{{\\bar\\theta}}(s_{{k+1}})+\\epsilon,
-1,1
\\right), \\\\
y_k^Q &= r_k + \\gamma(1-d_k)
\\min_i Q_{{\\bar\\phi_i}}(s_{{k+1}},\\tilde a_{{k+1}}),
\\end{{aligned}}
$$

with $\\gamma=0.99$ and policy delay 2. In BC, the critic receives executed
online transitions, while the actor is supervised toward the clean teacher
action:

$$
\\min_\\theta \\; \\mathbb{{E}}_{{(s,a_T)\\sim\\mathcal D_{{BC}}}}
\\left\\|\\pi_\\theta(s)-a_T\\right\\|_2^2 .
$$

For the current batch, the pretrained initialization is

$$
\\theta_0 \\leftarrow \\theta_{{\\mathrm{{ckpt}}}}, \\qquad
\\phi_0 \\sim \\mathrm{{Init}}, \\qquad
\\bar\\phi_0 \\leftarrow \\phi_0,
$$

so the actor prior is retained but the offline critic is discarded.

When the safety gate is active, a bounded Direct LMPC target
$(x_s,u_s,y_s)$ is selected with visible regularization weights
$w_u=w_x=0.1$. A candidate action is accepted only if the predicted first-step
Lyapunov value satisfies

$$
V(\\hat x_{{k+1}}^{{cand}}-x_s)
\\le
\\rho V(\\hat x_k-x_s)+\\epsilon,
\\qquad
\\rho=0.99,\\quad \\epsilon=10^{{-3}}.
$$

If the inequality fails, Direct LMPC supplies the applied fallback action. In
the no-gate runners, the same Direct LMPC computation is retained only as a
monitor: the TD3 action is applied, actual intervention remains zero, and the
diagnostic unsafe count estimates how often the safety layer would have acted.

## Algorithm

1. Load the pretrained TD3 checkpoint and infer actor/critic layer sizes.
2. Copy the pretrained actor into the online agent.
3. Reset the critic and target critic; reinitialize the critic optimizer.
4. For BC episodes, execute the teacher action plus $\\mathcal N(0,10^{{-4}})$
   noise, store the executed transition for critic learning, and store the clean
   teacher action for actor BC.
5. For handoff episodes, blend the teacher input and policy candidate with a
   linearly decreasing teacher weight.
6. For full RL episodes, execute the TD3 candidate subject to the selected
   gate/no-gate logic.
7. In safety-gate cases, apply the Direct LMPC fallback only when the candidate
   fails the model-based contraction test.
8. In no-gate cases, apply the TD3 action directly and log the Direct LMPC
   diagnostic would-activate signal.

## Current Batch Performance

{md_table(perf_rows, [
        ("Case", "Case"),
        ("Mean Rnp", "Mean Rnp"),
        ("Tail Rnp", "Tail Rnp"),
        ("Tail RMSE", "Tail RMSE"),
        ("Gate %", "Gate %"),
        ("Diag %", "Diag %"),
        ("Mean abs dU", "Mean abs dU"),
    ])}

![Tail summary]({rel_report(figs['tail_summary'])})

The LMPC-pretrained cases are now the strongest cases in the current batch.
The LMPC no-gate run gives the best tail reward (`{fmt(lmpc_nogate['tail50_reward_no_penalty'])}`)
and lowest tail RMSE (`{fmt(lmpc_nogate['tail50_output_rmse_mean'])}`). The LMPC safety-gate
case is also stable in the learning sense, although its tail reward is slightly
worse than its no-gate counterpart and it applies actual gate interventions in
about `{fmt(100.0*lmpc_gate['actual_intervention_rate'])}%` of steps.

The OF-MPC-pretrained cases should not be interpreted from their mean reward
alone. Their tail behavior recovers to approximately the old bounded-mixed
level, but their full-run mean is dominated by the handoff outlier discussed
below.

![Reward traces]({rel_report(figs['reward_traces'])})

## Critic Reset Versus No-Reset Low-Noise Batch

Positive reward deltas are better. Negative RMSE deltas are better.

{md_table(delta_low_rows, [
        ("Case", "Case"),
        ("Mean Rnp", "Mean Rnp"),
        ("Early", "Early"),
        ("Tail Rnp", "Tail Rnp"),
        ("Tail RMSE", "Tail RMSE"),
        ("Gate pp", "Gate pp"),
        ("Diag pp", "Diag pp"),
    ])}

![Phase delta]({rel_report(figs['phase_delta'])})

The central effect is clear: critic reset removes the early-full-RL collapse.
Relative to the no-reset low-noise batch, early full-RL reward improves by
`{fmt(low_delta.loc['LMPC pretrained + gate']['delta_early full_reward'])}` for LMPC + gate,
`{fmt(low_delta.loc['LMPC pretrained no gate']['delta_early full_reward'])}` for LMPC no gate,
`{fmt(low_delta.loc['OF-MPC pretrained + gate']['delta_early full_reward'])}` for OF-MPC + gate,
and `{fmt(low_delta.loc['OF-MPC pretrained no gate']['delta_early full_reward'])}` for OF-MPC no gate.

This supports the hypothesis that the offline critic was mismatched to the
online reward scale and closed-loop state-action distribution. The actor
pretraining remains useful, but the Q-function trained on offline synthetic
labels is not a reliable online initialization.

## Phase Diagnosis

{md_table(phase_rows, [
        ("Case", "Case"),
        ("BC", "BC"),
        ("handoff", "Handoff"),
        ("early full", "Early"),
        ("tail 50", "Tail50"),
    ])}

The BC phase is almost identical across the four current runs because the
teacher action dominates and the added action noise is tiny. The important
difference appears at handoff. For the OF-MPC-pretrained safety-gate run, the
handoff average is `{fmt(of_gate_handoff)}` and the full-run mean is
`{fmt(of_gate['mean_reward_no_penalty'])}` because episode 23 has a very large
tracking failure. The same pattern appears, though less severely, in the OF-MPC
no-gate run.

{md_table(handoff_rows, [
        ("Case", "Case"),
        ("Worst ep", "Worst ep"),
        ("Phase", "Phase"),
        ("Reward", "Reward"),
        ("RMSE", "RMSE"),
    ])}

![Handoff zoom]({rel_report(figs['handoff_zoom'])})

This is a mechanism-level result. Critic reset fixes the Q-scale mismatch, but
the handoff currently begins full TD3 actor updates while the critic is still
fresh and while the behavior action is a teacher-policy blend. That is a fragile
combination: the critic is being calibrated online, the actor begins to trust
its gradients, and the blend changes the executed distribution over only five
episodes. The OF-MPC-pretrained actor is most sensitive to this transition.

## Safety And Diagnostic Behavior

![Safety diagnostics]({rel_report(figs['safety'])})

Safety-gate interventions remain low in the current batch. This should be
interpreted carefully: low intervention frequency does not mean the gate is
irrelevant. It means the tested candidates usually satisfied the model-based
one-step contraction condition. The no-gate monitor signal is more revealing
for comparing policy aggressiveness. Diagnostic would-activate rates increase
in both no-gate reset runs, reaching `{fmt(100.0*lmpc_nogate['diagnostic_unsafe_rate'])}%` for
LMPC no gate and `{fmt(100.0*of_nogate['diagnostic_unsafe_rate'])}%` for OF-MPC no gate. Thus, the reset
policy is better for reward/tracking in the LMPC no-gate case, but it also
visits more actions that the Direct LMPC monitor would reject.

The safety gate is not a raw tracking optimizer. It certifies a contraction
condition around the bounded Direct LMPC target. During the OF-MPC handoff
spike, the gate can accept or only lightly intervene because the model-based
contraction condition is satisfied, even though raw setpoint tracking reward
becomes poor. This is exactly why reports and the paper should separate
practical safety certification from tracking-performance claims.

## Tail Tracking

![Tail tracking]({rel_report(figs['tracking'])})

The tail plots show that the critic-reset batch recovers after the handoff
transient. Tail differences are modest compared with the early and handoff
differences. This supports treating the OF-MPC issue as a transition-design
problem rather than a final-policy failure.

## Context Against The Older Moderate-Noise Batch

{md_table(old_delta_rows, [
        ("Case", "Case"),
        ("Mean Rnp", "Mean Rnp"),
        ("Tail Rnp", "Tail Rnp"),
        ("Tail RMSE", "Tail RMSE"),
        ("Gate pp", "Gate pp"),
        ("Diag pp", "Diag pp"),
    ])}

Against the older moderate-noise batch, the strongest clean conclusion is for
OF-MPC because the checkpoint is the same across batches. The reset version
recovers OF-MPC tail reward almost exactly, but still has a worse full-run mean
because of the handoff spike. For LMPC, comparison to the old batch is less
clean because the LMPC checkpoint changed; the reset batch should primarily be
compared to the immediately preceding low-noise LMPC batch that used the same
bounded-mixed checkpoint.

## Interpretation

The current evidence supports three conclusions.

First, critic reset is a useful correction. It keeps the useful policy prior
from pretraining while forcing the Q-functions to learn the online shaped reward,
the safety/fallback penalty structure, and the closed-loop rollout
distribution.

Second, tiny BC exploration is not the source of the remaining problem. The BC
episodes are well behaved. The failure is concentrated at handoff, after the
system moves from pure teacher-executed behavior into teacher-policy blending
and full TD3 actor updates.

Third, the safety-gate and no-gate comparisons should be read as deployment
mechanism comparisons, not only tracking comparisons. In the no-gate cases,
actual interventions are zero by design, but the Direct LMPC monitor indicates
whether the same actions would have been rejected under the gate.

## Recommended Next Experiment

Keep critic reset for pretrained online runs, but change the handoff update
logic before the next full batch:

1. Keep the pretrained actor and reset critic.
2. Keep pretrained BC std at `1e-4` or test `1e-3` as a small local-coverage
   ablation.
3. During handoff, collect blended teacher-policy transitions but freeze TD3
   actor-gradient updates; use critic-only updates plus optional actor BC.
4. Start full TD3 actor updates only after handoff, or after an additional
   short critic-calibration window.
5. Extend handoff from 5 episodes to 10-20 episodes for OF-MPC-pretrained runs,
   or cap the policy weight increase per episode.

This next experiment directly targets the observed mechanism. Simply changing
BC noise again would miss the main issue: the critic reset helped, but the fresh
critic needs a more conservative transition before actor-gradient updates are
allowed to dominate.

## Report Artifacts

- Metrics table: `{rel(tables['metrics'])}`
- Phase table: `{rel(tables['phases'])}`
- Delta table: `{rel(tables['deltas'])}`
- Figures: `{rel(FIG_DIR)}`

## Limitations

- These are single-seed training runs, not seed-averaged final evidence.
- The reported reward comparison uses `reward_no_penalty` for control
  performance; training reward remains relevant for learning but is not a fair
  cross-method control metric.
- Frozen saved-agent evaluation is still needed before claiming final controller
  performance.
- The LMPC old-batch comparison is checkpoint-confounded; the reset-vs-low
  comparison is the cleaner LMPC critic-reset test.
"""


def main() -> None:
    metrics, phases, deltas, meta = collect()
    tables = make_tables(metrics, phases, deltas)
    figs = make_figures(metrics, phases, deltas, meta)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(report_text(metrics, phases, deltas, figs, tables), encoding="utf-8")
    print(f"Wrote {rel(REPORT)}")
    print(f"Wrote {rel(FIG_DIR)}")
    print(f"Wrote {rel(TABLE_DIR)}")


if __name__ == "__main__":
    main()
