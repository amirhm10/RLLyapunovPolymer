"""Analyze the low-noise online TD3 runner batch against the prior batch."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "report" / "online_low_noise_regression_analysis_2026-06-12.md"
FIG_DIR = ROOT / "report" / "figures" / "2026-06-12_online_low_noise_regression_analysis"
TABLE_DIR = ROOT / "report" / "tables" / "2026-06-12_online_low_noise_regression_analysis"


CASES = [
    ("LMPC pretrained + gate", "OnlineTD3_LMPCPretrained_SafetyGate"),
    ("OF-MPC pretrained + gate", "OnlineTD3_OFMPCPretrained_SafetyGate"),
    ("LMPC pretrained no gate", "OnlineTD3_LMPCPretrained_NoSafetyGate"),
    ("OF-MPC pretrained no gate", "OnlineTD3_OFMPCPretrained_NoSafetyGate"),
    ("Cold start + gate", "OnlineTD3_ColdStart_SafetyGate"),
    ("Cold start no gate", "OnlineTD3_ColdStart_NoSafetyGate"),
]

OLD_NOISE_RUNS = {
    "LMPC pretrained + gate": "20260611_000544",
    "OF-MPC pretrained + gate": "20260611_000552",
    "LMPC pretrained no gate": "20260611_000541",
    "OF-MPC pretrained no gate": "20260611_000548",
    "Cold start + gate": "20260611_000537",
    "Cold start no gate": "20260611_000534",
}

LOW_NOISE_RUNS = {
    "LMPC pretrained + gate": "20260612_011534",
    "OF-MPC pretrained + gate": "20260612_011542",
    "LMPC pretrained no gate": "20260612_011530",
    "OF-MPC pretrained no gate": "20260612_011538",
    "Cold start + gate": "20260612_011526",
    "Cold start no gate": "20260612_011522",
}

CASE_ORDER = [name for name, _root in CASES]


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
        return f"{value:.3e}"
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


def end_column(episode: pd.DataFrame) -> str:
    if "step_stop_exclusive" in episode.columns:
        return "step_stop_exclusive"
    if "step_end_exclusive" in episode.columns:
        return "step_end_exclusive"
    raise KeyError("No episode stop column found.")


def arr_or_zero(arrays: np.lib.npyio.NpzFile, name: str, n: int) -> np.ndarray:
    if name in arrays.files:
        return np.asarray(arrays[name], dtype=float).reshape(-1)[:n]
    return np.zeros(n, dtype=float)


def arr_or_nan(arrays: np.lib.npyio.NpzFile, name: str, n: int) -> np.ndarray:
    if name in arrays.files:
        return np.asarray(arrays[name], dtype=float).reshape(-1)[:n]
    return np.full(n, np.nan, dtype=float)


def safe_mean(values: Any) -> float:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0 or np.all(np.isnan(arr)):
        return float("nan")
    return float(np.nanmean(arr))


def phase_slices(episode: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    n_ep = int(episode["episode"].max())
    return [
        ("BC", episode.loc[episode["episode"].between(1, 20)]),
        ("handoff", episode.loc[episode["episode"].between(21, 25)]),
        ("early full", episode.loc[episode["episode"].between(26, 75)]),
        ("mid full", episode.loc[episode["episode"].between(76, 250)]),
        ("tail 50", episode.loc[episode["episode"].between(max(1, n_ep - 49), n_ep)]),
    ]


def load_run(case: str, root_name: str, run_name: str, batch: str) -> tuple[dict[str, Any], pd.DataFrame, dict[str, Any]]:
    run_dir = ROOT / "results" / root_name / run_name
    arrays = np.load(run_dir / "arrays.npz", allow_pickle=True)
    episode = pd.read_csv(run_dir / "episode_table.csv")
    summary = read_json(run_dir / "summary.json")
    run_summary = read_json(run_dir / "run_summary.json")
    cfg = run_summary.get("config", {})
    phase_cfg = cfg.get("training_phase_config", {})
    stop_col = end_column(episode)
    n = int(episode[stop_col].max())

    rewards = arr_or_zero(arrays, "rewards", n)
    reward_no_penalty = arr_or_zero(arrays, "reward_no_penalty", n)
    fallback_penalty = arr_or_zero(arrays, "fallback_penalty", n)
    actual_intervention = arr_or_zero(arrays, "actual_intervention_flags", n)
    diagnostic_unsafe = arr_or_zero(arrays, "diagnostic_unsafe_flags", n)
    accepted = arr_or_zero(arrays, "accepted_flags", n)
    fallback_verified = arr_or_zero(arrays, "fallback_verified_flags", n)
    teacher_gap = arr_or_nan(arrays, "bc_teacher_gap_inf", n)
    handoff_gap = arr_or_nan(arrays, "handoff_candidate_gap_inf", n)
    executed_gap = arr_or_nan(arrays, "executed_action_gap_inf", n)

    u = np.asarray(arrays["u_applied_phys"], dtype=float)
    if u.ndim == 2 and u.shape[0] > 1:
        mean_abs_du = float(np.mean(np.abs(np.diff(u, axis=0))))
        max_abs_du = float(np.max(np.abs(np.diff(u, axis=0))))
    else:
        mean_abs_du = float("nan")
        max_abs_du = float("nan")

    row = {
        "case": case,
        "batch": batch,
        "run_dir": rel(run_dir),
        "n_steps": n,
        "initial_agent": Path(str(cfg.get("initial_agent_path") or "")).name,
        "bc_noise": phase_cfg.get("bc_behavior_noise"),
        "bc_std": phase_cfg.get("bc_exploration_std"),
        "handoff_std_end": phase_cfg.get("handoff_exploration_std_end"),
        "full_std_start": phase_cfg.get("full_rl_exploration_std_start"),
        "reward_mean": safe_mean(rewards),
        "reward_no_penalty_mean": safe_mean(reward_no_penalty),
        "fallback_penalty_mean": safe_mean(fallback_penalty),
        "output_rmse_mean": float(summary.get("output_rmse_mean", np.nan)),
        "tail50_reward_no_penalty": float(episode.tail(50)["reward_no_penalty_mean"].mean()),
        "tail50_reward": float(episode.tail(50)["reward_mean"].mean()),
        "tail50_fallback_penalty": float(episode.tail(50)["fallback_penalty_mean"].mean()),
        "tail50_output_rmse_mean": float(episode.tail(50)["output_rmse_mean"].mean()),
        "actual_intervention_rate": safe_mean(actual_intervention),
        "diagnostic_unsafe_rate": safe_mean(diagnostic_unsafe),
        "accepted_rate": safe_mean(accepted),
        "fallback_verified_rate": safe_mean(fallback_verified),
        "mean_abs_du_phys": mean_abs_du,
        "max_abs_du_phys": max_abs_du,
        "teacher_gap_mean": safe_mean(teacher_gap),
        "teacher_gap_tail50": safe_mean(
            teacher_gap[int(episode.tail(50)["step_start"].iloc[0]) : int(episode.tail(50)[stop_col].iloc[-1])]
        ),
        "handoff_gap_mean": safe_mean(handoff_gap),
        "executed_gap_mean": safe_mean(executed_gap),
    }

    phase_rows: list[dict[str, Any]] = []
    for phase, part in phase_slices(episode):
        if part.empty:
            continue
        start = int(part["step_start"].iloc[0])
        stop = int(part[stop_col].iloc[-1])
        phase_rows.append(
            {
                "case": case,
                "batch": batch,
                "phase": phase,
                "n_episodes": int(len(part)),
                "reward_no_penalty": float(part["reward_no_penalty_mean"].mean()),
                "reward": float(part["reward_mean"].mean()),
                "fallback_penalty": float(part["fallback_penalty_mean"].mean()),
                "output_rmse": float(part["output_rmse_mean"].mean()),
                "actual_intervention_rate": float(part["actual_intervention_rate"].mean()),
                "diagnostic_unsafe_rate": float(part["diagnostic_unsafe_rate"].mean()),
                "teacher_gap": safe_mean(teacher_gap[start:stop]),
                "handoff_gap": safe_mean(handoff_gap[start:stop]),
                "executed_gap": safe_mean(executed_gap[start:stop]),
            }
        )

    meta = {
        "episode": episode,
        "arrays": arrays,
        "config": cfg,
        "run_dir": run_dir,
    }
    return row, pd.DataFrame(phase_rows), meta


def collect() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    phase_frames: list[pd.DataFrame] = []
    meta: dict[str, dict[str, Any]] = {}
    roots = dict(CASES)
    for case in CASE_ORDER:
        for batch, run_map in (("old_noise", OLD_NOISE_RUNS), ("low_noise", LOW_NOISE_RUNS)):
            row, phases, case_meta = load_run(case, roots[case], run_map[case], batch)
            rows.append(row)
            phase_frames.append(phases)
            meta[f"{case}::{batch}"] = case_meta

    metrics = pd.DataFrame(rows)
    phases = pd.concat(phase_frames, ignore_index=True)
    old = metrics.loc[metrics["batch"] == "old_noise"].set_index("case")
    new = metrics.loc[metrics["batch"] == "low_noise"].set_index("case")
    delta_rows = []
    for case in CASE_ORDER:
        o = old.loc[case]
        n = new.loc[case]
        delta_rows.append(
            {
                "case": case,
                "old_run": o["run_dir"],
                "low_run": n["run_dir"],
                "old_agent": o["initial_agent"],
                "low_agent": n["initial_agent"],
                "delta_reward_no_penalty": n["reward_no_penalty_mean"] - o["reward_no_penalty_mean"],
                "delta_tail50_reward_no_penalty": n["tail50_reward_no_penalty"] - o["tail50_reward_no_penalty"],
                "delta_tail50_rmse": n["tail50_output_rmse_mean"] - o["tail50_output_rmse_mean"],
                "delta_fallback_penalty": n["fallback_penalty_mean"] - o["fallback_penalty_mean"],
                "delta_actual_intervention_pct": 100.0 * (n["actual_intervention_rate"] - o["actual_intervention_rate"]),
                "delta_diag_unsafe_pct": 100.0 * (n["diagnostic_unsafe_rate"] - o["diagnostic_unsafe_rate"]),
                "delta_mean_abs_du_phys": n["mean_abs_du_phys"] - o["mean_abs_du_phys"],
            }
        )
    deltas = pd.DataFrame(delta_rows)
    return metrics, phases, deltas, meta


def plot_tail_delta(deltas: pd.DataFrame) -> Path:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    order = deltas.copy()
    order["case"] = pd.Categorical(order["case"], CASE_ORDER, ordered=True)
    order = order.sort_values("case")
    x = np.arange(len(order))
    fig, axes = plt.subplots(2, 1, figsize=(11.5, 7.2), constrained_layout=True)
    axes[0].bar(x, order["delta_tail50_reward_no_penalty"], color="#b2182b")
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].set_ylabel("Delta tail-50 reward_no_penalty")
    axes[0].set_title("Low-noise minus old-noise late reward")
    axes[0].grid(axis="y", alpha=0.25)
    axes[1].bar(x, order["delta_tail50_rmse"], color="#2166ac")
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_ylabel("Delta tail-50 RMSE")
    axes[1].set_title("Low-noise minus old-noise late tracking")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].set_xticks(x, order["case"], rotation=25, ha="right")
    axes[0].set_xticks(x, order["case"], rotation=25, ha="right")
    path = FIG_DIR / "low_noise_tail_delta.png"
    fig.savefig(path, dpi=190)
    plt.close(fig)
    return path


def plot_phase_reward(phases: pd.DataFrame) -> Path:
    selected = phases.loc[phases["phase"].isin(["BC", "handoff", "early full", "tail 50"])].copy()
    selected["case"] = pd.Categorical(selected["case"], CASE_ORDER, ordered=True)
    selected["phase"] = pd.Categorical(selected["phase"], ["BC", "handoff", "early full", "tail 50"], ordered=True)
    fig, axes = plt.subplots(3, 2, figsize=(12.2, 10.0), sharey=False, constrained_layout=True)
    axes = axes.ravel()
    for ax, case in zip(axes, CASE_ORDER):
        part = selected.loc[selected["case"].eq(case)].sort_values(["phase", "batch"])
        phases_order = ["BC", "handoff", "early full", "tail 50"]
        old_vals = [part.loc[(part["phase"].eq(p)) & part["batch"].eq("old_noise"), "reward_no_penalty"].mean() for p in phases_order]
        new_vals = [part.loc[(part["phase"].eq(p)) & part["batch"].eq("low_noise"), "reward_no_penalty"].mean() for p in phases_order]
        x = np.arange(len(phases_order))
        ax.plot(x, old_vals, marker="o", label="old-noise", color="#1b7837")
        ax.plot(x, new_vals, marker="o", label="low-noise", color="#b2182b")
        ax.set_title(case)
        ax.set_xticks(x, phases_order, rotation=20)
        ax.grid(axis="y", alpha=0.25)
    axes[0].legend(frameon=False)
    fig.suptitle("Phase reward_no_penalty: old-noise vs low-noise")
    path = FIG_DIR / "low_noise_phase_reward.png"
    fig.savefig(path, dpi=190)
    plt.close(fig)
    return path


def plot_episode_traces(meta: dict[str, dict[str, Any]]) -> Path:
    cases = [
        "OF-MPC pretrained + gate",
        "OF-MPC pretrained no gate",
        "Cold start + gate",
        "Cold start no gate",
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12.2, 7.5), sharex=True, constrained_layout=True)
    axes = axes.ravel()
    for ax, case in zip(axes, cases):
        for batch, color in (("old_noise", "#1b7837"), ("low_noise", "#b2182b")):
            ep = meta[f"{case}::{batch}"]["episode"]
            smooth = ep["reward_no_penalty_mean"].rolling(5, min_periods=1, center=True).mean()
            ax.plot(ep["episode"], smooth, color=color, label=batch)
        ax.axvspan(1, 20, color="#dddddd", alpha=0.25)
        ax.axvspan(21, 25, color="#fee08b", alpha=0.25)
        ax.set_title(case)
        ax.set_ylabel("reward_no_penalty")
        ax.grid(axis="y", alpha=0.25)
    axes[0].legend(frameon=False)
    axes[-1].set_xlabel("episode")
    axes[-2].set_xlabel("episode")
    path = FIG_DIR / "low_noise_reward_traces.png"
    fig.savefig(path, dpi=190)
    plt.close(fig)
    return path


def build_report() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    metrics, phases, deltas, meta = collect()
    write_csv(TABLE_DIR / "metrics.csv", metrics.to_dict("records"))
    write_csv(TABLE_DIR / "phase_metrics.csv", phases.to_dict("records"))
    write_csv(TABLE_DIR / "low_minus_old_deltas.csv", deltas.to_dict("records"))

    tail_fig = plot_tail_delta(deltas)
    phase_fig = plot_phase_reward(phases)
    trace_fig = plot_episode_traces(meta)

    low = metrics.loc[metrics["batch"].eq("low_noise")].set_index("case")
    old = metrics.loc[metrics["batch"].eq("old_noise")].set_index("case")

    perf_rows = []
    for case in CASE_ORDER:
        row = low.loc[case]
        perf_rows.append(
            {
                "case": case,
                "reward": fmt(row["reward_no_penalty_mean"]),
                "tail": fmt(row["tail50_reward_no_penalty"]),
                "tail_rmse": fmt(row["tail50_output_rmse_mean"]),
                "gate": fmt(100.0 * row["actual_intervention_rate"]),
                "diag": fmt(100.0 * row["diagnostic_unsafe_rate"]),
                "du": fmt(row["mean_abs_du_phys"]),
            }
        )
    perf_table = md_table(
        perf_rows,
        [
            ("case", "Case"),
            ("reward", "Mean no-penalty reward"),
            ("tail", "Tail50 reward"),
            ("tail_rmse", "Tail50 RMSE"),
            ("gate", "Actual gate %"),
            ("diag", "Diag unsafe %"),
            ("du", "Mean abs dU phys"),
        ],
    )

    delta_rows = []
    for row in deltas.to_dict("records"):
        delta_rows.append(
            {
                "case": row["case"],
                "d_reward": fmt(row["delta_reward_no_penalty"]),
                "d_tail": fmt(row["delta_tail50_reward_no_penalty"]),
                "d_rmse": fmt(row["delta_tail50_rmse"]),
                "d_penalty": fmt(row["delta_fallback_penalty"]),
                "d_gate": fmt(row["delta_actual_intervention_pct"]),
                "d_diag": fmt(row["delta_diag_unsafe_pct"]),
            }
        )
    delta_table = md_table(
        delta_rows,
        [
            ("case", "Case"),
            ("d_reward", "Delta mean reward"),
            ("d_tail", "Delta tail reward"),
            ("d_rmse", "Delta tail RMSE"),
            ("d_penalty", "Delta penalty"),
            ("d_gate", "Delta gate pp"),
            ("d_diag", "Delta diag pp"),
        ],
    )

    phase_rows = []
    for case in CASE_ORDER:
        for phase in ["BC", "handoff", "early full", "tail 50"]:
            old_row = phases.loc[(phases["case"].eq(case)) & phases["batch"].eq("old_noise") & phases["phase"].eq(phase)]
            low_row = phases.loc[(phases["case"].eq(case)) & phases["batch"].eq("low_noise") & phases["phase"].eq(phase)]
            if old_row.empty or low_row.empty:
                continue
            phase_rows.append(
                {
                    "case_phase": f"{case} - {phase}",
                    "old": fmt(float(old_row["reward_no_penalty"].iloc[0])),
                    "low": fmt(float(low_row["reward_no_penalty"].iloc[0])),
                    "delta": fmt(float(low_row["reward_no_penalty"].iloc[0] - old_row["reward_no_penalty"].iloc[0])),
                    "old_rmse": fmt(float(old_row["output_rmse"].iloc[0])),
                    "low_rmse": fmt(float(low_row["output_rmse"].iloc[0])),
                }
            )
    phase_table = md_table(
        phase_rows,
        [
            ("case_phase", "Case phase"),
            ("old", "Old reward"),
            ("low", "Low reward"),
            ("delta", "Delta"),
            ("old_rmse", "Old RMSE"),
            ("low_rmse", "Low RMSE"),
        ],
    )

    run_rows = []
    for case in CASE_ORDER:
        run_rows.append(
            {
                "case": case,
                "old": old.loc[case, "run_dir"],
                "low": low.loc[case, "run_dir"],
                "old_agent": old.loc[case, "initial_agent"],
                "low_agent": low.loc[case, "initial_agent"],
            }
        )
    run_table = md_table(
        run_rows,
        [("case", "Case"), ("old", "Old-noise run"), ("low", "Low-noise run"), ("old_agent", "Old agent"), ("low_agent", "Low agent")],
    )

    report = f"""# Low-Noise Online Runner Regression Analysis

Date: 2026-06-12

## Question

The six online TD3 disturbance runners were rerun after the BC/handoff exploration
change. Performance got much worse. This report compares that low-noise batch
against the previous bounded-mixed online batch.

Short answer: the user's impression is right for the pretrained runs, especially
through handoff and early full RL. It is not true for cold-start runs. Cold-start
benefits from the lower BC noise because the old `0.1` BC noise was too large.
The pretrained runs need some local action variation during BC/handoff, or a
critic recalibration phase, before full actor updates take over.

## Data Used

{run_table}

The low-noise batch used:

- pretrained BC: `bc_behavior_noise="none"`, `bc_exploration_std=0.0`
- cold-start BC: `bc_exploration_std=0.005`
- handoff noise ending at `0.005` for pretrained and `0.01` for cold-start
- full-RL exploration unchanged after handoff

Important caveat: the LMPC-pretrained low-noise runs also loaded the newer
bounded-mixed LMPC checkpoint. Their degradation is therefore a combined
checkpoint-plus-schedule effect. The OF-MPC-pretrained and cold-start cases are
cleaner tests of the low-noise schedule.

## Low-Noise Batch Performance

{perf_table}

![Tail deltas]({rel_report(tail_fig)})

## Low-Noise Minus Old-Noise

Positive reward deltas are better. Negative RMSE deltas are better.

{delta_table}

The cleanest schedule-only comparisons are the OF-MPC-pretrained and cold-start
cases because their checkpoint status did not change.

- OF-MPC-pretrained runs got worse, especially in handoff and early full RL.
- Cold-start runs got better in mean reward and early learning, with tail
  performance roughly tied or slightly better.
- LMPC-pretrained runs are confounded by a checkpoint change: the low-noise runs
  loaded the newer bounded-mixed LMPC checkpoint, while the old-noise runs loaded
  the older governed-reference checkpoint.

![Phase reward]({rel_report(phase_fig)})

![Reward traces]({rel_report(trace_fig)})

## Phase Diagnosis

{phase_table}

The important pattern is not that BC became worse. BC improves in every case.
The failure mode for pretrained runs starts at handoff and early full RL:

- OF-MPC pretrained + gate: BC improves by `+20.468`, but handoff drops by
  `-56.279` and early full RL drops by `-1138`.
- OF-MPC pretrained no gate: BC improves by `+4.695`, but handoff drops by
  `-62.280` and early full RL drops by `-44.718`.

This points to under-exploration and critic-distribution mismatch for pretrained
online learning:

1. In BC, the critic sees a narrow teacher-driven state-action distribution.
2. The actor is also pulled tightly toward the clean teacher action.
3. Handoff uses very small policy-side noise.
4. Full RL begins from a policy/critic pair that has not seen enough local action
   variation around the teacher trajectory.
5. When full exploration resumes, the critic is less prepared for the policy
   actions and the actor update can drift into poorer behavior.

So the previous noisy BC was ugly, but it may have been doing something useful:
it gave the critic online-reward data around the teacher action neighborhood.
Removing that variation made the early supervised behavior cleaner but less
useful for later TD3 learning.

## What This Means

The low-noise idea was half right:

- For cold-start, reducing BC noise from `0.1` to `0.005` clearly helped.
- For pretrained runs, setting BC noise to exactly zero was too conservative.

A better compromise is:

- pretrained BC std: `0.01` or `0.02`, not `0.0`
- cold-start BC std: keep `0.005` to `0.01`
- pretrained handoff should ramp to the full-RL std, not stop at `0.005`
- cold-start handoff can stay small or modestly increase
- keep full-RL std unchanged

That preserves teacher-centered behavior while giving the critic enough local
action perturbations to learn the online reward landscape.

## Recommended Next Step

Do not fully revert the low-noise schedule. Split the policy:

| Runner family | Recommended BC std | Recommended handoff end | Full RL start |
| :--- | ---: | ---: | ---: |
| pretrained | 0.010 to 0.020 | 0.020 | 0.020 |
| cold-start | 0.005 to 0.010 | 0.010 to 0.030 | 0.100 |

Then rerun the schedule-isolation cases first:

1. `OnlineTD3_OFMPCPretrained_SafetyGate.py`
2. `OnlineTD3_OFMPCPretrained_NoSafetyGate.py`
3. `OnlineTD3_ColdStart_SafetyGate.py`
4. `OnlineTD3_ColdStart_NoSafetyGate.py`

Those isolate the schedule without the LMPC checkpoint confound. If they recover,
then rerun the two LMPC-pretrained cases.

## Relation To The Previous Strategy Report

This result strengthens the case for critic recalibration. The issue is not just
teacher noise; it is the critic's online data distribution and reward scale.

Before implementing DAgger-style relabeling, I would now do:

1. moderate pretrained BC/handoff exploration, not zero exploration
2. actor-frozen critic recalibration for pretrained runs
3. critic last-layer reset if recalibration alone does not help

DAgger-style relabeling is still promising, but it should not be implemented as
pure clean-teacher imitation only. It should include either local action
perturbations for critic coverage or a separate critic recalibration phase.

## Exported Tables

- `{rel_report(TABLE_DIR / 'metrics.csv')}`
- `{rel_report(TABLE_DIR / 'phase_metrics.csv')}`
- `{rel_report(TABLE_DIR / 'low_minus_old_deltas.csv')}`
"""
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(report, encoding="utf-8")


if __name__ == "__main__":
    build_report()
    print(f"Wrote {REPORT}")
