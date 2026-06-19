from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = REPO_ROOT / "results"
REPORT_ROOT = REPO_ROOT / "report"
FIG_ROOT = REPORT_ROOT / "figures" / "2026-06-19_online_td3_latest"
REPORT_PATH = REPORT_ROOT / "online_td3_latest_analysis_2026-06-19.md"


@dataclass(frozen=True)
class CaseSpec:
    key: str
    label: str
    result_root: str
    family: str
    safety_gate: bool
    pretrained: bool


CASES = [
    CaseSpec(
        key="cold_gate",
        label="Cold start + gate",
        result_root="OnlineTD3_ColdStart_SafetyGate",
        family="Cold start",
        safety_gate=True,
        pretrained=False,
    ),
    CaseSpec(
        key="cold_nogate",
        label="Cold start no gate",
        result_root="OnlineTD3_ColdStart_NoSafetyGate",
        family="Cold start",
        safety_gate=False,
        pretrained=False,
    ),
    CaseSpec(
        key="of_gate",
        label="OF-MPC pretrained + gate",
        result_root="OnlineTD3_OFMPCPretrained_SafetyGate",
        family="OF-MPC pretrained",
        safety_gate=True,
        pretrained=True,
    ),
    CaseSpec(
        key="of_nogate",
        label="OF-MPC pretrained no gate",
        result_root="OnlineTD3_OFMPCPretrained_NoSafetyGate",
        family="OF-MPC pretrained",
        safety_gate=False,
        pretrained=True,
    ),
]


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _is_completed_run(path: Path) -> bool:
    if not path.is_dir():
        return False
    if path.name.startswith("diagnostic"):
        return False
    step = path / "step_table.csv"
    ep = path / "episode_table.csv"
    summary = path / "summary.json"
    if not (step.exists() and ep.exists() and summary.exists()):
        return False
    try:
        meta = _load_json(summary)
    except Exception:
        return False
    return int(meta.get("n_steps", 0) or meta.get("wall_clock_n_steps", 0) or 0) >= 100_000


def latest_completed_run(result_root: str) -> Path:
    root = RESULTS_ROOT / result_root
    candidates = [p for p in root.iterdir() if _is_completed_run(p)]
    if not candidates:
        raise FileNotFoundError(f"No completed run found under {root}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def latest_pending_dirs(result_root: str, completed_path: Path) -> list[Path]:
    root = RESULTS_ROOT / result_root
    out = []
    for path in sorted([p for p in root.iterdir() if p.is_dir()], key=lambda p: p.stat().st_mtime, reverse=True):
        if path.name.startswith("diagnostic"):
            continue
        if path.stat().st_mtime <= completed_path.stat().st_mtime:
            continue
        if _is_completed_run(path):
            continue
        if not (path / "step_table.csv").exists():
            out.append(path)
        if len(out) >= 1:
            break
    return out


def phase_labels(step_df: pd.DataFrame) -> pd.Series:
    phase = step_df.get("policy_phase", pd.Series(["unknown"] * len(step_df))).fillna("unknown").astype(str)
    handoff = step_df.get("handoff_active", pd.Series([False] * len(step_df))).fillna(False).astype(bool)
    out = phase.copy()
    out[(phase == "behavior_clone_teacher")] = "teacher critic"
    out[(phase == "full_rl") & handoff] = "handoff"
    out[(phase == "full_rl") & (~handoff)] = "full RL"
    return out


def safe_mean(values) -> float:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0 or np.all(~np.isfinite(arr)):
        return float("nan")
    return float(np.nanmean(arr))


def safe_max(values) -> float:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0 or np.all(~np.isfinite(arr)):
        return float("nan")
    return float(np.nanmax(arr))


def phase_output_rmse(arrays: dict, idx: np.ndarray) -> tuple[float, float, float]:
    if idx.size == 0:
        return float("nan"), float("nan"), float("nan")
    y = np.asarray(arrays["y_system"], dtype=float)[:-1, :]
    y_sp = np.asarray(arrays["y_sp_phys_store"], dtype=float)
    err = y[idx, :] - y_sp[idx, :]
    rmse = np.sqrt(np.nanmean(err * err, axis=0))
    return float(np.nanmean(rmse)), float(rmse[0]), float(rmse[1])


def load_case(spec: CaseSpec) -> dict:
    path = latest_completed_run(spec.result_root)
    summary = _load_json(path / "summary.json")
    record = _load_json(path / "record.json") if (path / "record.json").exists() else dict(summary)
    run_summary = _load_json(path / "run_summary.json") if (path / "run_summary.json").exists() else {}
    step = pd.read_csv(path / "step_table.csv", low_memory=False).copy()
    episode = pd.read_csv(path / "episode_table.csv")
    arrays_np = np.load(path / "arrays.npz", allow_pickle=True)
    arrays = {k: arrays_np[k] for k in arrays_np.files}
    step["analysis_phase"] = phase_labels(step)
    return {
        "spec": spec,
        "path": path,
        "summary": summary,
        "record": record,
        "run_summary": run_summary,
        "step": step,
        "episode": episode,
        "arrays": arrays,
    }


def summarize_case(case: dict) -> dict:
    spec = case["spec"]
    rec = case["record"]
    path = case["path"]
    cfg = (case["run_summary"].get("config") or {})
    training_cfg = cfg.get("training_phase_config") or {}
    step = case["step"]
    if spec.safety_gate:
        candidate_pass_rate = float(rec.get("accepted_rate", np.nan))
    elif "diagnostic_candidate_accepted" in step.columns:
        candidate_pass_rate = safe_mean(step["diagnostic_candidate_accepted"])
    else:
        candidate_pass_rate = float("nan")
    return {
        "case": spec.label,
        "run_id": path.name,
        "family": spec.family,
        "safety_gate": spec.safety_gate,
        "pretrained": spec.pretrained,
        "run_dir": str(path.relative_to(REPO_ROOT)),
        "mtime": pd.Timestamp.fromtimestamp(path.stat().st_mtime).isoformat(),
        "episodes": int(rec.get("wall_clock_n_episodes", len(case["episode"]))),
        "steps": int(rec.get("wall_clock_n_steps", len(step))),
        "reward_no_penalty": float(rec.get("reward_no_penalty_mean", np.nan)),
        "training_reward": float(rec.get("reward_mean", np.nan)),
        "fallback_penalty_mean": float(rec.get("fallback_penalty_mean", np.nan)),
        "output_rmse_mean": float(rec.get("output_rmse_mean", np.nan)),
        "eta_rmse": float(rec.get("output0_rmse", np.nan)),
        "T_rmse": float(rec.get("output1_rmse", np.nan)),
        "candidate_pass_rate": candidate_pass_rate,
        "accepted_rate": float(rec.get("accepted_rate", np.nan)),
        "actual_intervention_rate": float(rec.get("actual_intervention_rate", np.nan)),
        "fallback_rate": float(rec.get("fallback_rate", np.nan)),
        "diagnostic_unsafe_rate": float(rec.get("diagnostic_unsafe_rate", np.nan)),
        "target_failures": int(rec.get("n_target_failures", 0) or 0),
        "max_action_gap": float(rec.get("executed_action_gap_inf_max", np.nan)),
        "bc_teacher_gap_mean": float(rec.get("bc_teacher_gap_inf_mean", np.nan)),
        "bc_teacher_gap_max": float(rec.get("bc_teacher_gap_inf_max", np.nan)),
        "bc_update_mode": training_cfg.get("bc_update_mode"),
        "teacher_episodes": training_cfg.get("behavior_clone_teacher_episodes"),
        "handoff_episodes": training_cfg.get("handoff_episodes"),
        "bc_exploration_std": training_cfg.get("bc_exploration_std"),
        "handoff_exploration_end": training_cfg.get("handoff_exploration_std_end"),
        "full_rl_exploration_start": training_cfg.get("full_rl_exploration_std_start"),
        "full_rl_exploration_end": training_cfg.get("full_rl_exploration_std_end"),
        "bc_exploration_space": training_cfg.get("bc_exploration_space"),
        "full_rl_exploration_space": training_cfg.get("full_rl_exploration_space"),
    }


def phase_metrics(case: dict) -> list[dict]:
    rows = []
    spec = case["spec"]
    step = case["step"]
    arrays = case["arrays"]
    for phase in ["teacher critic", "handoff", "full RL"]:
        mask = step["analysis_phase"].eq(phase).to_numpy()
        idx = np.flatnonzero(mask)
        if idx.size == 0:
            continue
        rmse_mean, rmse0, rmse1 = phase_output_rmse(arrays, idx)
        rows.append(
            {
                "case": spec.label,
                "phase": phase,
                "steps": int(idx.size),
                "reward_no_penalty": safe_mean(step.loc[mask, "reward_no_penalty"]),
                "training_reward": safe_mean(step.loc[mask, "reward"]),
                "output_rmse_mean": rmse_mean,
                "eta_rmse": rmse0,
                "T_rmse": rmse1,
                "actual_intervention_rate": safe_mean(step.loc[mask, "actual_intervention"]),
                "fallback_rate": safe_mean(step.loc[mask, "fallback_mpc_active"]),
                "diagnostic_unsafe_rate": safe_mean(step.loc[mask, "diagnostic_unsafe"]),
                "accepted_rate": safe_mean(step.loc[mask, "accepted"]),
                "bc_teacher_gap_mean": safe_mean(step.loc[mask, "bc_teacher_gap_inf"]),
                "max_action_gap": safe_max(step.loc[mask, "executed_action_gap_inf"]),
            }
        )
    return rows


def write_csvs(summary_df: pd.DataFrame, phase_df: pd.DataFrame, episode_df: pd.DataFrame, pending_df: pd.DataFrame) -> None:
    FIG_ROOT.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(FIG_ROOT / "summary_metrics.csv", index=False)
    phase_df.to_csv(FIG_ROOT / "phase_metrics.csv", index=False)
    episode_df.to_csv(FIG_ROOT / "episode_metrics.csv", index=False)
    pending_df.to_csv(FIG_ROOT / "pending_runs.csv", index=False)


def save_bar_summary(summary_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    labels = summary_df["case"].tolist()
    x = np.arange(len(labels))
    plots = [
        ("reward_no_penalty", "Reward no penalty", axes[0, 0]),
        ("output_rmse_mean", "Mean output RMSE", axes[0, 1]),
        ("actual_intervention_rate", "Actual intervention rate", axes[1, 0]),
        ("diagnostic_unsafe_rate", "Diagnostic unsafe rate", axes[1, 1]),
    ]
    colors = ["#3b6ea8", "#7a9e4f", "#c46a3a", "#6f5aa8"]
    for col, title, ax in plots:
        vals = summary_df[col].astype(float).to_numpy()
        ax.bar(x, vals, color=colors)
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25, ha="right")
        ax.grid(axis="y", alpha=0.25)
    fig.savefig(FIG_ROOT / "summary_bar_metrics.png", dpi=180)
    plt.close(fig)


def save_episode_trends(case_data: list[dict]) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True, constrained_layout=True)
    for case in case_data:
        label = case["spec"].label
        ep = case["episode"]
        x = ep["episode"]
        axes[0].plot(x, ep["reward_no_penalty_mean"], label=label, linewidth=1.3)
        axes[1].plot(x, ep["output_rmse_mean"], label=label, linewidth=1.3)
        if case["spec"].safety_gate:
            y = ep["actual_intervention_rate"]
        else:
            y = ep["diagnostic_unsafe_rate"]
        axes[2].plot(x, y, label=label, linewidth=1.3)
    axes[0].set_ylabel("Reward no penalty")
    axes[1].set_ylabel("Output RMSE")
    axes[2].set_ylabel("Gate/diagnostic rate")
    axes[2].set_xlabel("Episode")
    for ax in axes:
        ax.grid(alpha=0.25)
    axes[0].legend(ncol=2, fontsize=8)
    fig.savefig(FIG_ROOT / "episode_trends.png", dpi=180)
    plt.close(fig)


def save_phase_metrics(phase_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    phases = ["teacher critic", "handoff", "full RL"]
    width = 0.18
    x = np.arange(len(phases))
    for idx, case in enumerate(phase_df["case"].drop_duplicates()):
        sub = phase_df[phase_df["case"] == case].set_index("phase").reindex(phases)
        axes[0].bar(x + (idx - 1.5) * width, sub["output_rmse_mean"], width=width, label=case)
        rate_col = "actual_intervention_rate" if "gate" in case and "no gate" not in case else "diagnostic_unsafe_rate"
        axes[1].bar(x + (idx - 1.5) * width, sub[rate_col], width=width, label=case)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(phases)
    axes[0].set_ylabel("Mean output RMSE")
    axes[0].set_title("Tracking by training phase")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(phases)
    axes[1].set_ylabel("Intervention or diagnostic unsafe rate")
    axes[1].set_title("Gate activity by training phase")
    for ax in axes:
        ax.grid(axis="y", alpha=0.25)
    axes[0].legend(fontsize=8)
    fig.savefig(FIG_ROOT / "phase_metrics.png", dpi=180)
    plt.close(fig)


def save_last_episode_tracking(case_data: list[dict]) -> None:
    fig, axes = plt.subplots(len(case_data), 2, figsize=(13, 11), sharex=True, constrained_layout=True)
    for row, case in enumerate(case_data):
        ep = case["episode"].iloc[-1]
        start = int(ep["step_start"])
        stop = int(ep["step_stop_exclusive"])
        t = np.arange(stop - start)
        y = np.asarray(case["arrays"]["y_system"], dtype=float)[:-1, :][start:stop]
        y_sp = np.asarray(case["arrays"]["y_sp_phys_store"], dtype=float)[start:stop]
        for j, name in enumerate(["eta", "T"]):
            ax = axes[row, j]
            ax.plot(t, y[:, j], label="output", linewidth=1.2)
            ax.plot(t, y_sp[:, j], label="setpoint", linestyle="--", linewidth=1.0)
            ax.set_title(f"{case['spec'].label}: {name}")
            ax.grid(alpha=0.25)
            if row == len(case_data) - 1:
                ax.set_xlabel("Step in final episode")
            if j == 0:
                ax.set_ylabel("Physical output")
            if row == 0 and j == 1:
                ax.legend(fontsize=8)
    fig.savefig(FIG_ROOT / "final_episode_tracking.png", dpi=180)
    plt.close(fig)


def save_input_activity(case_data: list[dict]) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True, constrained_layout=True)
    for case in case_data:
        ep = case["episode"].iloc[-1]
        start = int(ep["step_start"])
        stop = int(ep["step_stop_exclusive"])
        t = np.arange(stop - start)
        u = np.asarray(case["arrays"]["u_applied_phys"], dtype=float)[start:stop]
        axes[0].plot(t, u[:, 0], label=case["spec"].label, linewidth=1.1)
        axes[1].plot(t, u[:, 1], label=case["spec"].label, linewidth=1.1)
    axes[0].set_ylabel("Qc")
    axes[1].set_ylabel("Qm")
    axes[1].set_xlabel("Step in final episode")
    for ax in axes:
        ax.grid(alpha=0.25)
    axes[0].legend(ncol=2, fontsize=8)
    fig.savefig(FIG_ROOT / "final_episode_inputs.png", dpi=180)
    plt.close(fig)


def md_table(df: pd.DataFrame, columns: list[str], formats: dict[str, str] | None = None) -> str:
    formats = formats or {}
    rows = []
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] + ["---:" for _ in columns[1:]]) + " |"
    rows.extend([header, sep])
    for _, row in df.iterrows():
        cells = []
        for col in columns:
            val = row[col]
            if pd.isna(val):
                text = ""
            elif col in formats:
                text = formats[col].format(val)
            else:
                text = str(val)
            cells.append(text)
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join(rows)


def write_report(summary_df: pd.DataFrame, phase_df: pd.DataFrame, pending_df: pd.DataFrame) -> None:
    perf_cols = ["case", "reward_no_penalty", "training_reward", "output_rmse_mean", "eta_rmse", "T_rmse"]
    safety_cols = [
        "case",
        "candidate_pass_rate",
        "actual_intervention_rate",
        "fallback_rate",
        "diagnostic_unsafe_rate",
        "target_failures",
    ]
    fmt = {
        "reward_no_penalty": "{:.3f}",
        "training_reward": "{:.3f}",
        "output_rmse_mean": "{:.3f}",
        "eta_rmse": "{:.3f}",
        "T_rmse": "{:.3f}",
        "candidate_pass_rate": "{:.3%}",
        "accepted_rate": "{:.3%}",
        "actual_intervention_rate": "{:.3%}",
        "fallback_rate": "{:.3%}",
        "diagnostic_unsafe_rate": "{:.3%}",
        "target_failures": "{:.0f}",
    }
    phase_small = phase_df.copy()
    phase_small["intervention_or_diag_rate"] = np.where(
        phase_small["case"].str.contains("no gate", case=False),
        phase_small["diagnostic_unsafe_rate"],
        phase_small["actual_intervention_rate"],
    )
    phase_cols = [
        "case",
        "phase",
        "steps",
        "reward_no_penalty",
        "output_rmse_mean",
        "intervention_or_diag_rate",
    ]
    phase_fmt = {
        "steps": "{:.0f}",
        "reward_no_penalty": "{:.3f}",
        "output_rmse_mean": "{:.3f}",
        "intervention_or_diag_rate": "{:.3%}",
    }

    pending_lines = []
    if len(pending_df) > 0:
        for _, row in pending_df.iterrows():
            pending_lines.append(f"- `{row['run_dir']}` has no final `step_table.csv` yet.")
    else:
        pending_lines.append("- No pending online TD3 folders were detected.")

    text = f"""# Latest Online TD3 Comparison

Date: 2026-06-19

## Objective

This report compares the latest completed online TD3 disturbance runs for the
four active runners:

- cold start with GART-LMPC safety gate
- cold start without active safety gate
- OF-MPC-pretrained with GART-LMPC safety gate
- OF-MPC-pretrained without active safety gate

The analysis uses the latest completed 300-episode result folders available at
report-generation time. Four newer `20260619_1316xx` jobs were detected but did
not yet contain final `step_table.csv` or `episode_table.csv` exports, so they
are listed as pending rather than mixed into the completed-run comparison.

## Data Used

{md_table(summary_df[["case", "run_id", "episodes", "steps"]], ["case", "run_id", "episodes", "steps"], {"episodes": "{:.0f}", "steps": "{:.0f}"})}

Full selected paths are recorded in
`report/figures/2026-06-19_online_td3_latest/summary_metrics.csv`.

Pending current runs:

{chr(10).join(pending_lines)}

Generated analysis artifacts:

- `report/figures/2026-06-19_online_td3_latest/summary_metrics.csv`
- `report/figures/2026-06-19_online_td3_latest/phase_metrics.csv`
- `report/figures/2026-06-19_online_td3_latest/episode_metrics.csv`
- `report/figures/2026-06-19_online_td3_latest/pending_runs.csv`

## Method

All four completed runs use the polymer CSTR disturbance setup with 300
episodes and 800 steps per episode. The analyzed completed runs include the
noisy GART-LMPC teacher critic warmup schedule:

$$
\\text{{teacher episodes}} = 10,
\\qquad
\\text{{update mode}} = \\text{{critic TD only}},
\\qquad
\\text{{handoff episodes}} = 10.
$$

The safety-gate cases evaluate the TD3 candidate action using the GART target
and first-step Lyapunov contraction test. If the candidate fails, the applied
input is replaced by the GART-LMPC fallback. The no-gate cases still record a
diagnostic gate decision, but execute the candidate action without replacement.

Important timing note: the completed runs analyzed here predate the final
probe-style full-RL exploration commit. The currently running `20260619_1316xx`
jobs are the runs expected to reflect that final exploration change.

## Overall Performance

{md_table(summary_df[perf_cols], perf_cols, fmt)}

![Summary metrics](figures/2026-06-19_online_td3_latest/summary_bar_metrics.png)

## Gate And Diagnostic Reliability

{md_table(summary_df[safety_cols], safety_cols, fmt)}

For gate runs, `candidate_pass_rate` is the actual accepted-candidate rate. For
no-gate runs, it is the diagnostic candidate pass rate. `actual_intervention_rate`
is the fraction of gate-run steps where the gate changed the candidate input
through fallback or hold-previous logic. For no-gate runs,
`diagnostic_unsafe_rate` is the would-have-been rejected rate under the
diagnostic GART gate.

## Phase Breakdown

{md_table(phase_small[phase_cols], phase_cols, phase_fmt)}

![Phase metrics](figures/2026-06-19_online_td3_latest/phase_metrics.png)

## Episode Trends

![Episode trends](figures/2026-06-19_online_td3_latest/episode_trends.png)

The episode curves show that the safety-gate runs pay an explicit training
reward penalty whenever fallback is active. Therefore, `reward_no_penalty` is
the cleaner control-performance comparison across gate and no-gate cases.

## Final-Episode Tracking And Inputs

![Final episode tracking](figures/2026-06-19_online_td3_latest/final_episode_tracking.png)

![Final episode inputs](figures/2026-06-19_online_td3_latest/final_episode_inputs.png)

## Interpretation

The latest completed runs show the same broad pattern as the earlier
2026-06-17 analysis: the no-gate cases have better average tracking and better
`reward_no_penalty` on these completed runs, while the gate cases prevent or
replace a subset of candidate actions. The safety gate is therefore acting as a
robustness layer, but these completed results do not yet show a tracking
advantage for the gate.

The OF-MPC-pretrained no-gate run is the strongest completed result by average
`reward_no_penalty` and mean output RMSE. The OF-MPC-pretrained gate run has a
larger intervention burden and worse tracking, which suggests the gate is
constraining or replacing candidate actions often enough to reduce nominal
performance in this configuration.

For the cold-start pair, the gate protects against rejected candidates but also
introduces fallback penalties and some tracking degradation. This is not
necessarily a failure of the gate: it means the comparison is currently a
robustness-versus-performance tradeoff rather than a clean tracking win.

## Risks And Consistency Checks

- The newest probe-style full-RL exploration runs are still pending. Do not use
  this report as final evidence for the new full-RL exploration change.
- The completed runs include noisy teacher critic warmup but do not expose the
  newer `behavior_exploration_space` columns in `step_table.csv`, confirming
  they were produced before the latest diagnostic/export changes.
- Gate and no-gate training rewards are not directly comparable because only
  gate runs include fallback penalties. Use `reward_no_penalty` for
  cross-method control-performance comparison.

## Recommended Next Experiment

After the four `20260619_1316xx` runs finish, rerun this analysis script without
changing the selection logic. The next report should compare:

- accepted candidate rate before and after probe-style full-RL exploration
- intervention rate in safety-gate runs
- diagnostic unsafe rate in no-gate runs
- final 100-episode `reward_no_penalty` and output RMSE
- whether cold-start no-gate becomes less rough when the same exploration scale
  is applied directly in `u_dev` coordinates

The result that would support the new exploration change is a lower diagnostic
unsafe/intervention rate without losing exploration-driven improvement in
`reward_no_penalty`.
"""
    REPORT_PATH.write_text(text, encoding="utf-8")


def main() -> None:
    FIG_ROOT.mkdir(parents=True, exist_ok=True)
    case_data = [load_case(spec) for spec in CASES]
    summary_df = pd.DataFrame([summarize_case(case) for case in case_data])
    phase_df = pd.DataFrame([row for case in case_data for row in phase_metrics(case)])
    episode_rows = []
    for case in case_data:
        ep = case["episode"].copy()
        ep.insert(0, "case", case["spec"].label)
        episode_rows.append(ep)
    episode_df = pd.concat(episode_rows, ignore_index=True)
    pending_rows = []
    for spec, case in zip(CASES, case_data):
        for path in latest_pending_dirs(spec.result_root, case["path"]):
            pending_rows.append(
                {
                    "case": spec.label,
                    "run_dir": str(path.relative_to(REPO_ROOT)),
                    "mtime": pd.Timestamp.fromtimestamp(path.stat().st_mtime).isoformat(),
                }
            )
    pending_df = pd.DataFrame(pending_rows)

    write_csvs(summary_df, phase_df, episode_df, pending_df)
    save_bar_summary(summary_df)
    save_episode_trends(case_data)
    save_phase_metrics(phase_df)
    save_last_episode_tracking(case_data)
    save_input_activity(case_data)
    write_report(summary_df, phase_df, pending_df)
    print(f"Wrote {REPORT_PATH.relative_to(REPO_ROOT)}")
    print(f"Wrote figures under {FIG_ROOT.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
