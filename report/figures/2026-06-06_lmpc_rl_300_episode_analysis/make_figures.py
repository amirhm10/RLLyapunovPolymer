from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


REPO = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent

STUDIES = {
    "direct_lmpc": {
        "title": "Direct LMPC",
        "root": REPO / "results" / "directLyap" / "20260606_020549",
        "cases": {
            "lyap_governed_reference": "LMPC",
            "mpc_only": "MPC-only",
        },
    },
    "cold_start": {
        "title": "Cold-start safety-gate RL",
        "root": REPO / "results" / "ColdStart" / "20260606_020555",
        "cases": {
            "rl_gate_governed_reference": "RL gate",
            "mpc_only": "No-gate diag",
        },
    },
    "pretrained": {
        "title": "Pretrained safety-gate RL",
        "root": REPO / "results" / "Pretrain" / "20260606_020559",
        "cases": {
            "rl_gate_governed_reference": "RL gate",
            "mpc_only": "No-gate diag",
        },
    },
}

DT = 0.5


def as_float(value, default=np.nan):
    if value is None:
        return default
    try:
        value = float(value)
    except Exception:
        return default
    return value if np.isfinite(value) else default


def arr(a, key, default=None):
    if key in a.files:
        return np.asarray(a[key], dtype=float)
    return default


def row_inf(x):
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        return np.abs(x)
    return np.nanmax(np.abs(x), axis=1)


def load_bundle(root: Path, case: str):
    summary = json.loads((root / case / "summary.json").read_text())
    arrays = np.load(root / case / "arrays.npz", allow_pickle=True)
    return summary, arrays


def output_arrays(a):
    if "y_minus_y_sp_phys_store" in a.files:
        err = arr(a, "y_minus_y_sp_phys_store")
        n = err.shape[0]
        y = arr(a, "y_system")[:n]
        sp = y - err
        return y, sp, err
    y = arr(a, "y_system")
    sp = arr(a, "y_sp_phys_store")
    if y is None or sp is None:
        raise KeyError("No physical output/setpoint arrays found.")
    n = min(y.shape[0], sp.shape[0])
    y = y[:n]
    sp = sp[:n]
    return y, sp, y - sp


def effective_margin(a):
    for key in ("contraction_margin_applied", "contraction_margin", "final_lyap_margin"):
        if key in a.files:
            return np.asarray(a[key], dtype=float).reshape(-1)
    V = arr(a, "V_k")
    Vn = arr(a, "V_next_first")
    B = arr(a, "V_bound")
    if V is not None and Vn is not None and B is not None:
        n = min(len(V), len(Vn), len(B))
        return np.asarray(Vn[:n] - B[:n], dtype=float).reshape(-1)
    return np.array([], dtype=float)


def episode_len(summary, n):
    ep = int(as_float(summary.get("wall_clock_n_episodes"), 0))
    if ep <= 0:
        ep = 300
    return max(int(n // ep), 1), ep


def per_episode(values, ep_len, n_ep, reducer=np.nanmean):
    values = np.asarray(values, dtype=float).reshape(-1)
    out = np.full(n_ep, np.nan)
    for i in range(n_ep):
        sl = slice(i * ep_len, min((i + 1) * ep_len, len(values)))
        if sl.start < sl.stop:
            out[i] = reducer(values[sl])
    return out


def per_episode_rmse(err, ep_len, n_ep):
    err = np.asarray(err, dtype=float)
    out = np.full(n_ep, np.nan)
    for i in range(n_ep):
        sl = slice(i * ep_len, min((i + 1) * ep_len, err.shape[0]))
        if sl.start < sl.stop:
            rmse = np.sqrt(np.nanmean(err[sl] ** 2, axis=0))
            out[i] = float(np.nanmean(rmse))
    return out


def safe_rate(summary, a, key, fallback_key=None):
    if key in summary and summary[key] is not None:
        return as_float(summary[key])
    if fallback_key and fallback_key in a.files:
        return float(np.nanmean(arr(a, fallback_key)))
    return np.nan


def metric_record(study_key, study_title, case, label, summary, a):
    y, sp, err = output_arrays(a)
    n = err.shape[0]
    ep_len, n_ep = episode_len(summary, n)
    rmse_vec = np.sqrt(np.nanmean(err ** 2, axis=0))
    mae_vec = np.nanmean(np.abs(err), axis=0)
    u = arr(a, "u_applied_phys")
    if u is not None and len(u) > 1:
        du_inf = row_inf(np.diff(u, axis=0))
        mean_du_inf = float(np.nanmean(du_inf))
        max_du_inf = float(np.nanmax(du_inf))
    else:
        mean_du_inf = np.nan
        max_du_inf = np.nan

    target_mismatch = arr(a, "target_quality_mismatch_inf")
    if target_mismatch is None:
        target_mismatch = arr(a, "target_mismatch_inf")
    target_residual = arr(a, "target_residual_total_norm")
    target_rate = arr(a, "target_rate_inf")
    reward_no_penalty = arr(a, "reward_no_penalty")
    reward = arr(a, "rewards")
    fallback_flags = arr(a, "reward_fallback_active_flags", np.zeros(n))
    intervention_flags = arr(a, "actual_intervention_flags", np.zeros(n))
    unsafe_flags = arr(a, "diagnostic_unsafe_flags", np.zeros(n))
    accepted_flags = arr(a, "accepted_flags")

    hard_flags = None
    for key in (
        "first_step_contraction_satisfied_applied_flags",
        "first_step_contraction_satisfied_flags",
        "verified_flags",
    ):
        if key in a.files:
            hard_flags = arr(a, key)
            break

    return {
        "study": study_key,
        "study_title": study_title,
        "case": case,
        "label": label,
        "root": str(summary.get("debug_dir", "")),
        "n_steps": int(n),
        "n_episodes": int(n_ep),
        "episode_len": int(ep_len),
        "reward_mean": as_float(summary.get("reward_mean"), np.nanmean(reward) if reward is not None else np.nan),
        "reward_no_penalty_mean": as_float(
            summary.get("reward_no_penalty_mean"), np.nanmean(reward_no_penalty) if reward_no_penalty is not None else np.nan
        ),
        "output0_rmse": float(rmse_vec[0]),
        "output1_rmse": float(rmse_vec[1]),
        "output_rmse_mean": float(np.nanmean(rmse_vec)),
        "output0_mae": float(mae_vec[0]),
        "output1_mae": float(mae_vec[1]),
        "output_mae_mean": float(np.nanmean(mae_vec)),
        "output_max_inf": float(np.nanmax(row_inf(err))),
        "target_mismatch_mean": float(np.nanmean(target_mismatch)) if target_mismatch is not None else np.nan,
        "target_mismatch_max": float(np.nanmax(target_mismatch)) if target_mismatch is not None else np.nan,
        "target_residual_mean": float(np.nanmean(target_residual)) if target_residual is not None else np.nan,
        "target_residual_max": float(np.nanmax(target_residual)) if target_residual is not None else np.nan,
        "target_rate_mean": float(np.nanmean(target_rate)) if target_rate is not None else np.nan,
        "target_rate_max": float(np.nanmax(target_rate)) if target_rate is not None else np.nan,
        "solver_success_rate": as_float(summary.get("solver_success_rate"), np.nan),
        "target_success_rate": as_float(summary.get("target_success_rate"), summary.get("n_target_success", np.nan) / max(n, 1)),
        "hard_contraction_rate": as_float(
            summary.get("hard_contraction_rate"), np.nanmean(hard_flags) if hard_flags is not None else np.nan
        ),
        "diagnostic_unsafe_rate": safe_rate(summary, a, "diagnostic_unsafe_rate", "diagnostic_unsafe_flags"),
        "fallback_rate": safe_rate(summary, a, "fallback_rate", "reward_fallback_active_flags"),
        "actual_intervention_rate": safe_rate(summary, a, "actual_intervention_rate", "actual_intervention_flags"),
        "accepted_rate": as_float(summary.get("accepted_rate"), np.nanmean(accepted_flags) if accepted_flags is not None else np.nan),
        "verified_rate": as_float(summary.get("verified_rate"), np.nanmean(hard_flags) if hard_flags is not None else np.nan),
        "fallback_penalty_mean": as_float(summary.get("fallback_penalty_mean"), np.nan),
        "weighted_correction_gap_mean": as_float(summary.get("weighted_correction_gap_mean"), np.nan),
        "mean_du_inf": mean_du_inf,
        "max_du_inf": max_du_inf,
        "wall_clock_seconds": as_float(summary.get("wall_clock_seconds"), np.nan),
        "_episode_rmse": per_episode_rmse(err, ep_len, n_ep),
        "_episode_reward_no_penalty": per_episode(
            reward_no_penalty if reward_no_penalty is not None else np.full(n, np.nan), ep_len, n_ep
        ),
        "_episode_reward": per_episode(reward if reward is not None else np.full(n, np.nan), ep_len, n_ep),
        "_episode_fallback": per_episode(fallback_flags, ep_len, n_ep),
        "_episode_intervention": per_episode(intervention_flags, ep_len, n_ep),
        "_episode_unsafe": per_episode(unsafe_flags, ep_len, n_ep),
        "_margin": effective_margin(a),
        "_y": y,
        "_sp": sp,
        "_err": err,
        "_arrays": a,
    }


def load_all():
    records = []
    for study_key, spec in STUDIES.items():
        for case, label in spec["cases"].items():
            summary, arrays = load_bundle(spec["root"], case)
            records.append(metric_record(study_key, spec["title"], case, label, summary, arrays))
    return records


def public_record(rec):
    return {
        k: v
        for k, v in rec.items()
        if not k.startswith("_") and not isinstance(v, np.ndarray)
    }


def save_metrics(records):
    public = [public_record(r) for r in records]
    (OUT / "metrics_summary.json").write_text(json.dumps(public, indent=2))
    headers = list(public[0].keys())
    with (OUT / "metrics_table.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(public)

    late = []
    for r in records:
        n_tail = min(50, r["n_episodes"])
        row = {
            "study": r["study"],
            "label": r["label"],
            "tail_episodes": n_tail,
            "last50_episode_rmse_mean": float(np.nanmean(r["_episode_rmse"][-n_tail:])),
            "last50_reward_no_penalty_mean": float(np.nanmean(r["_episode_reward_no_penalty"][-n_tail:])),
            "last50_reward_mean": float(np.nanmean(r["_episode_reward"][-n_tail:])),
            "last50_fallback_rate": float(np.nanmean(r["_episode_fallback"][-n_tail:])),
            "last50_intervention_rate": float(np.nanmean(r["_episode_intervention"][-n_tail:])),
            "last50_diagnostic_unsafe_rate": float(np.nanmean(r["_episode_unsafe"][-n_tail:])),
        }
        late.append(row)
    (OUT / "late_episode_metrics.json").write_text(json.dumps(late, indent=2))
    with (OUT / "late_episode_metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(late[0].keys()))
        writer.writeheader()
        writer.writerows(late)


def case_name(rec):
    return f"{rec['study_title']}\n{rec['label']}"


def make_overall_performance(records):
    labels = [case_name(r) for r in records]
    x = np.arange(len(records))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    vals = [r["output_rmse_mean"] for r in records]
    axes[0].bar(x, vals, color="#2b6cb0")
    axes[0].set_title("Mean output RMSE")
    axes[0].set_ylabel("physical units")
    vals = [r["reward_no_penalty_mean"] for r in records]
    axes[1].bar(x, vals, color="#2f855a")
    axes[1].set_title("Reward without penalties")
    axes[1].set_ylabel("higher is better")
    vals = [r["target_mismatch_mean"] for r in records]
    axes[2].bar(x, vals, color="#805ad5")
    axes[2].set_title("Mean target mismatch")
    axes[2].set_ylabel("inf norm, physical/scaled export")
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "fig_01_overall_performance.png", dpi=220)
    plt.close(fig)


def make_safety_rates(records):
    labels = [case_name(r) for r in records]
    x = np.arange(len(records))
    width = 0.23
    fallback = [np.nan if r["study"] == "direct_lmpc" else r["fallback_rate"] for r in records]
    intervention = [
        np.nan if r["study"] == "direct_lmpc" else r["actual_intervention_rate"] for r in records
    ]
    fig, ax = plt.subplots(figsize=(13.5, 5.2))
    ax.bar(x - width, [r["diagnostic_unsafe_rate"] for r in records], width, label="diagnostic unsafe", color="#c53030")
    ax.bar(x, fallback, width, label="fallback", color="#dd6b20")
    ax.bar(x + width, intervention, width, label="actual intervention", color="#2b6cb0")
    ax.set_title("Safety and intervention rates")
    ax.set_ylabel("rate")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=3, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / "fig_02_safety_rates.png", dpi=220)
    plt.close(fig)


def make_episode_trends(records):
    groups = list(STUDIES)
    fig, axes = plt.subplots(len(groups), 2, figsize=(14, 10.5), sharex=True)
    for row, study_key in enumerate(groups):
        group_records = [r for r in records if r["study"] == study_key]
        for rec in group_records:
            ep = np.arange(1, rec["n_episodes"] + 1)
            axes[row, 0].plot(ep, rec["_episode_rmse"], linewidth=1.1, label=rec["label"])
            axes[row, 1].plot(ep, rec["_episode_reward_no_penalty"], linewidth=1.1, label=rec["label"])
        axes[row, 0].set_title(f"{group_records[0]['study_title']}: output RMSE")
        axes[row, 1].set_title(f"{group_records[0]['study_title']}: reward no penalty")
        axes[row, 0].set_ylabel("RMSE")
        axes[row, 1].set_ylabel("reward")
        for ax in axes[row]:
            ax.grid(alpha=0.25)
            ax.legend(loc="best")
    axes[-1, 0].set_xlabel("episode")
    axes[-1, 1].set_xlabel("episode")
    fig.tight_layout()
    fig.savefig(OUT / "fig_03_episode_rmse_reward.png", dpi=220)
    plt.close(fig)


def make_tail_tracking(records, tail_steps=1600):
    groups = list(STUDIES)
    fig, axes = plt.subplots(len(groups), 2, figsize=(15, 10.5))
    output_names = ["eta", "T"]
    for row, study_key in enumerate(groups):
        group_records = [r for r in records if r["study"] == study_key]
        n = min(r["_y"].shape[0] for r in group_records)
        start = max(n - tail_steps, 0)
        t = np.arange(start, n) * DT
        sp = group_records[0]["_sp"][start:n]
        for j in range(2):
            ax = axes[row, j]
            ax.plot(t, sp[:, j], color="black", linestyle="--", linewidth=1.2, label="setpoint")
            for rec in group_records:
                ax.plot(t, rec["_y"][start:n, j], linewidth=1.0, label=rec["label"])
            ax.set_title(f"{group_records[0]['study_title']}: {output_names[j]} tail")
            ax.set_xlabel("time")
            ax.set_ylabel(output_names[j])
            ax.grid(alpha=0.25)
            if row == 0 and j == 0:
                ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(OUT / "fig_04_tail_tracking.png", dpi=220)
    plt.close(fig)


def make_margin_hist(records):
    fig, axes = plt.subplots(3, 2, figsize=(14, 10.5))
    for ax, rec in zip(axes.ravel(), records):
        m = rec["_margin"]
        m = m[np.isfinite(m)]
        if len(m) == 0:
            ax.text(0.5, 0.5, "no margin data", ha="center", va="center")
        else:
            lo, hi = np.nanquantile(m, [0.01, 0.995])
            if lo == hi:
                lo, hi = np.nanmin(m), np.nanmax(m)
            bins = np.linspace(lo, hi, 80)
            ax.hist(np.clip(m, lo, hi), bins=bins, color="#4a5568", alpha=0.85)
            ax.axvline(0.0, color="#c53030", linewidth=1.4, label="violation threshold")
            ax.set_title(f"{rec['study_title']} - {rec['label']}")
            ax.set_xlabel("contraction margin")
            ax.set_ylabel("count")
            ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(OUT / "fig_05_contraction_margins.png", dpi=220)
    plt.close(fig)


def make_disturbance(records):
    rec = records[0]
    a = rec["_arrays"]
    qi = arr(a, "qi")
    qs = arr(a, "qs")
    ha = arr(a, "ha")
    n = len(qi)
    step = max(n // 2500, 1)
    t = np.arange(0, n, step) * DT
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(t, qi[::step], label="Qi", linewidth=1.4)
    ax.plot(t, qs[::step], label="Qs", linewidth=1.4)
    ax.plot(t, ha[::step] / 1000.0, label="hA / 1000", linewidth=1.4)
    ax.set_title("Shared disturbance profile")
    ax.set_xlabel("time")
    ax.set_ylabel("profile value")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(OUT / "fig_06_disturbance_profile.png", dpi=220)
    plt.close(fig)


def disturbance_checks(records):
    checks = {}
    for study_key in STUDIES:
        group = [r for r in records if r["study"] == study_key]
        base = group[0]["_arrays"]
        other = group[1]["_arrays"]
        checks[study_key] = {}
        for key in ("qi", "qs", "ha"):
            x = arr(base, key)
            y = arr(other, key)
            n = min(len(x), len(y))
            checks[study_key][key] = float(np.nanmax(np.abs(x[:n] - y[:n])))
    (OUT / "disturbance_equality_checks.json").write_text(json.dumps(checks, indent=2))


def main():
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
        }
    )
    records = load_all()
    save_metrics(records)
    disturbance_checks(records)
    make_overall_performance(records)
    make_safety_rates(records)
    make_episode_trends(records)
    make_tail_tracking(records)
    make_margin_hist(records)
    make_disturbance(records)
    print(f"wrote figures and metrics to {OUT}")


if __name__ == "__main__":
    main()
