import csv
import json
import os
import pickle
import re
from datetime import datetime

import numpy as np

try:
    import pandas as pd

    HAS_PANDAS = True
except Exception:
    HAS_PANDAS = False

try:
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except Exception:
    HAS_MATPLOTLIB = False

from utils.scaling_helpers import apply_min_max, reverse_min_max
from utils.plot_style import paper_plot_context
from utils.path_helpers import repo_root, resolve_repo_path


_TARGET_STAGE_CODE_MAP = {
    "failed": 0.0,
    "legacy_slack_target": 1.0,
    "refined_step_a": 2.0,
    "frozen_dhat_exact": 3.0,
    "frozen_dhat_bounded_fallback": 4.0,
    "stage1": 5.0,
    "stage2": 6.0,
    "margin_search": 7.0,
    "hold_previous": 8.0,
    "exception": 9.0,
}

_TARGET_STAGE_TICKS = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
_TARGET_STAGE_LABELS = [
    "failed",
    "legacy_slack_target",
    "refined_step_a",
    "frozen_dhat_exact",
    "frozen_dhat_bounded_fallback",
    "gart_stage1",
    "gart_stage2",
    "gart_margin_search",
    "gart_hold_previous",
    "gart_exception",
]

_SAFETY_PAPER_FILENAME_MAP = {
    "outputs_vs_ysp_vs_ys.png": "out_vs_ysp_ys.png",
    "ys_decomposition_summary.png": "ys_dec_sum.png",
    "outputs_vs_setpoint.png": "out_vs_sp.png",
    "outputs_vs_setpoint_projection.png": "out_vs_sp_proj.png",
    "applied_inputs.png": "u_applied.png",
    "u_applied_vs_us.png": "u_vs_us.png",
    "candidate_vs_safe_dev.png": "u_cand_vs_safe.png",
    "lyapunov_values.png": "lyap_vals.png",
    "first_step_contraction_diagnostics.png": "first_step_diag.png",
    "outputs_vs_ysp_vs_ys_last_episode.png": "out_vs_ysp_ys_last.png",
    "ys_decomposition_summary_last_episode.png": "ys_dec_sum_last.png",
    "y_minus_ys_last_episode.png": "y_minus_ys_last.png",
    "u_applied_vs_us_last_episode.png": "u_vs_us_last.png",
    "lyapunov_last_episode.png": "lyap_last.png",
    "lyapunov_margin.png": "lyap_margin.png",
    "target_selector_metrics.png": "target_metrics.png",
    "selector_objective_terms.png": "selector_obj_terms.png",
    "selector_x_penalties.png": "selector_x_pen.png",
    "selector_x_gaps.png": "selector_x_gaps.png",
    "selector_residuals.png": "selector_resid.png",
    "y_minus_ys.png": "y_minus_ys.png",
    "target_selector_status.png": "target_status.png",
    "qcqp_status.png": "qcqp_status.png",
    "reward_trace.png": "reward_trace.png",
    "reward_average_summary.png": "reward_avg_sum.png",
    "activation_contraction_counts.png": "activation_counts.png",
    "correction_modes.png": "correction_modes.png",
    "solver_status_counts.png": "solver_status_counts.png",
    "fallback_solver_status_counts.png": "fallback_solver_counts.png",
}

_SAFETY_COMPARISON_FILENAME_MAP = {
    "comparison_outputs_last_episode.png": "cmp_out_last.png",
    "comparison_inputs_last_episode.png": "cmp_in_last.png",
    "comparison_contraction_margin_last_episode.png": "cmp_lyap_last.png",
    "comparison_reward_mean.png": "cmp_reward.png",
    "comparison_reward_no_penalty_mean.png": "cmp_reward_no_penalty.png",
    "comparison_average_episode_reward.png": "cmp_avg_ep_reward.png",
    "comparison_output_rmse.png": "cmp_out_rmse.png",
    "comparison_rates.png": "cmp_rates.png",
    "comparison_correction_modes.png": "cmp_modes.png",
    "comparison_executed_action_gap_box.png": "cmp_gap_box.png",
    "comparison_fallback_count_per_episode_box.png": "cmp_fallback_box.png",
    "comparison_wall_clock_runtime.png": "cmp_wall_clock.png",
    "comparison_target_diagnostics.png": "cmp_target_diag.png",
}

_WINDOWS_PATH_SOFT_LIMIT = 240
_EXPORT_PROFILES = {"debug", "compact"}


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _is_csv_scalar(value):
    if value is None:
        return True
    if isinstance(value, (str, bool, int, float, np.bool_, np.integer, np.floating)):
        return True
    return False


def _scalar_only_records(rows):
    return [
        {key: value for key, value in row.items() if _is_csv_scalar(value)}
        for row in rows
    ]


_SAFETY_STEP_VECTOR_DETAIL_COLUMNS = {
    "selector_objective_terms",
    "selector_Qr_diag_used",
    "selector_R_u_ref_diag_used",
    "selector_R_delta_u_sel_diag_used",
    "selector_Q_delta_x_diag_used",
    "selector_Q_x_ref_diag_used",
    "selector_Qx_base_diag_used",
    "selector_Rdu_diag_used",
    "u_cand",
    "u_constrained_mpc",
    "u_safe",
    "u_prev",
    "u_s",
    "u_fallback_mpc",
    "target_u_ref",
    "target_u_ref_weight",
    "target_x_ref",
    "target_x_ref_weight",
    "mpc_tracking_target",
    "qcqp_tracking_target",
    "y_s",
    "r_s",
    "x_s",
    "d_s",
    "cx_s",
    "cd_d_s",
    "solver_residuals",
    "upstream_candidate_info",
}

_SAFETY_COMPACT_STEP_COLUMNS = [
    "step",
    "source",
    "accepted",
    "verified",
    "correction_mode",
    "safety_active",
    "diagnostic_only",
    "diagnostic_candidate_accepted",
    "diagnostic_unsafe",
    "diagnostic_unstable",
    "actual_intervention",
    "projection_active",
    "fallback_mpc_active",
    "fallback_verified",
    "fallback_solver_status",
    "reward",
    "reward_base",
    "reward_no_penalty",
    "reward_augmented",
    "fallback_penalty",
    "weighted_correction_gap",
    "reward_fallback_active",
    "solver_status",
    "qcqp_attempted",
    "qcqp_solved",
    "qcqp_hard_accepted",
    "constrained_mpc_attempted",
    "constrained_mpc_solved",
    "constrained_mpc_applied",
    "V_k",
    "V_next_first",
    "V_bound",
    "contraction_margin",
    "contraction_margin_applied",
    "first_step_contraction_satisfied",
    "first_step_contraction_satisfied_applied",
    "final_lyap_margin",
    "lyap_acceptance_mode",
    "target_success",
    "target_failure",
    "target_stage",
    "target_source",
    "effective_target_success",
    "effective_target_source",
    "effective_target_stage",
    "effective_target_reused",
    "target_mismatch_inf",
    "target_quality_ok",
    "target_quality_bypass",
    "target_quality_reason",
    "target_rate_inf",
    "target_error_inf",
    "target_slack_inf",
    "selector_status",
    "selector_stage",
    "target_cond_M",
    "target_cond_G",
    "target_residual_total_norm",
    "target_us_u_ref_inf",
    "target_xs_x_ref_inf",
    "selector_x_s_minus_xhat_inf",
    "selector_x_s_minus_xprev_inf",
    "selector_u_s_minus_uapplied_inf",
    "selector_u_s_minus_uprev_inf",
    "executed_action_gap_inf",
]


def _filter_safety_step_records(records, export_profile="debug"):
    profile = _normalize_export_profile(export_profile)
    scalar_records = _scalar_only_records(records)
    filtered = []
    for row in scalar_records:
        if profile == "compact":
            filtered.append({key: row.get(key) for key in _SAFETY_COMPACT_STEP_COLUMNS if key in row})
        else:
            filtered.append(
                {
                    key: value
                    for key, value in row.items()
                    if key not in _SAFETY_STEP_VECTOR_DETAIL_COLUMNS
                }
            )
    return filtered


def _safety_plot_filename(filename, *, paper_style=False):
    filename = str(filename)
    if not paper_style:
        return filename
    return _SAFETY_PAPER_FILENAME_MAP.get(filename, filename)


def _safety_plot_path(output_dir, filename, *, paper_style=False):
    os.makedirs(output_dir, exist_ok=True)
    return os.path.join(output_dir, _safety_plot_filename(filename, paper_style=paper_style))


def _max_joined_path_len(base_dir, rel_paths):
    return max(len(os.path.join(base_dir, rel_path)) for rel_path in rel_paths)


def _maybe_windows_short_filename(output_dir, filename, short_name_map):
    filename = str(filename)
    if os.name != "nt":
        return filename
    if len(os.path.join(output_dir, filename)) <= _WINDOWS_PATH_SOFT_LIMIT:
        return filename
    return short_name_map.get(filename, filename)


def _safety_plot_rel_paths(*, short_paths):
    state_target_dir = "st" if short_paths else "state_target_channels"
    last_xhat_dir = "lx_xs" if short_paths else "last_episode_xhat_vs_xs"
    last_decomp_dir = "ly_dec" if short_paths else "last_episode_ys_decomposition"
    episode_samples_dir = "ep_samples" if short_paths else "episode_samples_by_tens"
    last_episode_dir = "last_ep" if short_paths else "last_episode_summary"
    return [
        "summary.json",
        "arrays.npz",
        os.path.join(
            "plots",
            state_target_dir,
            last_decomp_dir,
            "ys_decomposition_0.png",
        ),
        os.path.join(
            "plots",
            state_target_dir,
            last_xhat_dir,
            "xhat_0_vs_x_s_eff_0.png",
        ),
        os.path.join(
            "plots",
            _safety_plot_filename(
                "outputs_vs_ysp_vs_ys_last_episode.png",
                paper_style=short_paths,
            ),
        ),
        os.path.join(
            "plots",
            episode_samples_dir,
            "ep_001_001_010.png" if short_paths else "episode_001_from_001_010.png",
        ),
        os.path.join(
            "plots",
            last_episode_dir,
            "ep_001_last.png" if short_paths else "episode_001_last.png",
        ),
    ]


def _array_or_none(info, key):
    value = info.get(key)
    if value is None:
        return None
    return np.asarray(value, float).reshape(-1)


def _stack_vectors(lyap_info_storage, key, width, fill_value=np.nan):
    out = np.full((len(lyap_info_storage), width), float(fill_value), dtype=float)
    for idx, info in enumerate(lyap_info_storage):
        arr = _array_or_none(info, key)
        if arr is None:
            continue
        use = min(width, arr.size)
        out[idx, :use] = arr[:use]
    return out


def _selector_debug(info):
    target_info = info.get("target_info", {})
    if not isinstance(target_info, dict):
        return {}
    selector = target_info.get("selector_debug", {})
    return selector if isinstance(selector, dict) else {}


def _target_info(info):
    target_info = info.get("target_info", {})
    return target_info if isinstance(target_info, dict) else {}


def _safe_nanmean(values):
    arr = np.asarray(values, dtype=float).reshape(-1)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return None
    return float(np.mean(finite))


def _safe_nanmax(values):
    arr = np.asarray(values, dtype=float).reshape(-1)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return None
    return float(np.max(finite))


def _safe_nanmin(values):
    arr = np.asarray(values, dtype=float).reshape(-1)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return None
    return float(np.min(finite))


def _safe_nansum(values):
    arr = np.asarray(values, dtype=float).reshape(-1)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return None
    return float(np.sum(finite))


def _normalize_export_profile(export_profile):
    profile = str(export_profile or "debug").strip().lower()
    if profile not in _EXPORT_PROFILES:
        raise ValueError(f"Unknown export_profile={export_profile!r}. Expected one of {sorted(_EXPORT_PROFILES)}.")
    return profile


def _sanitize_method_slug(value, *, default="method", max_len=80):
    text = str(value or default).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    if not text:
        text = default
    return _truncate_path_component(text, max_len=max_len)


def _moving_average(values, window):
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size == 0:
        return arr
    window = max(int(window), 1)
    finite = np.where(np.isfinite(arr), arr, np.nan)
    out = np.full(arr.shape, np.nan, dtype=float)
    for idx in range(arr.size):
        start = max(0, idx - window + 1)
        chunk = finite[start : idx + 1]
        valid = chunk[np.isfinite(chunk)]
        if valid.size:
            out[idx] = float(np.mean(valid))
    return out


def _bundle_reward_no_penalty_array(bundle, rewards):
    rewards = np.asarray(rewards, dtype=float).reshape(-1)
    raw = bundle.get("reward_no_penalty", bundle.get("reward_base", rewards))
    arr = np.asarray(raw, dtype=float).reshape(-1)
    if arr.size == rewards.size:
        return arr
    aligned = np.full_like(rewards, np.nan, dtype=float)
    n = min(aligned.size, arr.size)
    if n > 0:
        aligned[:n] = arr[:n]
    return aligned


def _safety_active_flags_from_bundle(bundle, *, tol=1.0e-10):
    u_safe = np.asarray(bundle.get("u_safe_dev_store", []), dtype=float)
    u_cand = np.asarray(bundle.get("u_cand_dev_store", []), dtype=float)
    n = max(u_safe.shape[0] if u_safe.ndim >= 1 else 0, u_cand.shape[0] if u_cand.ndim >= 1 else 0)
    if n <= 0:
        n = len(bundle.get("correction_modes", []))
    active = np.zeros(n, dtype=float)
    if u_safe.ndim == 2 and u_cand.ndim == 2 and u_safe.shape == u_cand.shape:
        active = np.maximum(active, (np.nanmax(np.abs(u_safe - u_cand), axis=1) > float(tol)).astype(float))
    projection = np.asarray(bundle.get("projection_active_flags", []), dtype=float).reshape(-1)
    fallback = np.asarray(bundle.get("fallback_verified_flags", []), dtype=float).reshape(-1)
    constrained = np.asarray(bundle.get("constrained_mpc_applied_flags", []), dtype=float).reshape(-1)
    for flags in (projection, fallback, constrained):
        use = min(active.size, flags.size)
        if use:
            active[:use] = np.maximum(active[:use], np.nan_to_num(flags[:use], nan=0.0))
    inactive_modes = {"", "none", "None", "accepted_candidate", "mpc_only_diagnostic_bypass"}
    for idx, mode in enumerate(bundle.get("correction_modes", [])[: active.size]):
        mode_text = str(mode)
        if mode_text not in inactive_modes:
            active[idx] = 1.0
    return active


def _diagnostic_safety_active_flags_from_bundle(bundle):
    diagnostic = np.asarray(bundle.get("diagnostic_unsafe_flags", []), dtype=float).reshape(-1)
    contraction = np.asarray(bundle.get("diagnostic_unstable_flags", []), dtype=float).reshape(-1)
    n = max(diagnostic.size, contraction.size)
    if n <= 0:
        return np.zeros(0, dtype=float)
    out = np.zeros(n, dtype=float)
    if diagnostic.size:
        out[: diagnostic.size] = np.maximum(out[: diagnostic.size], np.nan_to_num(diagnostic, nan=0.0))
    if contraction.size:
        out[: contraction.size] = np.maximum(out[: contraction.size], np.nan_to_num(contraction, nan=0.0))
    return out


def _shade_active_intervals(ax, x_values, active_flags, *, color="tab:red", alpha=0.08):
    active = np.asarray(active_flags, dtype=float).reshape(-1) > 0.5
    x_values = np.asarray(x_values, dtype=float).reshape(-1)
    n = min(active.size, x_values.size)
    if n <= 0:
        return
    start = None
    for idx in range(n):
        if active[idx] and start is None:
            start = idx
        is_last_active = active[idx] and (idx == n - 1 or not active[idx + 1])
        if is_last_active and start is not None:
            ax.axvspan(x_values[start], x_values[idx] + 1.0, color=color, alpha=alpha, linewidth=0)
            start = None


def make_safety_filter_step_records(lyap_info_storage):
    records = []
    for step_idx, info in enumerate(lyap_info_storage):
        target_info = _target_info(info)
        selector = _selector_debug(info)
        selector_terms = info.get("selector_objective_terms") or {}
        upstream_info = info.get("upstream_candidate_info", {})
        target_success = bool(info.get("target_success", False))
        correction_mode = str(info.get("correction_mode", ""))
        row = {
            "step": int(step_idx),
            "step_idx": int(step_idx),
            "source": info.get("source"),
            "accepted": bool(info.get("accepted", False)),
            "verified": bool(info.get("verified", False)),
            "accept_reason": info.get("accept_reason"),
            "reject_reason": info.get("reject_reason"),
            "correction_mode": info.get("correction_mode"),
            "projection_active": bool(correction_mode == "optimized_correction"),
            "safety_active": bool(
                correction_mode not in {"", "none", "None", "accepted_candidate", "mpc_only_diagnostic_bypass"}
            ),
            "diagnostic_only": bool(info.get("diagnostic_only", info.get("lyap_acceptance_mode") == "diagnostic_only")),
            "diagnostic_candidate_accepted": info.get("diagnostic_candidate_accepted"),
            "diagnostic_unsafe": bool(info.get("diagnostic_unsafe", False)),
            "diagnostic_unstable": bool(
                info.get(
                    "diagnostic_unstable",
                    False if info.get("candidate_lyap_ok") is None else not bool(info.get("candidate_lyap_ok")),
                )
            ),
            "actual_intervention": bool(
                info.get(
                    "actual_intervention",
                    correction_mode not in {"", "none", "None", "accepted_candidate", "mpc_only_diagnostic_bypass"},
                )
            ),
            "fallback_mpc_active": bool(correction_mode.startswith("fallback_mpc")),
            "reward": info.get("reward"),
            "reward_base": info.get("reward_base"),
            "reward_no_penalty": info.get("reward_no_penalty", info.get("reward_base")),
            "reward_augmented": info.get("reward_augmented"),
            "fallback_penalty": info.get("fallback_penalty"),
            "weighted_correction_gap": info.get("weighted_correction_gap"),
            "reward_fallback_active": bool(info.get("reward_fallback_active", False)),
            "policy_phase": info.get("policy_phase"),
            "behavior_policy_source": info.get("behavior_policy_source"),
            "behavior_noise_mode": info.get("behavior_noise_mode"),
            "behavior_exploration_sigma": info.get("behavior_exploration_sigma"),
            "training_update_mode": info.get("training_update_mode"),
            "critic_td_update_active": bool(info.get("critic_td_update_active", False)),
            "actor_bc_update_active": bool(info.get("actor_bc_update_active", False)),
            "td3_full_update_active": bool(info.get("td3_full_update_active", False)),
            "actor_bc_updates_per_step": info.get("actor_bc_updates_per_step"),
            "handoff_active": bool(info.get("handoff_active", False)),
            "handoff_alpha": info.get("handoff_alpha"),
            "bc_teacher_gap_inf": info.get("bc_teacher_gap_inf"),
            "handoff_candidate_gap_inf": info.get("handoff_candidate_gap_inf"),
            "candidate_bounds_ok": info.get("candidate_bounds_ok"),
            "candidate_move_ok": info.get("candidate_move_ok"),
            "candidate_lyap_ok": info.get("candidate_lyap_ok"),
            "candidate_first_step_lyap_ok": info.get("candidate_first_step_lyap_ok"),
            "first_step_contraction_triggered": bool(info.get("first_step_contraction_triggered", False)),
            "constrained_mpc_attempted": bool(info.get("constrained_mpc_attempted", False)),
            "constrained_mpc_solved": bool(info.get("constrained_mpc_solved", False)),
            "constrained_mpc_applied": bool(info.get("constrained_mpc_applied", False)),
            "constrained_mpc_failed_applied_candidate": bool(
                info.get("constrained_mpc_failed_applied_candidate", False)
            ),
            "qcqp_attempted": bool(info.get("qcqp_attempted", False)),
            "qcqp_solved": bool(info.get("qcqp_solved", False)),
            "qcqp_hard_accepted": bool(info.get("qcqp_hard_accepted", False)),
            "qcqp_status": info.get("qcqp_status"),
            "fallback_controller": info.get("fallback_controller"),
            "fallback_mode": info.get("fallback_mode"),
            "fallback_verified": info.get("fallback_verified"),
            "fallback_solver_status": info.get("fallback_solver_status"),
            "fallback_objective_value": info.get("fallback_objective_value"),
            "fallback_bounds_ok": info.get("fallback_bounds_ok"),
            "fallback_move_ok": info.get("fallback_move_ok"),
            "fallback_lyap_ok": info.get("fallback_lyap_ok"),
            "fallback_tracking_target_source": info.get("fallback_tracking_target_source"),
            "fallback_target_mismatch_inf": info.get("fallback_target_mismatch_inf"),
            "solver_status": info.get("solver_status"),
            "solver_name": info.get("solver_name"),
            "slack_v": info.get("slack_v"),
            "slack_u": info.get("slack_u"),
            "trust_region_violation": info.get("trust_region_violation"),
            "V_k": info.get("V_k"),
            "V_next_first": info.get("V_next_first"),
            "V_next_first_candidate": info.get("V_next_first_candidate"),
            "V_next_first_applied": info.get("V_next_first_applied"),
            "V_next_cand": info.get("V_next_cand"),
            "V_bound": info.get("V_bound"),
            "contraction_margin": info.get("contraction_margin"),
            "contraction_margin_candidate": info.get("contraction_margin_candidate"),
            "contraction_margin_applied": info.get("contraction_margin_applied"),
            "first_step_contraction_satisfied": info.get("first_step_contraction_satisfied"),
            "first_step_contraction_satisfied_applied": info.get("first_step_contraction_satisfied_applied"),
            "contraction_constraint_violation": info.get("contraction_constraint_violation"),
            "first_step_contraction_on": info.get("first_step_contraction_on"),
            "final_lyap_value": info.get("final_lyap_value"),
            "final_lyap_bound": info.get("final_lyap_bound"),
            "final_lyap_margin": info.get("final_lyap_margin"),
            "final_lyap_ok": info.get("final_lyap_ok"),
            "final_lyap_target_source": info.get("final_lyap_target_source"),
            "rho": info.get("rho"),
            "eps_lyap": info.get("eps_lyap"),
            "lyap_acceptance_mode": info.get("lyap_acceptance_mode"),
            "target_success": target_success,
            "target_failure": (not target_success),
            "target_stage": info.get("target_stage"),
            "target_generation_mode": info.get("target_generation_mode"),
            "target_source": info.get("target_source"),
            "selector_mode": info.get("selector_mode"),
            "effective_selector_mode": info.get("effective_selector_mode"),
            "selector_name": info.get("selector_name"),
            "effective_selector_name": info.get("effective_selector_name"),
            "current_target_success": info.get("current_target_success"),
            "current_target_stage": info.get("current_target_stage"),
            "effective_target_success": info.get("effective_target_success"),
            "effective_target_source": info.get("effective_target_source"),
            "effective_target_stage": info.get("effective_target_stage"),
            "effective_target_reused": info.get("effective_target_reused"),
            "d_s_minus_dhat_inf": info.get("d_s_minus_dhat_inf"),
            "d_s_frozen": info.get("d_s_frozen"),
            "d_s_optimized": info.get("d_s_optimized"),
            "box_solve_mode": info.get("box_solve_mode"),
            "exact_within_bounds": info.get("exact_within_bounds"),
            "exact_bound_violation_inf": info.get("exact_bound_violation_inf"),
            "bounded_residual_norm": info.get("bounded_residual_norm"),
            "selector_objective_value": info.get("selector_objective_value"),
            "selector_objective_terms": json.dumps(_jsonable(info.get("selector_objective_terms"))),
            "selector_term_target_tracking": selector_terms.get("target_tracking"),
            "selector_term_u_applied_anchor": selector_terms.get("u_applied_anchor"),
            "selector_term_u_prev_smoothing": selector_terms.get("u_prev_smoothing"),
            "selector_term_x_prev_smoothing": selector_terms.get("x_prev_smoothing"),
            "selector_term_xhat_anchor": selector_terms.get("xhat_anchor"),
            "backup_target_available": info.get("backup_target_available"),
            "mpc_tracking_target_source": info.get("mpc_tracking_target_source"),
            "qcqp_tracking_target_source": info.get("qcqp_tracking_target_source"),
            "target_mismatch_inf": info.get("target_mismatch_inf"),
            "target_quality_enabled": info.get("target_quality_enabled"),
            "target_quality_ok": info.get("target_quality_ok"),
            "target_quality_reason": info.get("target_quality_reason"),
            "target_quality_policy": info.get("target_quality_policy"),
            "target_quality_bypass": info.get("target_quality_bypass"),
            "target_quality_mismatch_inf": info.get("target_quality_mismatch_inf"),
            "target_quality_residual_norm": info.get("target_quality_residual_norm"),
            "target_rate_inf": info.get("target_rate_inf"),
            "performance_guard_enabled": info.get("performance_guard_enabled"),
            "performance_guard_ok": info.get("performance_guard_ok"),
            "performance_guard_reference_policy": info.get("performance_guard_reference_policy"),
            "performance_guard_candidate_cost": info.get("performance_guard_candidate_cost"),
            "performance_guard_reference_cost": info.get("performance_guard_reference_cost"),
            "performance_guard_tolerance": info.get("performance_guard_tolerance"),
            "residual_rl_enabled": info.get("residual_rl_enabled"),
            "residual_rl_baseline_policy": info.get("residual_rl_baseline_policy"),
            "residual_rl_authority_multiplier": info.get("residual_rl_authority_multiplier"),
            "target_error_inf": target_info.get("target_error_inf"),
            "target_slack_inf": target_info.get("target_slack_inf"),
            "selector_status": selector.get("status"),
            "selector_solver": selector.get("solver"),
            "selector_stage": target_info.get("solve_stage"),
            "selector_warm_start_enabled": info.get("selector_warm_start_enabled"),
            "selector_warm_start_available": info.get("selector_warm_start_available"),
            "selector_warm_start_used": info.get("selector_warm_start_used"),
            "selector_prev_input_term_active": info.get("selector_prev_input_term_active"),
            "selector_prev_state_term_active": info.get("selector_prev_state_term_active"),
            "target_cond_M": info.get("target_cond_M"),
            "target_cond_G": info.get("target_cond_G"),
            "target_residual_total_norm": info.get("target_residual_total_norm"),
            "target_u_ref_active": info.get("target_u_ref_active"),
            "target_u_ref_penalty": info.get("target_u_ref_penalty"),
            "target_us_u_ref_inf": info.get("target_us_u_ref_inf"),
            "target_x_ref_active": info.get("target_x_ref_active"),
            "target_x_ref_penalty": info.get("target_x_ref_penalty"),
            "target_xs_x_ref_inf": info.get("target_xs_x_ref_inf"),
            "selector_x_s_minus_xhat_inf": None if target_info.get("x_s_minus_xhat") is None else float(np.max(np.abs(np.asarray(target_info.get("x_s_minus_xhat"), float).reshape(-1)))),
            "selector_x_s_minus_xprev_inf": None if target_info.get("x_s_minus_x_prev") is None else float(np.max(np.abs(np.asarray(target_info.get("x_s_minus_x_prev"), float).reshape(-1)))),
            "selector_u_s_minus_uapplied_inf": None if target_info.get("u_s_minus_u_applied") is None else float(np.max(np.abs(np.asarray(target_info.get("u_s_minus_u_applied"), float).reshape(-1)))),
            "selector_u_s_minus_uprev_inf": None if target_info.get("u_s_minus_u_prev") is None else float(np.max(np.abs(np.asarray(target_info.get("u_s_minus_u_prev"), float).reshape(-1)))),
            "selector_Qr_diag_used": json.dumps(_jsonable(info.get("selector_Qr_diag_used"))),
            "selector_R_u_ref_diag_used": json.dumps(_jsonable(info.get("selector_R_u_ref_diag_used"))),
            "selector_R_delta_u_sel_diag_used": json.dumps(_jsonable(info.get("selector_R_delta_u_sel_diag_used"))),
            "selector_Q_delta_x_diag_used": json.dumps(_jsonable(info.get("selector_Q_delta_x_diag_used"))),
            "selector_Q_x_ref_diag_used": json.dumps(_jsonable(info.get("selector_Q_x_ref_diag_used"))),
            "selector_Qx_base_diag_used": json.dumps(_jsonable(info.get("selector_Qx_base_diag_used"))),
            "selector_Rdu_diag_used": json.dumps(_jsonable(info.get("selector_Rdu_diag_used"))),
            "allow_trust_region_slack": info.get("allow_trust_region_slack"),
            "setpoint_changed": info.get("setpoint_changed"),
            "u_cand": json.dumps(_jsonable(info.get("u_cand"))),
            "u_constrained_mpc": json.dumps(_jsonable(info.get("u_constrained_mpc"))),
            "u_safe": json.dumps(_jsonable(info.get("u_safe"))),
            "u_prev": json.dumps(_jsonable(info.get("u_prev"))),
            "u_s": json.dumps(_jsonable(info.get("u_s"))),
            "u_fallback_mpc": json.dumps(_jsonable(info.get("u_fallback_mpc"))),
            "behavior_action_pre_filter": json.dumps(_jsonable(info.get("behavior_action_pre_filter"))),
            "teacher_action_pre_filter": json.dumps(_jsonable(info.get("teacher_action_pre_filter"))),
            "teacher_u_dev_pre_filter": json.dumps(_jsonable(info.get("teacher_u_dev_pre_filter"))),
            "policy_action_pre_handoff": json.dumps(_jsonable(info.get("policy_action_pre_handoff"))),
            "policy_u_dev_pre_handoff": json.dumps(_jsonable(info.get("policy_u_dev_pre_handoff"))),
            "target_u_ref": json.dumps(_jsonable(info.get("target_u_ref"))),
            "target_u_ref_weight": json.dumps(_jsonable(info.get("target_u_ref_weight"))),
            "target_x_ref": json.dumps(_jsonable(info.get("target_x_ref"))),
            "target_x_ref_weight": json.dumps(_jsonable(info.get("target_x_ref_weight"))),
            "mpc_tracking_target": json.dumps(_jsonable(info.get("mpc_tracking_target"))),
            "qcqp_tracking_target": json.dumps(_jsonable(info.get("qcqp_tracking_target"))),
            "y_s": json.dumps(_jsonable(info.get("y_s"))),
            "r_s": json.dumps(_jsonable(info.get("r_s"))),
            "x_s": json.dumps(_jsonable(info.get("x_s"))),
            "d_s": json.dumps(_jsonable(info.get("d_s"))),
            "cx_s": json.dumps(_jsonable(info.get("cx_s"))),
            "cd_d_s": json.dumps(_jsonable(info.get("cd_d_s"))),
            "executed_action_gap_inf": None
            if info.get("u_safe") is None or info.get("u_cand") is None
            else float(
                np.max(
                    np.abs(
                        np.asarray(info.get("u_safe"), float).reshape(-1)
                        - np.asarray(info.get("u_cand"), float).reshape(-1)
                    )
                )
            ),
            "solver_residuals": json.dumps(_jsonable(info.get("solver_residuals", {}))),
            "upstream_candidate_info": json.dumps(_jsonable(upstream_info)),
        }
        records.append(row)
    return records


def make_safety_filter_df(lyap_info_storage):
    records = make_safety_filter_step_records(lyap_info_storage)
    if not HAS_PANDAS:
        raise ImportError("pandas is required to build a DataFrame.")
    return pd.DataFrame(records)


def make_lyap_df(lyap_info_storage, slack_thr=1e-9, du_thr=1e-10):
    df = make_safety_filter_df(lyap_info_storage)
    if "slack_v" in df.columns:
        df["slack_v_active"] = df["slack_v"].fillna(0.0).astype(float) > float(slack_thr)
    if "u_cand" in df.columns and "u_safe" in df.columns:
        df["du_filter_active"] = df.apply(
            lambda row: (
                np.max(
                    np.abs(
                        np.asarray(json.loads(row["u_safe"]), float)
                        - np.asarray(json.loads(row["u_cand"]), float)
                    )
                )
                > float(du_thr)
            ),
            axis=1,
        )
    return df


def summarize_safety_filter_bundle(bundle):
    lyap_info_storage = bundle["lyap_info_storage"]
    modes = [str(info.get("correction_mode", "none")) for info in lyap_info_storage]
    solver_statuses = [str(info.get("solver_status")) for info in lyap_info_storage if info.get("solver_status") is not None]
    fallback_statuses = [
        str(info.get("fallback_solver_status"))
        for info in lyap_info_storage
        if info.get("fallback_solver_status") is not None
    ]
    summary = {
        "source": bundle.get("source"),
        "n_steps": int(bundle.get("nFE", len(lyap_info_storage))),
        "n_verified": int(sum(bool(info.get("verified", False)) for info in lyap_info_storage)),
        "verified_rate": float(
            np.mean([1.0 if bool(info.get("verified", False)) else 0.0 for info in lyap_info_storage])
        )
        if lyap_info_storage
        else None,
        "accepted_rate": float(
            np.mean([1.0 if str(info.get("correction_mode", "")) == "accepted_candidate" else 0.0 for info in lyap_info_storage])
        )
        if lyap_info_storage
        else None,
        "n_target_success": int(sum(bool(info.get("current_target_success", info.get("target_success", False))) for info in lyap_info_storage)),
        "n_effective_target_success": int(sum(bool(info.get("effective_target_success", False)) for info in lyap_info_storage)),
        "n_target_reused": int(sum(bool(info.get("effective_target_reused", False)) for info in lyap_info_storage)),
        "n_accepted_candidate": int(sum(mode == "accepted_candidate" for mode in modes)),
        "n_optimized_correction": int(sum(mode == "optimized_correction" for mode in modes)),
        "n_fallback_mpc_verified": int(sum(mode == "fallback_mpc_verified" for mode in modes)),
        "n_fallback_mpc_unverified": int(sum(mode == "fallback_mpc_unverified" for mode in modes)),
        "n_target_fail_hold_prev": int(sum(mode == "target_fail_hold_prev" for mode in modes)),
        "n_solver_fail_hold_prev": int(sum(mode == "solver_fail_hold_prev" for mode in modes)),
        "n_secondary_fallbacks": int(sum(mode.endswith("_secondary") for mode in modes)),
        "n_target_failures": int(sum(not bool(info.get("current_target_success", info.get("target_success", False))) for info in lyap_info_storage)),
        "n_qcqp_attempted": int(sum(bool(info.get("qcqp_attempted", False)) for info in lyap_info_storage)),
        "n_qcqp_solved": int(sum(bool(info.get("qcqp_solved", False)) for info in lyap_info_storage)),
        "n_qcqp_hard_accepted": int(sum(bool(info.get("qcqp_hard_accepted", False)) for info in lyap_info_storage)),
        "n_first_step_contraction_triggered": int(sum(bool(info.get("first_step_contraction_triggered", False)) for info in lyap_info_storage)),
        "n_constrained_mpc_attempted": int(sum(bool(info.get("constrained_mpc_attempted", False)) for info in lyap_info_storage)),
        "n_constrained_mpc_solved": int(sum(bool(info.get("constrained_mpc_solved", False)) for info in lyap_info_storage)),
        "n_constrained_mpc_applied": int(sum(bool(info.get("constrained_mpc_applied", False)) for info in lyap_info_storage)),
        "n_constrained_mpc_failed_applied_candidate": int(sum(bool(info.get("constrained_mpc_failed_applied_candidate", False)) for info in lyap_info_storage)),
        "reward_mean": float(np.mean(bundle["rewards"])) if len(bundle["rewards"]) > 0 else None,
        "reward_sum": float(np.sum(bundle["rewards"])) if len(bundle["rewards"]) > 0 else None,
        "reward_min": float(np.min(bundle["rewards"])) if len(bundle["rewards"]) > 0 else None,
        "reward_max": float(np.max(bundle["rewards"])) if len(bundle["rewards"]) > 0 else None,
        "reward_base_mean": _safe_nanmean(bundle.get("reward_base", [])),
        "reward_base_sum": _safe_nansum(bundle.get("reward_base", [])),
        "reward_no_penalty_mean": _safe_nanmean(bundle.get("reward_no_penalty", bundle.get("reward_base", []))),
        "reward_no_penalty_sum": _safe_nansum(bundle.get("reward_no_penalty", bundle.get("reward_base", []))),
        "reward_augmented_mean": _safe_nanmean(bundle.get("reward_augmented", [])),
        "reward_augmented_sum": _safe_nansum(bundle.get("reward_augmented", [])),
        "fallback_penalty_mean": _safe_nanmean(bundle.get("fallback_penalty", [])),
        "fallback_penalty_max": _safe_nanmax(bundle.get("fallback_penalty", [])),
        "fallback_penalty_sum": _safe_nansum(bundle.get("fallback_penalty", [])),
        "weighted_correction_gap_mean": _safe_nanmean(bundle.get("weighted_correction_gap", [])),
        "weighted_correction_gap_max": _safe_nanmax(bundle.get("weighted_correction_gap", [])),
        "diagnostic_unsafe_rate": _safe_nanmean(bundle.get("diagnostic_unsafe_flags", [])),
        "diagnostic_unstable_rate": _safe_nanmean(bundle.get("diagnostic_unstable_flags", [])),
        "diagnostic_safety_active_rate": _safe_nanmean(bundle.get("diagnostic_safety_active_flags", [])),
        "actual_intervention_rate": _safe_nanmean(bundle.get("actual_intervention_flags", [])),
        "target_error_inf_max": float(np.nanmax(bundle["target_error_inf"])) if bundle["target_error_inf"].size > 0 else None,
        "target_slack_inf_max": float(np.nanmax(bundle["target_slack_inf"])) if bundle["target_slack_inf"].size > 0 else None,
        "lyapunov_margin_min": float(np.nanmin(bundle["lyapunov_margin"])) if bundle["lyapunov_margin"].size > 0 else None,
        "target_mismatch_inf_max": float(np.nanmax(bundle["target_mismatch_inf"])) if bundle["target_mismatch_inf"].size > 0 else None,
        "target_quality_ok_rate": _safe_nanmean(bundle.get("target_quality_ok_flags", [])),
        "target_quality_bypass_rate": _safe_nanmean(bundle.get("target_quality_bypass_flags", [])),
        "target_rate_inf_max": _safe_nanmax(bundle.get("target_rate_inf", [])),
        "performance_guard_ok_rate": _safe_nanmean(bundle.get("performance_guard_ok_flags", [])),
        "d_s_minus_dhat_inf_max": float(np.nanmax(bundle["d_s_minus_dhat_inf"])) if bundle["d_s_minus_dhat_inf"].size > 0 else None,
        "target_cond_M_max": _safe_nanmax(bundle.get("target_cond_M", [])),
        "target_cond_G_max": _safe_nanmax(bundle.get("target_cond_G", [])),
        "target_residual_total_norm_max": _safe_nanmax(bundle.get("target_residual_total_norm", [])),
        "target_u_ref_active_steps": int(np.nansum(bundle.get("target_u_ref_active_flags", []))),
        "target_us_u_ref_inf_mean": _safe_nanmean(bundle.get("target_us_u_ref_inf", [])),
        "target_us_u_ref_inf_max": _safe_nanmax(bundle.get("target_us_u_ref_inf", [])),
        "target_x_ref_active_steps": int(np.nansum(bundle.get("target_x_ref_active_flags", []))),
        "target_xs_x_ref_inf_mean": _safe_nanmean(bundle.get("target_xs_x_ref_inf", [])),
        "target_xs_x_ref_inf_max": _safe_nanmax(bundle.get("target_xs_x_ref_inf", [])),
        "executed_action_gap_inf_mean": _safe_nanmean(bundle.get("executed_action_gap_inf", [])),
        "executed_action_gap_inf_max": _safe_nanmax(bundle.get("executed_action_gap_inf", [])),
        "handoff_active_rate": _safe_nanmean(bundle.get("handoff_active_flags", [])),
        "bc_teacher_gap_inf_mean": _safe_nanmean(bundle.get("bc_teacher_gap_inf", [])),
        "bc_teacher_gap_inf_max": _safe_nanmax(bundle.get("bc_teacher_gap_inf", [])),
    }
    timing = {}
    if isinstance(bundle.get("extra"), dict) and isinstance(bundle["extra"].get("timing"), dict):
        timing.update(bundle["extra"]["timing"])
    if isinstance(bundle.get("config"), dict):
        for key in (
            "wall_clock_seconds",
            "wall_clock_seconds_per_episode",
            "wall_clock_seconds_per_step",
            "wall_clock_steps_per_second",
            "wall_clock_n_steps",
            "wall_clock_n_episodes",
        ):
            if key in bundle["config"]:
                timing[key] = bundle["config"][key]
    for key, value in timing.items():
        summary[key] = value
    summary["fallback_rate"] = None if summary["n_steps"] <= 0 else float(
        (summary["n_fallback_mpc_verified"] + summary["n_fallback_mpc_unverified"]) / float(summary["n_steps"])
    )
    summary["mode_counts"] = {mode: int(modes.count(mode)) for mode in sorted(set(modes))}
    summary["solver_status_counts"] = {
        status: int(solver_statuses.count(status)) for status in sorted(set(solver_statuses))
    }
    summary["fallback_solver_status_counts"] = {
        status: int(fallback_statuses.count(status)) for status in sorted(set(fallback_statuses))
    }
    return summary


def build_safety_filter_run_bundle(
    source,
    results,
    steady_states=None,
    config=None,
    min_max_dict=None,
    data_min=None,
    data_max=None,
    extra=None,
):
    (
        y_system,
        u_applied_phys,
        avg_rewards,
        rewards,
        xhatdhat,
        nFE,
        time_in_sub_episodes,
        y_sp,
        yhat,
        e_store,
        qi,
        qs,
        ha,
        lyap_info_storage,
        u_safe_dev_store,
    ) = results

    y_system = np.asarray(y_system, float)
    u_applied_phys = np.asarray(u_applied_phys, float)
    rewards = np.asarray(rewards, float)
    xhatdhat = np.asarray(xhatdhat, float)
    y_sp = np.asarray(y_sp, float)
    yhat = np.asarray(yhat, float)
    e_store = np.asarray(e_store, float)
    qi = np.asarray(qi, float)
    qs = np.asarray(qs, float)
    ha = np.asarray(ha, float)
    u_safe_dev_store = np.asarray(u_safe_dev_store, float)

    n_u = u_safe_dev_store.shape[1]
    n_y = y_system.shape[1]

    bundle = {
        "source": str(source),
        "config": {} if config is None else config,
        "steady_states": steady_states,
        "min_max_dict": min_max_dict,
        "data_min": None if data_min is None else np.asarray(data_min, float),
        "data_max": None if data_max is None else np.asarray(data_max, float),
        "nFE": int(nFE),
        "time_in_sub_episodes": int(time_in_sub_episodes),
        "avg_rewards": list(avg_rewards),
        "rewards": rewards.copy(),
        "y_system": y_system.copy(),
        "u_applied_phys": u_applied_phys.copy(),
        "xhatdhat": xhatdhat.copy(),
        "y_sp": y_sp.copy(),
        "yhat": yhat.copy(),
        "e_store": e_store.copy(),
        "qi": qi.copy(),
        "qs": qs.copy(),
        "ha": ha.copy(),
        "lyap_info_storage": lyap_info_storage,
        "u_safe_dev_store": u_safe_dev_store.copy(),
        "u_cand_dev_store": _stack_vectors(lyap_info_storage, "u_cand", n_u),
        "u_prev_dev_store": _stack_vectors(lyap_info_storage, "u_prev", n_u),
        "behavior_action_pre_filter_store": _stack_vectors(lyap_info_storage, "behavior_action_pre_filter", n_u),
        "teacher_action_pre_filter_store": _stack_vectors(lyap_info_storage, "teacher_action_pre_filter", n_u),
        "teacher_u_dev_pre_filter_store": _stack_vectors(lyap_info_storage, "teacher_u_dev_pre_filter", n_u),
        "policy_action_pre_handoff_store": _stack_vectors(lyap_info_storage, "policy_action_pre_handoff", n_u),
        "policy_u_dev_pre_handoff_store": _stack_vectors(lyap_info_storage, "policy_u_dev_pre_handoff", n_u),
        "u_target_dev_store": _stack_vectors(lyap_info_storage, "u_s", n_u),
        "u_fallback_mpc_dev_store": _stack_vectors(lyap_info_storage, "u_fallback_mpc", n_u),
        "x_target_store": _stack_vectors(lyap_info_storage, "x_s", xhatdhat.shape[0] - n_y),
        "d_target_store": _stack_vectors(lyap_info_storage, "d_s", n_y),
        "y_target_store": _stack_vectors(lyap_info_storage, "y_s", n_y),
        "target_u_ref_store": _stack_vectors(lyap_info_storage, "target_u_ref", n_u),
        "target_u_ref_weight_store": _stack_vectors(lyap_info_storage, "target_u_ref_weight", n_u),
        "target_x_ref_store": _stack_vectors(lyap_info_storage, "target_x_ref", xhatdhat.shape[0] - n_y),
        "target_x_ref_weight_store": _stack_vectors(lyap_info_storage, "target_x_ref_weight", xhatdhat.shape[0] - n_y),
        "r_target_store": _stack_vectors(lyap_info_storage, "r_s", y_sp.shape[1]),
        "cx_s_store": _stack_vectors(lyap_info_storage, "cx_s", n_y),
        "cd_d_s_store": _stack_vectors(lyap_info_storage, "cd_d_s", n_y),
        "reward_base": np.array([info.get("reward_base", np.nan) for info in lyap_info_storage], dtype=float),
        "reward_no_penalty": np.array(
            [info.get("reward_no_penalty", info.get("reward_base", np.nan)) for info in lyap_info_storage],
            dtype=float,
        ),
        "reward_augmented": np.array([info.get("reward_augmented", info.get("reward", np.nan)) for info in lyap_info_storage], dtype=float),
        "fallback_penalty": np.array([info.get("fallback_penalty", 0.0) for info in lyap_info_storage], dtype=float),
        "weighted_correction_gap": np.array([info.get("weighted_correction_gap", 0.0) for info in lyap_info_storage], dtype=float),
        "reward_fallback_active_flags": np.array(
            [1.0 if bool(info.get("reward_fallback_active", False)) else 0.0 for info in lyap_info_storage],
            dtype=float,
        ),
        "selector_target_tracking_term": np.array(
            [
                (info.get("selector_objective_terms") or {}).get("target_tracking", np.nan)
                for info in lyap_info_storage
            ],
            dtype=float,
        ),
        "selector_u_applied_anchor_term": np.array(
            [
                (info.get("selector_objective_terms") or {}).get("u_applied_anchor", np.nan)
                for info in lyap_info_storage
            ],
            dtype=float,
        ),
        "selector_u_prev_smoothing_term": np.array(
            [
                (info.get("selector_objective_terms") or {}).get("u_prev_smoothing", np.nan)
                for info in lyap_info_storage
            ],
            dtype=float,
        ),
        "selector_x_prev_smoothing_term": np.array(
            [
                (info.get("selector_objective_terms") or {}).get("x_prev_smoothing", np.nan)
                for info in lyap_info_storage
            ],
            dtype=float,
        ),
        "selector_xhat_anchor_term": np.array(
            [
                (info.get("selector_objective_terms") or {}).get("xhat_anchor", np.nan)
                for info in lyap_info_storage
            ],
            dtype=float,
        ),
        "selector_objective_value": np.array(
            [info.get("selector_objective_value", np.nan) for info in lyap_info_storage],
            dtype=float,
        ),
        "selector_x_s_minus_xhat_inf": np.array(
            [
                np.nan if info.get("target_info", {}).get("x_s_minus_xhat") is None
                else float(np.max(np.abs(np.asarray(info.get("target_info", {}).get("x_s_minus_xhat"), float).reshape(-1))))
                for info in lyap_info_storage
            ],
            dtype=float,
        ),
        "selector_x_s_minus_xprev_inf": np.array(
            [
                np.nan if info.get("target_info", {}).get("x_s_minus_x_prev") is None
                else float(np.max(np.abs(np.asarray(info.get("target_info", {}).get("x_s_minus_x_prev"), float).reshape(-1))))
                for info in lyap_info_storage
            ],
            dtype=float,
        ),
        "V_k": np.array([info.get("V_k", np.nan) for info in lyap_info_storage], dtype=float),
        "V_next_first": np.array([info.get("V_next_first", np.nan) for info in lyap_info_storage], dtype=float),
        "V_next_first_candidate": np.array([info.get("V_next_first_candidate", np.nan) for info in lyap_info_storage], dtype=float),
        "V_next_first_applied": np.array([info.get("V_next_first_applied", np.nan) for info in lyap_info_storage], dtype=float),
        "V_next_cand": np.array([info.get("V_next_cand", np.nan) for info in lyap_info_storage], dtype=float),
        "V_bound": np.array([info.get("V_bound", np.nan) for info in lyap_info_storage], dtype=float),
        "contraction_margin": np.array([info.get("contraction_margin", np.nan) for info in lyap_info_storage], dtype=float),
        "contraction_margin_candidate": np.array([info.get("contraction_margin_candidate", np.nan) for info in lyap_info_storage], dtype=float),
        "contraction_margin_applied": np.array([info.get("contraction_margin_applied", np.nan) for info in lyap_info_storage], dtype=float),
        "first_step_contraction_satisfied_flags": np.array(
            [1.0 if bool(info.get("first_step_contraction_satisfied", False)) else 0.0 for info in lyap_info_storage],
            dtype=float,
        ),
        "first_step_contraction_satisfied_applied_flags": np.array(
            [1.0 if bool(info.get("first_step_contraction_satisfied_applied", False)) else 0.0 for info in lyap_info_storage],
            dtype=float,
        ),
        "final_lyap_value": np.array([info.get("final_lyap_value", np.nan) for info in lyap_info_storage], dtype=float),
        "final_lyap_bound": np.array([info.get("final_lyap_bound", np.nan) for info in lyap_info_storage], dtype=float),
        "final_lyap_margin": np.array([info.get("final_lyap_margin", np.nan) for info in lyap_info_storage], dtype=float),
        "final_lyap_target_source": [info.get("final_lyap_target_source") for info in lyap_info_storage],
        "lyapunov_margin": np.array(
            [
                np.nan
                if info.get("V_next_cand") is None or info.get("V_bound") is None
                else float(info.get("V_bound")) - float(info.get("V_next_cand"))
                for info in lyap_info_storage
            ],
            dtype=float,
        ),
        "target_error_inf": np.array(
            [info.get("target_info", {}).get("target_error_inf", np.nan) for info in lyap_info_storage],
            dtype=float,
        ),
        "target_slack_inf": np.array(
            [info.get("target_info", {}).get("target_slack_inf", np.nan) for info in lyap_info_storage],
            dtype=float,
        ),
        "selector_dyn_residual_inf": np.array(
            [info.get("target_info", {}).get("dyn_residual_inf", np.nan) for info in lyap_info_storage],
            dtype=float,
        ),
        "selector_bound_violation_inf": np.array(
            [info.get("target_info", {}).get("bound_violation_inf", np.nan) for info in lyap_info_storage],
            dtype=float,
        ),
        "target_mismatch_inf": np.array(
            [info.get("target_mismatch_inf", np.nan) for info in lyap_info_storage],
            dtype=float,
        ),
        "target_quality_ok_flags": np.array(
            [
                np.nan if info.get("target_quality_ok") is None else (1.0 if bool(info.get("target_quality_ok")) else 0.0)
                for info in lyap_info_storage
            ],
            dtype=float,
        ),
        "target_quality_bypass_flags": np.array(
            [1.0 if bool(info.get("target_quality_bypass", False)) else 0.0 for info in lyap_info_storage],
            dtype=float,
        ),
        "target_quality_mismatch_inf": np.array(
            [info.get("target_quality_mismatch_inf", np.nan) for info in lyap_info_storage],
            dtype=float,
        ),
        "target_quality_residual_norm": np.array(
            [info.get("target_quality_residual_norm", np.nan) for info in lyap_info_storage],
            dtype=float,
        ),
        "target_rate_inf": np.array([info.get("target_rate_inf", np.nan) for info in lyap_info_storage], dtype=float),
        "performance_guard_ok_flags": np.array(
            [
                np.nan if info.get("performance_guard_ok") is None else (1.0 if bool(info.get("performance_guard_ok")) else 0.0)
                for info in lyap_info_storage
            ],
            dtype=float,
        ),
        "performance_guard_candidate_cost": np.array(
            [info.get("performance_guard_candidate_cost", np.nan) for info in lyap_info_storage],
            dtype=float,
        ),
        "performance_guard_reference_cost": np.array(
            [info.get("performance_guard_reference_cost", np.nan) for info in lyap_info_storage],
            dtype=float,
        ),
        "target_cond_M": np.array(
            [info.get("target_cond_M", np.nan) for info in lyap_info_storage],
            dtype=float,
        ),
        "target_cond_G": np.array(
            [info.get("target_cond_G", np.nan) for info in lyap_info_storage],
            dtype=float,
        ),
        "target_residual_total_norm": np.array(
            [info.get("target_residual_total_norm", np.nan) for info in lyap_info_storage],
            dtype=float,
        ),
        "target_u_ref_active_flags": np.array(
            [1.0 if bool(info.get("target_u_ref_active", False)) else 0.0 for info in lyap_info_storage],
            dtype=float,
        ),
        "target_x_ref_active_flags": np.array(
            [1.0 if bool(info.get("target_x_ref_active", False)) else 0.0 for info in lyap_info_storage],
            dtype=float,
        ),
        "target_us_u_ref_inf": np.array(
            [info.get("target_us_u_ref_inf", np.nan) for info in lyap_info_storage],
            dtype=float,
        ),
        "target_xs_x_ref_inf": np.array(
            [info.get("target_xs_x_ref_inf", np.nan) for info in lyap_info_storage],
            dtype=float,
        ),
        "executed_action_gap_inf": np.array(
            [
                np.nan
                if info.get("u_safe") is None or info.get("u_cand") is None
                else float(
                    np.max(
                        np.abs(
                            np.asarray(info.get("u_safe"), float).reshape(-1)
                            - np.asarray(info.get("u_cand"), float).reshape(-1)
                        )
                    )
                )
                for info in lyap_info_storage
            ],
            dtype=float,
        ),
        "target_success_flags": np.array(
            [1.0 if bool(info.get("target_success", False)) else 0.0 for info in lyap_info_storage],
            dtype=float,
        ),
        "target_failure_flags": np.array(
            [0.0 if bool(info.get("target_success", False)) else 1.0 for info in lyap_info_storage],
            dtype=float,
        ),
        "target_stage_code": np.array(
            [
                _TARGET_STAGE_CODE_MAP.get(
                    str(info.get("target_stage", "")),
                    _TARGET_STAGE_CODE_MAP["failed"],
                )
                for info in lyap_info_storage
            ],
            dtype=float,
        ),
        "effective_target_success_flags": np.array(
            [1.0 if bool(info.get("effective_target_success", False)) else 0.0 for info in lyap_info_storage],
            dtype=float,
        ),
        "effective_target_reused_flags": np.array(
            [1.0 if bool(info.get("effective_target_reused", False)) else 0.0 for info in lyap_info_storage],
            dtype=float,
        ),
        "d_s_minus_dhat_inf": np.array(
            [info.get("d_s_minus_dhat_inf", np.nan) for info in lyap_info_storage],
            dtype=float,
        ),
        "qcqp_attempted_flags": np.array(
            [1.0 if bool(info.get("qcqp_attempted", False)) else 0.0 for info in lyap_info_storage],
            dtype=float,
        ),
        "qcqp_solved_flags": np.array(
            [1.0 if bool(info.get("qcqp_solved", False)) else 0.0 for info in lyap_info_storage],
            dtype=float,
        ),
        "qcqp_hard_accepted_flags": np.array(
            [1.0 if bool(info.get("qcqp_hard_accepted", False)) else 0.0 for info in lyap_info_storage],
            dtype=float,
        ),
        "candidate_first_step_lyap_ok_flags": np.array(
            [np.nan if info.get("candidate_first_step_lyap_ok") is None else (1.0 if bool(info.get("candidate_first_step_lyap_ok")) else 0.0) for info in lyap_info_storage],
            dtype=float,
        ),
        "first_step_contraction_triggered_flags": np.array(
            [1.0 if bool(info.get("first_step_contraction_triggered", False)) else 0.0 for info in lyap_info_storage],
            dtype=float,
        ),
        "constrained_mpc_attempted_flags": np.array(
            [1.0 if bool(info.get("constrained_mpc_attempted", False)) else 0.0 for info in lyap_info_storage],
            dtype=float,
        ),
        "constrained_mpc_solved_flags": np.array(
            [1.0 if bool(info.get("constrained_mpc_solved", False)) else 0.0 for info in lyap_info_storage],
            dtype=float,
        ),
        "constrained_mpc_applied_flags": np.array(
            [1.0 if bool(info.get("constrained_mpc_applied", False)) else 0.0 for info in lyap_info_storage],
            dtype=float,
        ),
        "constrained_mpc_failed_applied_candidate_flags": np.array(
            [1.0 if bool(info.get("constrained_mpc_failed_applied_candidate", False)) else 0.0 for info in lyap_info_storage],
            dtype=float,
        ),
        "verified_flags": np.array(
            [1.0 if bool(info.get("verified", False)) else 0.0 for info in lyap_info_storage],
            dtype=float,
        ),
        "accepted_flags": np.array(
            [1.0 if bool(info.get("accepted", False)) else 0.0 for info in lyap_info_storage],
            dtype=float,
        ),
        "diagnostic_only_flags": np.array(
            [1.0 if bool(info.get("diagnostic_only", info.get("lyap_acceptance_mode") == "diagnostic_only")) else 0.0 for info in lyap_info_storage],
            dtype=float,
        ),
        "diagnostic_candidate_accepted_flags": np.array(
            [np.nan if info.get("diagnostic_candidate_accepted") is None else (1.0 if bool(info.get("diagnostic_candidate_accepted")) else 0.0) for info in lyap_info_storage],
            dtype=float,
        ),
        "diagnostic_unsafe_flags": np.array(
            [1.0 if bool(info.get("diagnostic_unsafe", False)) else 0.0 for info in lyap_info_storage],
            dtype=float,
        ),
        "diagnostic_unstable_flags": np.array(
            [1.0 if bool(info.get("diagnostic_unstable", False)) else 0.0 for info in lyap_info_storage],
            dtype=float,
        ),
        "actual_intervention_flags": np.array(
            [
                1.0
                if bool(
                    info.get(
                        "actual_intervention",
                        str(info.get("correction_mode", "")) not in {
                            "",
                            "none",
                            "None",
                            "accepted_candidate",
                            "mpc_only_diagnostic_bypass",
                        },
                    )
                )
                else 0.0
                for info in lyap_info_storage
            ],
            dtype=float,
        ),
        "handoff_active_flags": np.array(
            [1.0 if bool(info.get("handoff_active", False)) else 0.0 for info in lyap_info_storage],
            dtype=float,
        ),
        "handoff_alpha": np.array([info.get("handoff_alpha", np.nan) for info in lyap_info_storage], dtype=float),
        "bc_teacher_gap_inf": np.array([info.get("bc_teacher_gap_inf", np.nan) for info in lyap_info_storage], dtype=float),
        "handoff_candidate_gap_inf": np.array(
            [info.get("handoff_candidate_gap_inf", np.nan) for info in lyap_info_storage],
            dtype=float,
        ),
        "diagnostic_safety_active_flags": np.maximum(
            np.array([1.0 if bool(info.get("diagnostic_unsafe", False)) else 0.0 for info in lyap_info_storage], dtype=float),
            np.array([1.0 if bool(info.get("diagnostic_unstable", False)) else 0.0 for info in lyap_info_storage], dtype=float),
        ),
        "fallback_verified_flags": np.array(
            [1.0 if bool(info.get("fallback_verified", False)) else 0.0 for info in lyap_info_storage],
            dtype=float,
        ),
        "projection_active_flags": np.array(
            [1.0 if str(info.get("correction_mode", "")) == "optimized_correction" else 0.0 for info in lyap_info_storage],
            dtype=float,
        ),
        "correction_modes": [str(info.get("correction_mode", "none")) for info in lyap_info_storage],
        "accept_reasons": [info.get("accept_reason") for info in lyap_info_storage],
        "reject_reasons": [info.get("reject_reason") for info in lyap_info_storage],
        "solver_statuses": [info.get("solver_status") for info in lyap_info_storage],
        "fallback_solver_statuses": [info.get("fallback_solver_status") for info in lyap_info_storage],
        "extra": {} if extra is None else extra,
        "delta_t": None if extra is None else extra.get("delta_t"),
        "phase_plot_boundaries": None
        if extra is None
        else extra.get("phase_plot_boundaries", extra.get("warm_start_plot")),
        "warm_start_plot": None
        if extra is None
        else extra.get("warm_start_plot", extra.get("phase_plot_boundaries")),
        "start_plot_idx": 10 if extra is None else int(extra.get("start_plot_idx", 10)),
        "actor_losses": None if extra is None or extra.get("actor_losses") is None else np.asarray(extra.get("actor_losses"), float),
        "critic_losses": None if extra is None or extra.get("critic_losses") is None else np.asarray(extra.get("critic_losses"), float),
    }
    if steady_states is not None and data_min is not None and data_max is not None:
        data_min = np.asarray(data_min, float)
        data_max = np.asarray(data_max, float)
        ss_inputs = np.asarray(steady_states["ss_inputs"], float).reshape(-1)
        ss_outputs = np.asarray(steady_states["y_ss"], float).reshape(-1)
        ss_scaled_u = apply_min_max(ss_inputs, data_min[:n_u], data_max[:n_u])
        ss_scaled_y = apply_min_max(ss_outputs, data_min[n_u:], data_max[n_u:])
        y_scale = data_max[n_u:] - data_min[n_u:]

        bundle["u_target_phys_store"] = reverse_min_max(
            bundle["u_target_dev_store"] + ss_scaled_u.reshape(1, -1),
            data_min[:n_u],
            data_max[:n_u],
        )
        bundle["y_sp_phys_store"] = reverse_min_max(
            bundle["y_sp"] + ss_scaled_y.reshape(1, -1),
            data_min[n_u:],
            data_max[n_u:],
        )
        bundle["y_target_phys_store"] = reverse_min_max(
            bundle["y_target_store"] + ss_scaled_y.reshape(1, -1),
            data_min[n_u:],
            data_max[n_u:],
        )
        bundle["r_target_phys_store"] = reverse_min_max(
            bundle["r_target_store"] + ss_scaled_y.reshape(1, -1),
            data_min[n_u:],
            data_max[n_u:],
        )
        bundle["yhat_phys_store"] = reverse_min_max(
            bundle["yhat"].T + ss_scaled_y.reshape(1, -1),
            data_min[n_u:],
            data_max[n_u:],
        )
        bundle["y_target_phys_dev_store"] = bundle["y_target_store"] * y_scale.reshape(1, -1)
        bundle["r_target_phys_dev_store"] = bundle["r_target_store"] * y_scale.reshape(1, -1)
        bundle["cx_s_phys_dev_store"] = bundle["cx_s_store"] * y_scale.reshape(1, -1)
        bundle["cd_d_s_phys_dev_store"] = bundle["cd_d_s_store"] * y_scale.reshape(1, -1)
    else:
        bundle["u_target_phys_store"] = np.full_like(bundle["u_target_dev_store"], np.nan)
        bundle["y_sp_phys_store"] = np.full_like(bundle["y_sp"], np.nan)
        bundle["y_target_phys_store"] = np.full_like(bundle["y_target_store"], np.nan)
        bundle["r_target_phys_store"] = np.full_like(bundle["r_target_store"], np.nan)
        bundle["yhat_phys_store"] = np.full_like(bundle["yhat"].T, np.nan)
        bundle["y_target_phys_dev_store"] = np.full_like(bundle["y_target_store"], np.nan)
        bundle["r_target_phys_dev_store"] = np.full_like(bundle["r_target_store"], np.nan)
        bundle["cx_s_phys_dev_store"] = np.full_like(bundle["cx_s_store"], np.nan)
        bundle["cd_d_s_phys_dev_store"] = np.full_like(bundle["cd_d_s_store"], np.nan)

    bundle["summary"] = summarize_safety_filter_bundle(bundle)
    return bundle


def _normalize_step_matrix(values, n_steps, width):
    values = np.asarray(values, dtype=float)
    if values.ndim == 1:
        if values.size != width:
            raise ValueError(f"Expected vector of width {width}, got {values.size}.")
        return np.tile(values.reshape(1, -1), (n_steps, 1))
    if values.shape == (width, n_steps):
        return values.T.copy()
    if values.shape == (n_steps, width):
        return values.copy()
    raise ValueError(f"Cannot normalize array of shape {values.shape} into ({n_steps}, {width}).")


def _physical_setpoint_steps(bundle):
    y_sp = _normalize_step_matrix(
        bundle["y_sp"],
        int(bundle["nFE"]),
        int(np.asarray(bundle["y_system"], dtype=float).shape[1]),
    )
    steady_states = bundle.get("steady_states")
    data_min = bundle.get("data_min")
    data_max = bundle.get("data_max")
    if steady_states is None or data_min is None or data_max is None:
        return y_sp.copy()

    n_u = int(np.asarray(bundle["u_applied_phys"], dtype=float).shape[1])
    y_ss_scaled = apply_min_max(
        np.asarray(steady_states["y_ss"], dtype=float).reshape(-1),
        np.asarray(data_min, dtype=float)[n_u:],
        np.asarray(data_max, dtype=float)[n_u:],
    )
    return reverse_min_max(
        y_sp + y_ss_scaled.reshape(1, -1),
        np.asarray(data_min, dtype=float)[n_u:],
        np.asarray(data_max, dtype=float)[n_u:],
    )


def _output_deviation_steps_to_physical(bundle, values):
    values = np.asarray(values, dtype=float)
    if values.ndim == 1:
        values = values.reshape(1, -1)

    data_min = bundle.get("data_min")
    data_max = bundle.get("data_max")
    if data_min is None or data_max is None:
        return values.copy(), "scaled deviation"

    n_u = int(np.asarray(bundle["u_applied_phys"], dtype=float).shape[1])
    y_scale = np.asarray(data_max, dtype=float)[n_u:] - np.asarray(data_min, dtype=float)[n_u:]
    return values * y_scale.reshape(1, -1), "physical deviation"


def _output_deviation_steps_to_physical_absolute(bundle, values):
    values = np.asarray(values, dtype=float)
    if values.ndim == 1:
        values = values.reshape(1, -1)

    steady_states = bundle.get("steady_states")
    data_min = bundle.get("data_min")
    data_max = bundle.get("data_max")
    if steady_states is None or data_min is None or data_max is None:
        return values.copy(), "scaled deviation"

    n_u = int(np.asarray(bundle["u_applied_phys"], dtype=float).shape[1])
    y_ss_scaled = apply_min_max(
        np.asarray(steady_states["y_ss"], dtype=float).reshape(-1),
        np.asarray(data_min, dtype=float)[n_u:],
        np.asarray(data_max, dtype=float)[n_u:],
    )
    return (
        reverse_min_max(
            values + y_ss_scaled.reshape(1, -1),
            np.asarray(data_min, dtype=float)[n_u:],
            np.asarray(data_max, dtype=float)[n_u:],
        ),
        "physical units",
    )


def _bundle_output_plot_array(bundle, physical_key, deviation_key):
    if physical_key in bundle:
        arr = np.asarray(bundle[physical_key], dtype=float)
        if arr.size and np.any(np.isfinite(arr)):
            return arr.copy(), "physical units"
    return _output_deviation_steps_to_physical_absolute(bundle, bundle[deviation_key])


def _bundle_output_deviation_plot_array(bundle, physical_key, deviation_key):
    if physical_key in bundle:
        arr = np.asarray(bundle[physical_key], dtype=float)
        if arr.size and np.any(np.isfinite(arr)):
            return arr.copy(), "physical deviation"
    return _output_deviation_steps_to_physical(bundle, bundle[deviation_key])


def safety_output_rmse_post_step(bundle):
    y_sp_phys = _physical_setpoint_steps(bundle)
    y_post = np.asarray(bundle["y_system"], dtype=float)[1 : 1 + y_sp_phys.shape[0], :]
    n_rows = min(y_post.shape[0], y_sp_phys.shape[0])
    n_cols = min(y_post.shape[1], y_sp_phys.shape[1])
    if n_rows <= 0 or n_cols <= 0:
        return np.array([], dtype=float)
    err = y_post[:n_rows, :n_cols] - y_sp_phys[:n_rows, :n_cols]
    return np.sqrt(np.mean(err**2, axis=0))


def make_safety_filter_episode_records(bundle):
    episode_len = int(bundle.get("time_in_sub_episodes", 0))
    n_steps = int(bundle.get("nFE", 0))
    if episode_len <= 0 or n_steps <= 0:
        return []

    y_sp_phys = _physical_setpoint_steps(bundle)
    y_post = np.asarray(bundle["y_system"], dtype=float)[1 : 1 + n_steps, :]
    rewards = np.asarray(bundle["rewards"], dtype=float).reshape(-1)
    reward_no_penalty = np.asarray(
        bundle.get("reward_no_penalty", bundle.get("reward_base", [])),
        dtype=float,
    ).reshape(-1)
    fallback_penalty = np.asarray(bundle.get("fallback_penalty", []), dtype=float).reshape(-1)
    diagnostic_unsafe = np.asarray(bundle.get("diagnostic_unsafe_flags", []), dtype=float).reshape(-1)
    diagnostic_unstable = np.asarray(bundle.get("diagnostic_unstable_flags", []), dtype=float).reshape(-1)
    actual_intervention = np.asarray(bundle.get("actual_intervention_flags", []), dtype=float).reshape(-1)
    contraction_margin = np.asarray(bundle.get("contraction_margin", []), dtype=float).reshape(-1)
    executed_action_gap_inf = np.asarray(bundle.get("executed_action_gap_inf", []), dtype=float).reshape(-1)
    modes = list(bundle.get("correction_modes", []))
    n_episodes = int(np.ceil(n_steps / float(episode_len)))

    records = []
    for episode_idx in range(n_episodes):
        start = episode_idx * episode_len
        stop = min((episode_idx + 1) * episode_len, n_steps)
        if stop <= start:
            continue

        err = y_post[start:stop, :] - y_sp_phys[start:stop, :]
        rmse = np.sqrt(np.mean(err**2, axis=0)) if err.size else np.array([], dtype=float)
        max_abs = np.max(np.abs(err), axis=0) if err.size else np.array([], dtype=float)
        mode_slice = modes[start:stop]
        row = {
            "episode": int(episode_idx + 1),
            "step_start": int(start),
            "step_stop_exclusive": int(stop),
            "n_steps": int(stop - start),
            "reward_mean": float(np.mean(rewards[start:stop])) if stop > start else None,
            "reward_sum": float(np.sum(rewards[start:stop])) if stop > start else None,
            "reward_no_penalty_mean": _safe_nanmean(reward_no_penalty[start:stop]),
            "reward_no_penalty_sum": _safe_nansum(reward_no_penalty[start:stop]),
            "fallback_penalty_mean": _safe_nanmean(fallback_penalty[start:stop]),
            "fallback_penalty_sum": _safe_nansum(fallback_penalty[start:stop]),
            "diagnostic_unsafe_count": int(np.nansum(diagnostic_unsafe[start:stop] > 0.5)),
            "diagnostic_unsafe_rate": _safe_nanmean(diagnostic_unsafe[start:stop]),
            "diagnostic_unstable_count": int(np.nansum(diagnostic_unstable[start:stop] > 0.5)),
            "diagnostic_unstable_rate": _safe_nanmean(diagnostic_unstable[start:stop]),
            "actual_intervention_count": int(np.nansum(actual_intervention[start:stop] > 0.5)),
            "actual_intervention_rate": _safe_nanmean(actual_intervention[start:stop]),
            "accepted_candidate_count": int(sum(mode == "accepted_candidate" for mode in mode_slice)),
            "fallback_count": int(
                sum(mode in {"fallback_mpc_verified", "fallback_mpc_unverified"} for mode in mode_slice)
            ),
            "fallback_verified_count": int(sum(mode == "fallback_mpc_verified" for mode in mode_slice)),
            "fallback_unverified_count": int(sum(mode == "fallback_mpc_unverified" for mode in mode_slice)),
            "target_fail_hold_prev_count": int(sum(mode == "target_fail_hold_prev" for mode in mode_slice)),
            "solver_fail_hold_prev_count": int(sum(mode == "solver_fail_hold_prev" for mode in mode_slice)),
            "target_failure_count": int(
                np.sum(np.asarray(bundle.get("target_failure_flags", np.zeros(n_steps)), dtype=float)[start:stop] > 0.5)
            ),
            "min_contraction_margin": _safe_nanmin(contraction_margin[start:stop]),
            "max_executed_action_gap_inf": _safe_nanmax(executed_action_gap_inf[start:stop]),
            "output_rmse_mean": _safe_nanmean(rmse),
            "output_max_abs_error": _safe_nanmax(max_abs),
        }
        for idx, value in enumerate(rmse):
            row[f"output{idx}_rmse"] = float(value)
        for idx, value in enumerate(max_abs):
            row[f"output{idx}_max_abs_error"] = float(value)
        records.append(row)
    return records


def make_safety_filter_comparison_record(case_name, bundle, debug_dir=None):
    summary = dict(bundle.get("summary", {}))
    rmse = safety_output_rmse_post_step(bundle)
    episode_records = make_safety_filter_episode_records(bundle)

    record = {
        "case_name": str(case_name),
        "source": summary.get("source", bundle.get("source")),
        "n_steps": summary.get("n_steps", bundle.get("nFE")),
        "reward_mean": summary.get("reward_mean"),
        "reward_sum": summary.get("reward_sum"),
        "reward_base_mean": summary.get("reward_base_mean"),
        "reward_base_sum": summary.get("reward_base_sum"),
        "reward_no_penalty_mean": summary.get("reward_no_penalty_mean"),
        "reward_no_penalty_sum": summary.get("reward_no_penalty_sum"),
        "reward_augmented_mean": summary.get("reward_augmented_mean"),
        "reward_augmented_sum": summary.get("reward_augmented_sum"),
        "fallback_penalty_mean": summary.get("fallback_penalty_mean"),
        "fallback_penalty_max": summary.get("fallback_penalty_max"),
        "fallback_penalty_sum": summary.get("fallback_penalty_sum"),
        "weighted_correction_gap_mean": summary.get("weighted_correction_gap_mean"),
        "weighted_correction_gap_max": summary.get("weighted_correction_gap_max"),
        "verified_rate": summary.get("verified_rate"),
        "accepted_rate": summary.get("accepted_rate"),
        "fallback_rate": summary.get("fallback_rate"),
        "diagnostic_unsafe_rate": summary.get("diagnostic_unsafe_rate"),
        "diagnostic_unstable_rate": summary.get("diagnostic_unstable_rate"),
        "actual_intervention_rate": summary.get("actual_intervention_rate"),
        "diagnostic_safety_active_rate": summary.get("diagnostic_safety_active_rate"),
        "n_accepted_candidate": summary.get("n_accepted_candidate"),
        "n_fallback_mpc_verified": summary.get("n_fallback_mpc_verified"),
        "n_fallback_mpc_unverified": summary.get("n_fallback_mpc_unverified"),
        "n_target_fail_hold_prev": summary.get("n_target_fail_hold_prev"),
        "n_solver_fail_hold_prev": summary.get("n_solver_fail_hold_prev"),
        "n_target_failures": summary.get("n_target_failures"),
        "lyapunov_margin_min": summary.get("lyapunov_margin_min"),
        "target_cond_M_max": summary.get("target_cond_M_max"),
        "target_cond_G_max": summary.get("target_cond_G_max"),
        "target_residual_total_norm_max": summary.get("target_residual_total_norm_max"),
        "target_u_ref_active_steps": summary.get("target_u_ref_active_steps"),
        "target_us_u_ref_inf_mean": summary.get("target_us_u_ref_inf_mean"),
        "target_us_u_ref_inf_max": summary.get("target_us_u_ref_inf_max"),
        "target_x_ref_active_steps": summary.get("target_x_ref_active_steps"),
        "target_xs_x_ref_inf_mean": summary.get("target_xs_x_ref_inf_mean"),
        "target_xs_x_ref_inf_max": summary.get("target_xs_x_ref_inf_max"),
        "executed_action_gap_inf_mean": summary.get("executed_action_gap_inf_mean"),
        "executed_action_gap_inf_max": summary.get("executed_action_gap_inf_max"),
        "episode_reward_mean": _safe_nanmean([row.get("reward_mean") for row in episode_records]),
        "episode_reward_no_penalty_mean": _safe_nanmean(
            [row.get("reward_no_penalty_mean") for row in episode_records]
        ),
        "episode_fallback_mean": _safe_nanmean([row.get("fallback_count") for row in episode_records]),
        "episode_gap_inf_max": _safe_nanmax([row.get("max_executed_action_gap_inf") for row in episode_records]),
        "wall_clock_seconds": summary.get("wall_clock_seconds"),
        "wall_clock_seconds_per_episode": summary.get("wall_clock_seconds_per_episode"),
        "wall_clock_seconds_per_step": summary.get("wall_clock_seconds_per_step"),
        "wall_clock_steps_per_second": summary.get("wall_clock_steps_per_second"),
        "debug_dir": None if debug_dir is None else str(debug_dir),
    }
    for idx, value in enumerate(rmse):
        record[f"output{idx}_rmse"] = float(value)
    record["output_rmse_mean"] = _safe_nanmean(rmse)
    return record


def _write_csv(path, records):
    if not records:
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([])
        return
    fieldnames = list(records[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def _safety_npz_arrays(bundle, export_profile="debug"):
    profile = _normalize_export_profile(export_profile)
    arrays = {
        "y_system": bundle["y_system"],
        "u_applied_phys": bundle["u_applied_phys"],
        "y_sp": bundle["y_sp"],
        "qi": bundle["qi"],
        "qs": bundle["qs"],
        "ha": bundle["ha"],
        "u_safe_dev_store": bundle["u_safe_dev_store"],
        "u_cand_dev_store": bundle["u_cand_dev_store"],
        "behavior_action_pre_filter_store": bundle["behavior_action_pre_filter_store"],
        "teacher_action_pre_filter_store": bundle["teacher_action_pre_filter_store"],
        "teacher_u_dev_pre_filter_store": bundle["teacher_u_dev_pre_filter_store"],
        "policy_action_pre_handoff_store": bundle["policy_action_pre_handoff_store"],
        "policy_u_dev_pre_handoff_store": bundle["policy_u_dev_pre_handoff_store"],
        "u_target_dev_store": bundle["u_target_dev_store"],
        "u_fallback_mpc_dev_store": bundle["u_fallback_mpc_dev_store"],
        "y_target_store": bundle["y_target_store"],
        "target_mismatch_inf": bundle["target_mismatch_inf"],
        "target_residual_total_norm": bundle["target_residual_total_norm"],
        "target_stage_code": bundle["target_stage_code"],
        "executed_action_gap_inf": bundle["executed_action_gap_inf"],
        "reward_base": bundle["reward_base"],
        "reward_no_penalty": bundle.get("reward_no_penalty", bundle["reward_base"]),
        "reward_augmented": bundle["reward_augmented"],
        "fallback_penalty": bundle["fallback_penalty"],
        "weighted_correction_gap": bundle["weighted_correction_gap"],
        "contraction_margin": bundle["contraction_margin"],
        "contraction_margin_applied": bundle["contraction_margin_applied"],
        "lyapunov_margin": bundle["lyapunov_margin"],
        "rewards": bundle["rewards"],
        "verified_flags": bundle["verified_flags"],
        "accepted_flags": bundle["accepted_flags"],
        "diagnostic_only_flags": bundle["diagnostic_only_flags"],
        "diagnostic_candidate_accepted_flags": bundle["diagnostic_candidate_accepted_flags"],
        "diagnostic_unsafe_flags": bundle["diagnostic_unsafe_flags"],
        "diagnostic_unstable_flags": bundle["diagnostic_unstable_flags"],
        "diagnostic_safety_active_flags": bundle["diagnostic_safety_active_flags"],
        "actual_intervention_flags": bundle["actual_intervention_flags"],
        "handoff_active_flags": bundle["handoff_active_flags"],
        "handoff_alpha": bundle["handoff_alpha"],
        "bc_teacher_gap_inf": bundle["bc_teacher_gap_inf"],
        "handoff_candidate_gap_inf": bundle["handoff_candidate_gap_inf"],
        "fallback_verified_flags": bundle["fallback_verified_flags"],
        "reward_fallback_active_flags": bundle["reward_fallback_active_flags"],
        "projection_active_flags": bundle["projection_active_flags"],
        "safety_active_flags": _safety_active_flags_from_bundle(bundle),
    }
    optional_keys = [
        "y_sp_phys_store",
        "y_target_phys_store",
        "r_target_phys_store",
        "yhat_phys_store",
        "u_target_phys_store",
    ]
    for key in optional_keys:
        if key in bundle:
            arrays[key] = bundle[key]
    if profile == "debug":
        debug_keys = [
            "xhatdhat",
            "yhat",
            "e_store",
            "u_prev_dev_store",
            "target_u_ref_store",
            "target_u_ref_weight_store",
            "target_x_ref_store",
            "target_x_ref_weight_store",
            "x_target_store",
            "d_target_store",
            "r_target_store",
            "cx_s_store",
            "cd_d_s_store",
            "selector_target_tracking_term",
            "selector_u_applied_anchor_term",
            "selector_u_prev_smoothing_term",
            "selector_x_prev_smoothing_term",
            "selector_xhat_anchor_term",
            "selector_objective_value",
            "selector_x_s_minus_xhat_inf",
            "selector_x_s_minus_xprev_inf",
            "V_k",
            "V_next_first",
            "V_next_first_candidate",
            "V_next_first_applied",
            "V_next_cand",
            "V_bound",
            "contraction_margin_candidate",
            "first_step_contraction_satisfied_flags",
            "first_step_contraction_satisfied_applied_flags",
            "final_lyap_value",
            "final_lyap_bound",
            "final_lyap_margin",
            "target_error_inf",
            "target_slack_inf",
            "selector_dyn_residual_inf",
            "selector_bound_violation_inf",
            "target_cond_M",
            "target_cond_G",
            "target_u_ref_active_flags",
            "target_x_ref_active_flags",
            "target_us_u_ref_inf",
            "target_xs_x_ref_inf",
            "target_success_flags",
            "target_failure_flags",
            "effective_target_success_flags",
            "effective_target_reused_flags",
            "d_s_minus_dhat_inf",
            "qcqp_attempted_flags",
            "qcqp_solved_flags",
            "qcqp_hard_accepted_flags",
            "candidate_first_step_lyap_ok_flags",
            "first_step_contraction_triggered_flags",
            "constrained_mpc_attempted_flags",
            "constrained_mpc_solved_flags",
            "constrained_mpc_applied_flags",
            "constrained_mpc_failed_applied_candidate_flags",
        ]
        for key in debug_keys:
            if key in bundle:
                arrays[key] = bundle[key]
    return arrays


def _save_npz(path, bundle, export_profile="debug"):
    np.savez_compressed(path, **_safety_npz_arrays(bundle, export_profile=export_profile))


def _comparison_plot_path(output_dir, filename):
    os.makedirs(output_dir, exist_ok=True)
    return os.path.join(
        output_dir,
        _maybe_windows_short_filename(
            output_dir,
            filename,
            _SAFETY_COMPARISON_FILENAME_MAP,
        ),
    )


def _record_series(records, key):
    out = []
    for record in records:
        value = record.get(key)
        if value is None:
            out.append(np.nan)
        else:
            try:
                out.append(float(value))
            except Exception:
                out.append(np.nan)
    return np.asarray(out, dtype=float)


def _episode_fallback_or_would_activate_counts(case_name, bundle):
    records = make_safety_filter_episode_records(bundle)
    config = bundle.get("config", {}) if isinstance(bundle.get("config"), dict) else {}
    is_mpc_only = (
        str(case_name).strip().lower() == "mpc_only"
        or str(config.get("controller_mode", "")).strip().lower() == "mpc_only"
        or str(config.get("projection_backend", "")).strip().lower() in {"mpc_only", "mpc_only_diagnostic"}
    )
    key = "diagnostic_unsafe_count" if is_mpc_only else "fallback_count"
    values = np.asarray([row.get(key, np.nan) for row in records], dtype=float)
    return values[np.isfinite(values)]


def _save_safety_comparison_bar(records, keys, labels, ylabel, title, path):
    x = np.arange(len(records))
    case_labels = [str(record["case_name"]) for record in records]
    width = min(0.8 / max(len(keys), 1), 0.35)
    fig, ax = plt.subplots(figsize=(11, 4.8))
    for idx, key in enumerate(keys):
        offset = (idx - (len(keys) - 1) / 2.0) * width
        ax.bar(x + offset, _record_series(records, key), width=width, label=labels[idx])
    ax.set_xticks(x)
    ax.set_xticklabels(case_labels, rotation=25, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def _save_safety_mode_stacked_bar(records, path):
    x = np.arange(len(records))
    case_labels = [str(record["case_name"]) for record in records]
    accepted = _record_series(records, "n_accepted_candidate")
    fallback_verified = _record_series(records, "n_fallback_mpc_verified")
    fallback_unverified = _record_series(records, "n_fallback_mpc_unverified")
    target_fail = _record_series(records, "n_target_fail_hold_prev")
    solver_fail = _record_series(records, "n_solver_fail_hold_prev")

    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.bar(x, accepted, label="accepted_candidate")
    ax.bar(x, fallback_verified, bottom=accepted, label="fallback_verified")
    ax.bar(x, fallback_unverified, bottom=accepted + fallback_verified, label="fallback_unverified")
    ax.bar(
        x,
        target_fail,
        bottom=accepted + fallback_verified + fallback_unverified,
        label="target_fail_hold_prev",
    )
    ax.bar(
        x,
        solver_fail,
        bottom=accepted + fallback_verified + fallback_unverified + target_fail,
        label="solver_fail_hold_prev",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(case_labels, rotation=25, ha="right")
    ax.set_ylabel("step count")
    ax.set_title("RL Safety Gate Correction Modes")
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def _save_safety_distribution_boxplot(data_by_case, ylabel, title, path):
    case_labels = [label for label, values in data_by_case if len(values) > 0]
    values = [values for _label, values in data_by_case if len(values) > 0]
    if not values:
        return None
    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.boxplot(values, labels=case_labels, showfliers=False)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def _save_safety_last_episode_overlays(bundles_by_case, output_dir):
    if not bundles_by_case:
        return {}

    first_bundle = next(iter(bundles_by_case.values()))
    episode_len = int(first_bundle.get("time_in_sub_episodes", 0))
    if episode_len <= 0:
        return {}

    paths = {}
    time_y = np.arange(episode_len + 1)
    time_u = np.arange(episode_len)

    y_sp_phys = _physical_setpoint_steps(first_bundle)
    y_sp_last = y_sp_phys[-episode_len:, :]

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    for case_name, bundle in bundles_by_case.items():
        y_data = np.asarray(bundle["y_system"], dtype=float)
        y_last = y_data[-(episode_len + 1) :, :]
        axes[0].plot(time_y, y_last[:, 0], linewidth=1.8, label=case_name)
        axes[1].plot(time_y, y_last[:, 1], linewidth=1.8, label=case_name)
    axes[0].step(time_u, y_sp_last[:, 0], where="post", linestyle="--", color="k", label="setpoint")
    axes[1].step(time_u, y_sp_last[:, 1], where="post", linestyle="--", color="k", label="setpoint")
    axes[0].set_ylabel("output 0")
    axes[1].set_ylabel("output 1")
    axes[1].set_xlabel("step in last episode")
    axes[0].set_title("Last-Episode Output Overlay")
    for ax in axes:
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.legend(loc="best")
    fig.tight_layout()
    paths["outputs_last_episode"] = _comparison_plot_path(output_dir, "comparison_outputs_last_episode.png")
    fig.savefig(paths["outputs_last_episode"], dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    for case_name, bundle in bundles_by_case.items():
        u_data = np.asarray(bundle["u_applied_phys"], dtype=float)
        u_last = u_data[-episode_len:, :]
        axes[0].step(time_u, u_last[:, 0], where="post", linewidth=1.8, label=case_name)
        axes[1].step(time_u, u_last[:, 1], where="post", linewidth=1.8, label=case_name)
    axes[0].set_ylabel("input 0")
    axes[1].set_ylabel("input 1")
    axes[1].set_xlabel("step in last episode")
    axes[0].set_title("Last-Episode Input Overlay")
    for ax in axes:
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.legend(loc="best")
    fig.tight_layout()
    paths["inputs_last_episode"] = _comparison_plot_path(output_dir, "comparison_inputs_last_episode.png")
    fig.savefig(paths["inputs_last_episode"], dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 4.8))
    for case_name, bundle in bundles_by_case.items():
        margin = np.asarray(bundle.get("contraction_margin", []), dtype=float)
        if margin.size < episode_len:
            continue
        ax.plot(time_u, margin[-episode_len:], linewidth=1.8, label=case_name)
    ax.axhline(0.0, color="k", linestyle="--", linewidth=1.0)
    ax.set_ylabel("V_next - V_bound")
    ax.set_xlabel("step in last episode")
    ax.set_title("Last-Episode Contraction Margin Overlay")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="best")
    fig.tight_layout()
    paths["contraction_margin_last_episode"] = _comparison_plot_path(
        output_dir,
        "comparison_contraction_margin_last_episode.png",
    )
    fig.savefig(paths["contraction_margin_last_episode"], dpi=300, bbox_inches="tight")
    plt.close(fig)
    return paths


def _truncate_path_component(value, max_len=64):
    value = str(value)
    if len(value) <= max_len:
        return value
    if max_len <= 5:
        return value[:max_len]
    head = (max_len - 1) // 2
    tail = max_len - head - 1
    return f"{value[:head]}_{value[-tail:]}"


def _unique_directory_candidate(base_path):
    candidate = str(base_path)
    idx = 2
    while os.path.exists(candidate):
        candidate = f"{base_path}_{idx:02d}"
        idx += 1
    return candidate


def _project_safety_debug_max_path_len(
    out_dir,
):
    long_len = _max_joined_path_len(
        out_dir,
        _safety_plot_rel_paths(short_paths=False),
    )
    if os.name != "nt" or long_len <= _WINDOWS_PATH_SOFT_LIMIT:
        return long_len
    return _max_joined_path_len(
        out_dir,
        _safety_plot_rel_paths(short_paths=True),
    )


def _should_use_short_safety_plot_paths(output_dir):
    if os.name != "nt":
        return False
    return _max_joined_path_len(
        output_dir,
        _safety_plot_rel_paths(short_paths=False),
    ) > _WINDOWS_PATH_SOFT_LIMIT


def _select_safety_debug_output_dir(
    directory,
    prefix_name,
    *,
    timestamp,
):
    method_slug = _sanitize_method_slug(prefix_name, default="safety_filter")
    standard_base = os.path.join(directory, method_slug)
    standard_candidate = _unique_directory_candidate(standard_base)
    if os.name != "nt":
        return standard_candidate

    short_stamp = str(timestamp)[-6:]
    candidate_bases = [
        standard_base,
        os.path.join(directory, _truncate_path_component(method_slug, max_len=56)),
        os.path.join(directory, f"{_truncate_path_component(method_slug, max_len=48)}_{short_stamp}"),
    ]

    best_candidate = standard_candidate
    best_len = _project_safety_debug_max_path_len(best_candidate)
    for base_path in candidate_bases:
        candidate = _unique_directory_candidate(base_path)
        projected_len = _project_safety_debug_max_path_len(candidate)
        if projected_len < best_len:
            best_candidate = candidate
            best_len = projected_len
        if projected_len <= _WINDOWS_PATH_SOFT_LIMIT:
            return candidate
    return best_candidate


def _plot_safety_filter_bundle_impl(bundle, output_dir, *, paper_style=False):
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib is required for plotting debug artifacts.")

    os.makedirs(output_dir, exist_ok=True)
    short_path_mode = _should_use_short_safety_plot_paths(output_dir)
    use_short_names = bool(paper_style or short_path_mode)
    plot_path = lambda filename: _safety_plot_path(output_dir, filename, paper_style=use_short_names)

    y_system = bundle["y_system"]
    u_applied_phys = bundle["u_applied_phys"]
    y_sp = np.asarray(bundle["y_sp"], float)
    V_k = bundle["V_k"]
    V_next_first = bundle["V_next_first"]
    V_next_first_candidate = bundle["V_next_first_candidate"]
    V_next_first_applied = bundle["V_next_first_applied"]
    V_next_cand = bundle["V_next_cand"]
    V_bound = bundle["V_bound"]
    contraction_margin = bundle["contraction_margin"]
    contraction_margin_candidate = bundle["contraction_margin_candidate"]
    contraction_margin_applied = bundle["contraction_margin_applied"]
    first_step_contraction_satisfied_flags = np.asarray(bundle["first_step_contraction_satisfied_flags"], float)
    first_step_contraction_satisfied_applied_flags = np.asarray(bundle["first_step_contraction_satisfied_applied_flags"], float)
    final_lyap_value = bundle["final_lyap_value"]
    final_lyap_bound = bundle["final_lyap_bound"]
    final_lyap_margin = bundle["final_lyap_margin"]
    lyapunov_margin = bundle["lyapunov_margin"]
    target_error_inf = bundle["target_error_inf"]
    target_slack_inf = bundle["target_slack_inf"]
    selector_dyn_residual_inf = bundle["selector_dyn_residual_inf"]
    selector_bound_violation_inf = bundle["selector_bound_violation_inf"]
    target_mismatch_inf = bundle["target_mismatch_inf"]
    target_success_flags = np.asarray(bundle["target_success_flags"], float)
    target_failure_flags = np.asarray(bundle["target_failure_flags"], float)
    target_stage_code = np.asarray(bundle["target_stage_code"], float)
    effective_target_reused_flags = np.asarray(bundle["effective_target_reused_flags"], float)
    qcqp_attempted_flags = np.asarray(bundle["qcqp_attempted_flags"], float)
    qcqp_solved_flags = np.asarray(bundle["qcqp_solved_flags"], float)
    qcqp_hard_accepted_flags = np.asarray(bundle["qcqp_hard_accepted_flags"], float)
    candidate_first_step_lyap_ok_flags = np.asarray(bundle["candidate_first_step_lyap_ok_flags"], float)
    first_step_contraction_triggered_flags = np.asarray(bundle["first_step_contraction_triggered_flags"], float)
    constrained_mpc_attempted_flags = np.asarray(bundle["constrained_mpc_attempted_flags"], float)
    constrained_mpc_solved_flags = np.asarray(bundle["constrained_mpc_solved_flags"], float)
    constrained_mpc_applied_flags = np.asarray(bundle["constrained_mpc_applied_flags"], float)
    constrained_mpc_failed_applied_candidate_flags = np.asarray(bundle["constrained_mpc_failed_applied_candidate_flags"], float)
    u_cand_dev = bundle["u_cand_dev_store"]
    u_safe_dev = bundle["u_safe_dev_store"]
    rewards = np.asarray(bundle["rewards"], dtype=float).reshape(-1)
    reward_no_penalty = _bundle_reward_no_penalty_array(bundle, rewards)
    projection_active = np.asarray(bundle["projection_active_flags"], float)
    fallback_verified_flags = np.asarray(bundle["fallback_verified_flags"], float)
    xhatdhat = np.asarray(bundle["xhatdhat"], float)
    x_target_store = np.asarray(bundle["x_target_store"], float)
    d_target_store = np.asarray(bundle["d_target_store"], float)
    y_target_store = np.asarray(bundle["y_target_store"], float)
    r_target_store = np.asarray(bundle["r_target_store"], float)
    cx_s_store = np.asarray(bundle["cx_s_store"], float)
    cd_d_s_store = np.asarray(bundle["cd_d_s_store"], float)
    selector_target_tracking_term = np.asarray(bundle["selector_target_tracking_term"], float)
    selector_u_applied_anchor_term = np.asarray(bundle["selector_u_applied_anchor_term"], float)
    selector_u_prev_smoothing_term = np.asarray(bundle["selector_u_prev_smoothing_term"], float)
    selector_x_prev_smoothing_term = np.asarray(bundle["selector_x_prev_smoothing_term"], float)
    selector_xhat_anchor_term = np.asarray(bundle["selector_xhat_anchor_term"], float)
    selector_objective_value = np.asarray(bundle["selector_objective_value"], float)
    selector_x_s_minus_xhat_inf = np.asarray(bundle["selector_x_s_minus_xhat_inf"], float)
    selector_x_s_minus_xprev_inf = np.asarray(bundle["selector_x_s_minus_xprev_inf"], float)

    u_target_plot = np.asarray(bundle["u_target_phys_store"], float)
    if not (u_target_plot.size and np.any(np.isfinite(u_target_plot))):
        u_target_plot = np.asarray(bundle["u_target_dev_store"], float)

    y_sp_plot, y_sp_unit_label = _bundle_output_plot_array(bundle, "y_sp_phys_store", "y_sp")
    y_target_plot, y_target_unit_label = _bundle_output_plot_array(bundle, "y_target_phys_store", "y_target_store")
    r_target_plot, _ = _bundle_output_plot_array(bundle, "r_target_phys_store", "r_target_store")
    yhat_plot = np.asarray(bundle["yhat_phys_store"], float)
    if not (yhat_plot.size and np.any(np.isfinite(yhat_plot))):
        yhat_plot = np.asarray(bundle["yhat"], float).T

    y_target_dev_plot, y_target_dev_unit_label = _bundle_output_deviation_plot_array(
        bundle,
        "y_target_phys_dev_store",
        "y_target_store",
    )
    cx_s_plot, _ = _bundle_output_deviation_plot_array(bundle, "cx_s_phys_dev_store", "cx_s_store")
    cd_d_s_plot, _ = _bundle_output_deviation_plot_array(bundle, "cd_d_s_phys_dev_store", "cd_d_s_store")

    time_y = np.arange(y_system.shape[0])
    time_u = np.arange(u_applied_phys.shape[0])
    n_y = y_system.shape[1]
    n_aug = xhatdhat.shape[0]
    n_x = max(n_aug - n_y, 0)
    first_step_replacement_mode = bool(
        "first_step" in str(bundle.get("source", "")).lower()
        and (
            np.any(constrained_mpc_attempted_flags > 0.5)
            or np.any(np.isfinite(candidate_first_step_lyap_ok_flags))
            or np.any(first_step_contraction_triggered_flags > 0.5)
        )
    )

    def _paper_dirname(long_name, short_name):
        return short_name if use_short_names else long_name

    def _plot_augmented_states(data, filename, dhat_only=False):
        data = np.asarray(data, float)
        if data.ndim != 2 or data.shape[1] <= 0:
            return

        if dhat_only:
            row_offset = n_x
            state_rows = list(range(row_offset, data.shape[0]))
            label_prefix = "dhat"
        else:
            row_offset = 0
            state_rows = list(range(data.shape[0]))
            label_prefix = None

        if not state_rows:
            return

        fig, axes = plt.subplots(len(state_rows), 1, figsize=(10, 2.6 * len(state_rows)), sharex=True)
        axes = np.atleast_1d(axes)
        time_vals = np.arange(data.shape[1])

        for ax_idx, row_idx in enumerate(state_rows):
            ax = axes[ax_idx]
            is_dhat = row_idx >= n_x
            color = "tab:orange" if is_dhat else "tab:blue"
            line_width = 2.0 if is_dhat else 1.8
            if dhat_only:
                label = f"dhat_{row_idx - row_offset}"
            elif label_prefix is None and is_dhat:
                label = f"dhat_{row_idx - n_x}"
            else:
                label = f"xhat_{row_idx}"
            ax.plot(time_vals, data[row_idx, :], color=color, linewidth=line_width, label=label)
            ax.grid(True, linestyle="--", alpha=0.35)
            ax.legend(loc="best")

        axes[-1].set_xlabel("step")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches="tight")
        plt.close(fig)

    def _plot_state_target_channels(base_dir, state_data, target_data, state_prefix, target_prefix, color_state, color_target):
        state_data = np.asarray(state_data, float)
        target_data = np.asarray(target_data, float)
        if state_data.ndim != 2 or target_data.ndim != 2:
            return
        if state_data.shape[0] != target_data.shape[0] or target_data.shape[1] <= 0:
            return

        os.makedirs(base_dir, exist_ok=True)
        time_vals = np.arange(target_data.shape[1])
        for idx in range(target_data.shape[0]):
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(time_vals, state_data[idx, :target_data.shape[1]], linewidth=2.0, color=color_state, label=f"{state_prefix}{idx}")
            ax.plot(time_vals, target_data[idx, :], linewidth=2.0, linestyle="--", color=color_target, label=f"{target_prefix}{idx}")
            ax.grid(True, linestyle="--", alpha=0.35)
            ax.legend(loc="best")
            ax.set_xlabel("step")
            plt.tight_layout()
            plt.savefig(
                os.path.join(base_dir, f"{state_prefix}{idx}_vs_{target_prefix}{idx}.png"),
                dpi=300,
                bbox_inches="tight",
            )
            plt.close(fig)

    def _plot_target_vs_target_channels(base_dir, target_data, ref_data, target_prefix, ref_prefix, color_target, color_ref):
        target_data = np.asarray(target_data, float)
        ref_data = np.asarray(ref_data, float)
        if target_data.ndim != 2 or ref_data.ndim != 2:
            return
        if target_data.shape[0] != ref_data.shape[0] or target_data.shape[1] != ref_data.shape[1]:
            return

        os.makedirs(base_dir, exist_ok=True)
        time_vals = np.arange(target_data.shape[1])
        for idx in range(target_data.shape[0]):
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(time_vals, target_data[idx, :], linewidth=2.0, color=color_target, label=f"{target_prefix}{idx}")
            ax.plot(time_vals, ref_data[idx, :], linewidth=2.0, linestyle="--", color=color_ref, label=f"{ref_prefix}{idx}")
            ax.grid(True, linestyle="--", alpha=0.35)
            ax.legend(loc="best")
            ax.set_xlabel("step")
            plt.tight_layout()
            plt.savefig(
                os.path.join(base_dir, f"{target_prefix}{idx}_vs_{ref_prefix}{idx}.png"),
                dpi=300,
                bbox_inches="tight",
            )
            plt.close(fig)

    def _plot_decomposition_channels(base_dir, cx_data, cd_data, y_data):
        cx_data = np.asarray(cx_data, float)
        cd_data = np.asarray(cd_data, float)
        y_data = np.asarray(y_data, float)
        if cx_data.ndim != 2 or cd_data.ndim != 2 or y_data.ndim != 2:
            return
        if not (cx_data.shape == cd_data.shape == y_data.shape):
            return

        os.makedirs(base_dir, exist_ok=True)
        time_vals = np.arange(y_data.shape[1])
        for idx in range(y_data.shape[0]):
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(time_vals, cx_data[idx, :], linewidth=2.0, color="tab:blue", label=f"Cx_s_{idx}")
            ax.plot(time_vals, cd_data[idx, :], linewidth=2.0, color="tab:orange", label=f"Cd_d_s_{idx}")
            ax.plot(time_vals, y_data[idx, :], linewidth=2.0, linestyle="--", color="tab:green", label=f"y_s_dev_{idx}")
            ax.grid(True, linestyle="--", alpha=0.35)
            ax.legend(loc="best")
            ax.set_xlabel("step")
            plt.tight_layout()
            plt.savefig(
                os.path.join(base_dir, f"ys_decomposition_{idx}.png"),
                dpi=300,
                bbox_inches="tight",
            )
            plt.close(fig)

    _plot_augmented_states(xhatdhat, "xhatdhat_full.png")
    state_target_dir = os.path.join(output_dir, _paper_dirname("state_target_channels", "st"))
    _plot_state_target_channels(
        os.path.join(state_target_dir, _paper_dirname("full_xhat_vs_xs", "fx_xs")),
        xhatdhat[:n_x, :len(time_u)],
        x_target_store.T,
        "xhat_",
        "x_s_eff_",
        "tab:blue",
        "tab:red",
    )
    _plot_state_target_channels(
        os.path.join(state_target_dir, _paper_dirname("full_dhat_vs_ds", "fd_ds")),
        xhatdhat[n_x:, :len(time_u)],
        d_target_store.T,
        "dhat_",
        "d_s_eff_",
        "tab:orange",
        "tab:green",
    )
    _plot_target_vs_target_channels(
        os.path.join(state_target_dir, _paper_dirname("full_ys_vs_ysp", "fy_ysp")),
        y_target_plot.T,
        y_sp_plot.T,
        "ys_",
        "ysp_",
        "tab:purple",
        "tab:gray",
    )
    _plot_target_vs_target_channels(
        os.path.join(state_target_dir, _paper_dirname("full_rs_vs_ysp", "fr_ysp")),
        r_target_plot.T,
        y_sp_plot.T,
        "rs_",
        "ysp_",
        "tab:brown",
        "tab:gray",
    )
    _plot_decomposition_channels(
        os.path.join(state_target_dir, _paper_dirname("full_ys_decomposition", "fy_dec")),
        cx_s_plot.T,
        cd_d_s_plot.T,
        y_target_dev_plot.T,
    )

    fig, axes = plt.subplots(n_y, 1, figsize=(10, 3.2 * n_y), sharex=True)
    axes = np.atleast_1d(axes)
    y_system_at_u = y_system[:-1, :]
    for idx in range(n_y):
        ax = axes[idx]
        ax.plot(time_u, y_system_at_u[:, idx], linewidth=2.0, color="tab:blue", label=f"y_output_{idx}")
        ax.step(time_u, y_sp_plot[:, idx], where="post", linewidth=2.0, linestyle="--", color="tab:gray", label=f"y_sp_{idx}")
        ax.step(time_u, y_target_plot[:, idx], where="post", linewidth=2.0, linestyle="-.", color="tab:purple", label=f"y_s_{idx}")
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.legend(loc="best")
        ax.set_ylabel(y_target_unit_label)
    axes[-1].set_xlabel("step")
    plt.tight_layout()
    plt.savefig(plot_path("outputs_vs_ysp_vs_ys.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(n_y, 1, figsize=(10, 3.2 * n_y), sharex=True)
    axes = np.atleast_1d(axes)
    for idx in range(n_y):
        ax = axes[idx]
        ax.plot(time_u, cx_s_plot[:, idx], linewidth=2.0, color="tab:blue", label=f"Cx_s_{idx}")
        ax.plot(time_u, cd_d_s_plot[:, idx], linewidth=2.0, color="tab:orange", label=f"Cd_d_s_{idx}")
        ax.plot(time_u, y_target_dev_plot[:, idx], linewidth=2.0, linestyle="--", color="tab:green", label=f"y_s_dev_{idx}")
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.legend(loc="best")
        ax.set_ylabel(y_target_dev_unit_label)
    axes[-1].set_xlabel("step")
    plt.tight_layout()
    plt.savefig(plot_path("ys_decomposition_summary.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    plt.figure(figsize=(10, 6))
    for idx in range(y_system.shape[1]):
        plt.subplot(y_system.shape[1], 1, idx + 1)
        plt.plot(time_y, y_system[:, idx], label="output", linewidth=2)
        plt.step(time_u, y_sp_plot[:, idx] if y_sp_plot.ndim == 2 else y_sp_plot, where="post", linestyle="--", label="setpoint", linewidth=2)
        plt.grid(True, linestyle="--", alpha=0.35)
        plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path("outputs_vs_setpoint.png"), dpi=300, bbox_inches="tight")
    plt.close()

    fig, axes = plt.subplots(y_system.shape[1] + 1, 1, figsize=(10, 3.0 * (y_system.shape[1] + 1)), sharex=False)
    axes = np.atleast_1d(axes)
    for idx in range(y_system.shape[1]):
        ax = axes[idx]
        ax.plot(time_y, y_system[:, idx], label="output", linewidth=2)
        ax.step(
            time_u,
            y_sp_plot[:, idx] if y_sp_plot.ndim == 2 else y_sp_plot,
            where="post",
            linestyle="--",
            label="setpoint",
            linewidth=2,
        )
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.legend()
    ax = axes[-1]
    if first_step_replacement_mode:
        ax.step(time_u, candidate_first_step_lyap_ok_flags, where="post", linewidth=2, color="tab:blue", label="candidate_first_step_ok")
        ax.step(time_u, first_step_contraction_triggered_flags, where="post", linewidth=2, color="tab:orange", label="replacement_triggered")
        ax.step(time_u, constrained_mpc_applied_flags, where="post", linewidth=2, color="tab:green", label="constrained_mpc_applied")
        ax.step(
            time_u,
            constrained_mpc_failed_applied_candidate_flags,
            where="post",
            linewidth=1.5,
            linestyle="--",
            color="tab:red",
            label="constrained_failed_candidate_applied",
        )
    else:
        ax.step(time_u, qcqp_attempted_flags, where="post", linewidth=2, color="tab:orange", label="qcqp_attempted")
        ax.step(time_u, qcqp_hard_accepted_flags, where="post", linewidth=2, color="tab:green", label="qcqp_hard_accepted")
        ax.step(time_u, projection_active, where="post", linewidth=1.5, linestyle="--", color="tab:red", label="optimized_correction")
    ax.set_ylim(-0.05, 1.05)
    ax.set_yticks([0.0, 1.0])
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend()
    plt.tight_layout()
    plt.savefig(plot_path("outputs_vs_setpoint_projection.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    plt.figure(figsize=(10, 5))
    for idx in range(u_applied_phys.shape[1]):
        plt.step(time_u, u_applied_phys[:, idx], where="post", linewidth=2, label=f"u{idx}")
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path("applied_inputs.png"), dpi=300, bbox_inches="tight")
    plt.close()

    fig, axes = plt.subplots(u_applied_phys.shape[1], 1, figsize=(10, 3.0 * u_applied_phys.shape[1]), sharex=True)
    axes = np.atleast_1d(axes)
    for idx in range(u_applied_phys.shape[1]):
        ax = axes[idx]
        ax.step(time_u, u_applied_phys[:, idx], where="post", linewidth=2, label=f"u_applied_{idx}")
        ax.step(time_u, u_target_plot[:, idx], where="post", linewidth=2, linestyle="--", label=f"u_s_{idx}")
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.legend()
    plt.tight_layout()
    plt.savefig(plot_path("u_applied_vs_us.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    plt.figure(figsize=(10, 5))
    for idx in range(u_safe_dev.shape[1]):
        plt.plot(time_u, u_cand_dev[:, idx], linestyle="--", linewidth=1.5, label=f"cand_{idx}")
        plt.plot(time_u, u_safe_dev[:, idx], linewidth=2.0, label=f"safe_{idx}")
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path("candidate_vs_safe_dev.png"), dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(time_u, final_lyap_value, linewidth=2, label="final_lyap_value")
    plt.plot(time_u, final_lyap_bound, linewidth=2, linestyle="--", label="final_lyap_bound")
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path("lyapunov_values.png"), dpi=300, bbox_inches="tight")
    plt.close()

    if np.any(np.isfinite(V_next_first)) or np.any(np.isfinite(contraction_margin)):
        fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
        axes[0].plot(time_u, V_k, linewidth=2, label="V_k")
        if first_step_replacement_mode:
            axes[0].plot(time_u, V_next_first_candidate, linewidth=2, color="tab:orange", label="V_next_first_candidate")
            axes[0].plot(time_u, V_next_first_applied, linewidth=2, color="tab:green", label="V_next_first_applied")
        else:
            axes[0].plot(time_u, V_next_first, linewidth=2, label="V_next_first")
        axes[0].plot(time_u, V_bound, linewidth=2, linestyle="--", label="V_bound")
        axes[0].grid(True, linestyle="--", alpha=0.35)
        axes[0].legend()

        if first_step_replacement_mode:
            axes[1].plot(time_u, contraction_margin_candidate, linewidth=2, color="tab:orange", label="contraction_margin_candidate")
            axes[1].plot(time_u, contraction_margin_applied, linewidth=2, color="tab:green", label="contraction_margin_applied")
        else:
            axes[1].plot(time_u, contraction_margin, linewidth=2, color="tab:red", label="contraction_margin")
        axes[1].axhline(0.0, color="black", linestyle="--", linewidth=1.0)
        axes[1].grid(True, linestyle="--", alpha=0.35)
        axes[1].legend()

        if first_step_replacement_mode:
            axes[2].step(
                time_u,
                candidate_first_step_lyap_ok_flags,
                where="post",
                linewidth=2,
                color="tab:blue",
                label="candidate_first_step_ok",
            )
            axes[2].step(
                time_u,
                first_step_contraction_satisfied_applied_flags,
                where="post",
                linewidth=2,
                color="tab:green",
                label="applied_first_step_ok",
            )
            axes[2].step(
                time_u,
                constrained_mpc_failed_applied_candidate_flags,
                where="post",
                linewidth=1.5,
                linestyle="--",
                color="tab:red",
                label="constrained_failed_candidate_applied",
            )
        else:
            axes[2].step(
                time_u,
                first_step_contraction_satisfied_flags,
                where="post",
                linewidth=2,
                color="tab:green",
                label="first_step_contraction_satisfied",
            )
        axes[2].set_ylim(-0.05, 1.05)
        axes[2].set_yticks([0.0, 1.0])
        axes[2].grid(True, linestyle="--", alpha=0.35)
        axes[2].legend()
        axes[2].set_xlabel("step")
        plt.tight_layout()
        plt.savefig(plot_path("first_step_contraction_diagnostics.png"), dpi=300, bbox_inches="tight")
        plt.close(fig)

    last_len = int(bundle.get("time_in_sub_episodes", len(time_u)))
    if last_len <= 0:
        last_len = len(time_u)
    start_idx = max(0, len(time_u) - last_len)
    last_steps = time_u[start_idx:]
    start_idx_xhat = max(0, xhatdhat.shape[1] - (last_len + 1))
    xhatdhat_last = xhatdhat[:, start_idx_xhat:]
    x_target_last = x_target_store[start_idx:, :].T if x_target_store.size > 0 else x_target_store
    d_target_last = d_target_store[start_idx:, :].T if d_target_store.size > 0 else d_target_store
    y_target_last = y_target_plot[start_idx:, :].T if y_target_plot.size > 0 else y_target_plot
    r_target_last = r_target_plot[start_idx:, :].T if r_target_plot.size > 0 else r_target_plot
    y_sp_last = y_sp_plot[start_idx:, :].T if y_sp_plot.size > 0 else y_sp_plot
    yhat_last = yhat_plot[start_idx:, :]
    y_system_last = y_system[:-1, :][start_idx:, :]
    u_applied_last = u_applied_phys[start_idx:, :]
    u_target_last = u_target_plot[start_idx:, :]
    y_target_dev_last = y_target_dev_plot[start_idx:, :].T if y_target_dev_plot.size > 0 else y_target_dev_plot
    cx_s_last = cx_s_plot[start_idx:, :].T if cx_s_plot.size > 0 else cx_s_plot
    cd_d_s_last = cd_d_s_plot[start_idx:, :].T if cd_d_s_plot.size > 0 else cd_d_s_plot
    _plot_augmented_states(xhatdhat_last, "xhatdhat_last_episode.png")
    _plot_augmented_states(xhatdhat_last, "dhat_last_episode.png", dhat_only=True)
    _plot_state_target_channels(
        os.path.join(state_target_dir, _paper_dirname("last_episode_xhat_vs_xs", "lx_xs")),
        xhatdhat[:n_x, start_idx:start_idx + x_target_last.shape[1]],
        x_target_last,
        "xhat_",
        "x_s_eff_",
        "tab:blue",
        "tab:red",
    )
    _plot_state_target_channels(
        os.path.join(state_target_dir, _paper_dirname("last_episode_dhat_vs_ds", "ld_ds")),
        xhatdhat[n_x:, start_idx:start_idx + d_target_last.shape[1]],
        d_target_last,
        "dhat_",
        "d_s_eff_",
        "tab:orange",
        "tab:green",
    )
    _plot_target_vs_target_channels(
        os.path.join(state_target_dir, _paper_dirname("last_episode_ys_vs_ysp", "ly_ysp")),
        y_target_last,
        y_sp_last,
        "ys_",
        "ysp_",
        "tab:purple",
        "tab:gray",
    )
    _plot_target_vs_target_channels(
        os.path.join(state_target_dir, _paper_dirname("last_episode_rs_vs_ysp", "lr_ysp")),
        r_target_last,
        y_sp_last,
        "rs_",
        "ysp_",
        "tab:brown",
        "tab:gray",
    )
    _plot_decomposition_channels(
        os.path.join(state_target_dir, _paper_dirname("last_episode_ys_decomposition", "ly_dec")),
        cx_s_last,
        cd_d_s_last,
        y_target_dev_last,
    )

    fig, axes = plt.subplots(n_y, 1, figsize=(10, 3.2 * n_y), sharex=True)
    axes = np.atleast_1d(axes)
    for idx in range(n_y):
        ax = axes[idx]
        ax.plot(last_steps, y_system_last[:, idx], linewidth=2.0, color="tab:blue", label=f"y_output_{idx}")
        ax.step(last_steps, y_sp_last[idx, :], where="post", linewidth=2.0, linestyle="--", color="tab:gray", label=f"y_sp_{idx}")
        ax.step(last_steps, y_target_last[idx, :], where="post", linewidth=2.0, linestyle="-.", color="tab:purple", label=f"y_s_{idx}")
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.legend(loc="best")
        ax.set_ylabel(y_target_unit_label)
    axes[-1].set_xlabel("step")
    plt.tight_layout()
    plt.savefig(plot_path("outputs_vs_ysp_vs_ys_last_episode.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(n_y, 1, figsize=(10, 3.2 * n_y), sharex=True)
    axes = np.atleast_1d(axes)
    for idx in range(n_y):
        ax = axes[idx]
        ax.plot(last_steps, cx_s_last[idx, :], linewidth=2.0, color="tab:blue", label=f"Cx_s_{idx}")
        ax.plot(last_steps, cd_d_s_last[idx, :], linewidth=2.0, color="tab:orange", label=f"Cd_d_s_{idx}")
        ax.plot(last_steps, y_target_dev_last[idx, :], linewidth=2.0, linestyle="--", color="tab:green", label=f"y_s_dev_{idx}")
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.legend(loc="best")
        ax.set_ylabel(y_target_dev_unit_label)
    axes[-1].set_xlabel("step")
    plt.tight_layout()
    plt.savefig(plot_path("ys_decomposition_summary_last_episode.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(n_y, 1, figsize=(10, 3.0 * n_y), sharex=True)
    axes = np.atleast_1d(axes)
    for idx in range(n_y):
        ax = axes[idx]
        ax.plot(last_steps, y_system_last[:, idx] - y_target_last[idx, :], linewidth=2, label=f"y_system - y_s [{idx}]")
        ax.plot(last_steps, yhat_last[:, idx] - y_target_last[idx, :], linewidth=2, linestyle="--", label=f"yhat - y_s [{idx}]")
        ax.axhline(0.0, color="black", linestyle="--", linewidth=1.0)
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.legend()
    plt.tight_layout()
    plt.savefig(plot_path("y_minus_ys_last_episode.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(u_applied_phys.shape[1], 1, figsize=(10, 3.0 * u_applied_phys.shape[1]), sharex=True)
    axes = np.atleast_1d(axes)
    for idx in range(u_applied_phys.shape[1]):
        ax = axes[idx]
        ax.step(last_steps, u_applied_last[:, idx], where="post", linewidth=2, label=f"u_applied_{idx}")
        ax.step(last_steps, u_target_last[:, idx], where="post", linewidth=2, linestyle="--", label=f"u_s_{idx}")
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.legend()
    plt.tight_layout()
    plt.savefig(plot_path("u_applied_vs_us_last_episode.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    V_final_last = final_lyap_value[start_idx:]
    delta_V_last = np.full_like(V_final_last, np.nan)
    if V_final_last.size >= 2:
        delta_V_last[1:] = V_final_last[1:] - V_final_last[:-1]

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    axes[0].plot(last_steps, V_final_last, linewidth=2, label="final_lyap_value")
    axes[0].plot(last_steps, final_lyap_bound[start_idx:], linewidth=2, linestyle="--", label="final_lyap_bound")
    axes[0].grid(True, linestyle="--", alpha=0.35)
    axes[0].legend()
    axes[1].plot(last_steps, delta_V_last, linewidth=2, label="delta final_lyap_value")
    axes[1].axhline(0.0, color="black", linestyle="--", linewidth=1.0)
    axes[1].grid(True, linestyle="--", alpha=0.35)
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(plot_path("lyapunov_last_episode.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    plt.figure(figsize=(10, 5))
    plt.plot(time_u, final_lyap_margin, linewidth=2, label="final_lyap_margin")
    plt.plot(time_u, lyapunov_margin, linewidth=1.5, linestyle=":", label="candidate_lyap_margin")
    plt.axhline(0.0, color="black", linestyle="--", linewidth=1.0)
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path("lyapunov_margin.png"), dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(time_u, target_error_inf, linewidth=2, label="target_error_inf")
    plt.plot(time_u, target_slack_inf, linewidth=2, linestyle="--", label="target_slack_inf")
    if np.any(np.isfinite(target_mismatch_inf)):
        plt.plot(time_u, target_mismatch_inf, linewidth=2, linestyle=":", label="target_mismatch_inf")
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path("target_selector_metrics.png"), dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(time_u, selector_objective_value, linewidth=2, label="selector_objective")
    plt.plot(time_u, selector_target_tracking_term, linewidth=2, label="target_tracking")
    plt.plot(time_u, selector_u_applied_anchor_term, linewidth=2, label="u_applied_anchor")
    plt.plot(time_u, selector_u_prev_smoothing_term, linewidth=2, label="u_prev_smoothing")
    plt.plot(time_u, selector_x_prev_smoothing_term, linewidth=2, label="x_prev_smoothing")
    plt.plot(time_u, selector_xhat_anchor_term, linewidth=2, label="xhat_anchor")
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path("selector_objective_terms.png"), dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.semilogy(time_u, np.maximum(np.abs(selector_target_tracking_term), 1e-16), linewidth=2, label="target_tracking")
    plt.semilogy(time_u, np.maximum(np.abs(selector_x_prev_smoothing_term), 1e-16), linewidth=2, label="x_prev_smoothing")
    plt.semilogy(time_u, np.maximum(np.abs(selector_xhat_anchor_term), 1e-16), linewidth=2, label="xhat_anchor")
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path("selector_x_penalties.png"), dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(time_u, selector_x_s_minus_xhat_inf, linewidth=2, label="||x_s - xhat||_inf")
    plt.plot(time_u, selector_x_s_minus_xprev_inf, linewidth=2, label="||x_s - x_s_prev||_inf")
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path("selector_x_gaps.png"), dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(time_u, selector_dyn_residual_inf, linewidth=2, label="dyn_residual_inf")
    plt.plot(time_u, selector_bound_violation_inf, linewidth=2, linestyle="--", label="bound_violation_inf")
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path("selector_residuals.png"), dpi=300, bbox_inches="tight")
    plt.close()

    fig, axes = plt.subplots(n_y, 1, figsize=(10, 3.0 * n_y), sharex=True)
    axes = np.atleast_1d(axes)
    y_system_at_u = y_system[:-1, :]
    for idx in range(n_y):
        ax = axes[idx]
        ax.plot(time_u, y_system_at_u[:, idx] - y_target_plot[:, idx], linewidth=2, label=f"y_system - y_s [{idx}]")
        ax.plot(time_u, yhat_plot[:, idx] - y_target_plot[:, idx], linewidth=2, linestyle="--", label=f"yhat - y_s [{idx}]")
        ax.axhline(0.0, color="black", linestyle="--", linewidth=1.0)
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.legend()
    plt.tight_layout()
    plt.savefig(plot_path("y_minus_ys.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    if not paper_style:
        fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
        axes[0].step(time_u, target_failure_flags, where="post", linewidth=2, label="target_selector_failed")
        axes[0].step(time_u, target_success_flags, where="post", linewidth=1.5, linestyle="--", label="target_selector_success")
        axes[0].set_ylim(-0.05, 1.05)
        axes[0].set_yticks([0.0, 1.0])
        axes[0].grid(True, linestyle="--", alpha=0.35)
        axes[0].legend()
        axes[1].step(time_u, target_stage_code, where="post", linewidth=2, label="target_stage_code")
        axes[1].set_ylim(-0.25, max(_TARGET_STAGE_TICKS) + 0.25)
        axes[1].set_yticks(_TARGET_STAGE_TICKS)
        axes[1].set_yticklabels(_TARGET_STAGE_LABELS)
        axes[1].grid(True, linestyle="--", alpha=0.35)
        axes[1].legend()
        axes[2].step(time_u, effective_target_reused_flags, where="post", linewidth=2, label="effective_target_reused")
        axes[2].set_ylim(-0.05, 1.05)
        axes[2].set_yticks([0.0, 1.0])
        axes[2].grid(True, linestyle="--", alpha=0.35)
        axes[2].legend()
        plt.tight_layout()
        plt.savefig(plot_path("target_selector_status.png"), dpi=300, bbox_inches="tight")
        plt.close(fig)

        fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
        if first_step_replacement_mode:
            axes[0].step(time_u, candidate_first_step_lyap_ok_flags, where="post", linewidth=2, label="candidate_first_step_ok")
            axes[0].step(time_u, first_step_contraction_triggered_flags, where="post", linewidth=1.5, linestyle="--", label="replacement_triggered")
        else:
            axes[0].step(time_u, qcqp_attempted_flags, where="post", linewidth=2, label="qcqp_attempted")
        axes[0].set_ylim(-0.05, 1.05)
        axes[0].set_yticks([0.0, 1.0])
        axes[0].grid(True, linestyle="--", alpha=0.35)
        axes[0].legend()
        if first_step_replacement_mode:
            axes[1].step(time_u, constrained_mpc_attempted_flags, where="post", linewidth=2, label="constrained_mpc_attempted")
            axes[1].step(time_u, constrained_mpc_solved_flags, where="post", linewidth=1.5, linestyle="--", label="constrained_mpc_solved")
        else:
            axes[1].step(time_u, qcqp_solved_flags, where="post", linewidth=2, label="qcqp_solved")
            axes[1].step(time_u, projection_active, where="post", linewidth=1.5, linestyle="--", label="qcqp_hard_accept")
        axes[1].set_ylim(-0.05, 1.05)
        axes[1].set_yticks([0.0, 1.0])
        axes[1].grid(True, linestyle="--", alpha=0.35)
        axes[1].legend()
        if first_step_replacement_mode:
            axes[2].step(time_u, constrained_mpc_applied_flags, where="post", linewidth=2, label="constrained_mpc_applied")
            axes[2].step(
                time_u,
                constrained_mpc_failed_applied_candidate_flags,
                where="post",
                linewidth=1.5,
                linestyle="--",
                label="constrained_failed_candidate_applied",
            )
        else:
            axes[2].step(time_u, qcqp_hard_accepted_flags, where="post", linewidth=2, label="qcqp_hard_accepted")
            axes[2].step(time_u, bundle["fallback_verified_flags"], where="post", linewidth=1.5, linestyle="--", label="fallback_verified")
        axes[2].set_ylim(-0.05, 1.05)
        axes[2].set_yticks([0.0, 1.0])
        axes[2].grid(True, linestyle="--", alpha=0.35)
        axes[2].legend()
        plt.tight_layout()
        plt.savefig(plot_path("qcqp_status.png"), dpi=300, bbox_inches="tight")
        plt.close(fig)

        plt.figure(figsize=(10, 5))
        plt.plot(time_u, rewards, linewidth=1.7, label="training reward")
        plt.plot(time_u, reward_no_penalty, linewidth=1.7, linestyle="-.", label="reward_no_penalty")
        reward_ma = _moving_average(rewards, max(1, int(bundle.get("time_in_sub_episodes", 1))))
        reward_no_penalty_ma = _moving_average(
            reward_no_penalty,
            max(1, int(bundle.get("time_in_sub_episodes", 1))),
        )
        if reward_ma.size == len(time_u):
            plt.plot(time_u, reward_ma, linewidth=2, linestyle="--", label="episode-window avg training reward")
        if reward_no_penalty_ma.size == len(time_u):
            plt.plot(
                time_u,
                reward_no_penalty_ma,
                linewidth=2,
                linestyle=":",
                label="episode-window avg reward_no_penalty",
            )
        plt.grid(True, linestyle="--", alpha=0.35)
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_path("reward_trace.png"), dpi=300, bbox_inches="tight")
        plt.close()

        episode_len = int(bundle.get("time_in_sub_episodes", len(time_u)))
        if episode_len > 0 and len(time_u) > 0:
            n_episodes = int(np.ceil(len(time_u) / float(episode_len)))
            rng = np.random.default_rng(0)
            episode_plot_dir = os.path.join(output_dir, _paper_dirname("episode_samples_by_tens", "ep_samples"))
            last_episode_dir = os.path.join(output_dir, _paper_dirname("last_episode_summary", "last_ep"))
            os.makedirs(episode_plot_dir, exist_ok=True)
            os.makedirs(last_episode_dir, exist_ok=True)

            fallback_mpc_active = np.array(
                [1.0 if str(mode).startswith("fallback_mpc") else 0.0 for mode in bundle["correction_modes"]],
                dtype=float,
            )
            safety_active_flags = _safety_active_flags_from_bundle(bundle)
            diagnostic_safety_active_flags = _diagnostic_safety_active_flags_from_bundle(bundle)
            episode_reward_mean = np.asarray(
                [
                    np.nanmean(rewards[idx * episode_len : min((idx + 1) * episode_len, len(rewards))])
                    for idx in range(n_episodes)
                ],
                dtype=float,
            )
            episode_reward_no_penalty_mean = np.asarray(
                [
                    np.nanmean(
                        reward_no_penalty[idx * episode_len : min((idx + 1) * episode_len, len(reward_no_penalty))]
                    )
                    for idx in range(n_episodes)
                ],
                dtype=float,
            )

            fig, axes = plt.subplots(4, 1, figsize=(10, 9), sharex=False)
            ep_time = np.arange(n_episodes)
            fallback_rate = np.asarray(
                [
                    np.nanmean(fallback_mpc_active[idx * episode_len : min((idx + 1) * episode_len, fallback_mpc_active.size)])
                    for idx in range(n_episodes)
                ],
                dtype=float,
            )
            safety_rate = np.asarray(
                [
                    np.nanmean(safety_active_flags[idx * episode_len : min((idx + 1) * episode_len, safety_active_flags.size)])
                    for idx in range(n_episodes)
                ],
                dtype=float,
            )
            diagnostic_safety_rate = np.asarray(
                [
                    np.nanmean(
                        diagnostic_safety_active_flags[
                            idx * episode_len : min((idx + 1) * episode_len, diagnostic_safety_active_flags.size)
                        ]
                    )
                    for idx in range(n_episodes)
                ],
                dtype=float,
            )
            axes[0].plot(ep_time + 1, episode_reward_mean, linewidth=1.8, label="episode avg training reward")
            axes[0].plot(
                ep_time + 1,
                episode_reward_no_penalty_mean,
                linewidth=1.8,
                linestyle="-.",
                label="episode avg reward_no_penalty",
            )
            axes[0].plot(
                ep_time + 1,
                _moving_average(episode_reward_mean, 5),
                linewidth=2.2,
                linestyle="--",
                label="5-episode avg training reward",
            )
            axes[0].plot(
                ep_time + 1,
                _moving_average(episode_reward_no_penalty_mean, 5),
                linewidth=2.2,
                linestyle=":",
                label="5-episode avg reward_no_penalty",
            )
            axes[0].set_ylabel("reward")
            axes[1].plot(ep_time + 1, safety_rate, linewidth=1.8, color="tab:red", label="safety active rate")
            axes[1].plot(
                ep_time + 1,
                diagnostic_safety_rate,
                linewidth=1.8,
                linestyle=":",
                color="tab:orange",
                label="diagnostic would-activate rate",
            )
            axes[1].plot(ep_time + 1, fallback_rate, linewidth=1.8, linestyle="--", color="tab:purple", label="fallback rate")
            axes[1].set_ylabel("rate")
            axes[2].plot(time_u, rewards, linewidth=1.0, color="tab:blue", alpha=0.65, label="step training reward")
            axes[2].plot(
                time_u,
                reward_no_penalty,
                linewidth=1.0,
                color="tab:green",
                alpha=0.65,
                label="step reward_no_penalty",
            )
            axes[2].plot(time_u, reward_ma, linewidth=2.0, color="tab:orange", label="episode-window training avg")
            axes[2].plot(
                time_u,
                reward_no_penalty_ma,
                linewidth=2.0,
                linestyle=":",
                color="tab:purple",
                label="episode-window no-penalty avg",
            )
            axes[2].set_ylabel("reward")
            axes[3].step(time_u, safety_active_flags[: len(time_u)], where="post", linewidth=1.8, color="tab:red", label="safety active")
            axes[3].step(
                time_u,
                diagnostic_safety_active_flags[: len(time_u)],
                where="post",
                linewidth=1.5,
                linestyle=":",
                color="tab:orange",
                label="diagnostic would activate",
            )
            axes[3].step(time_u, fallback_mpc_active[: len(time_u)], where="post", linewidth=1.4, linestyle="--", color="tab:purple", label="fallback active")
            axes[3].set_ylim(-0.05, 1.05)
            axes[3].set_yticks([0.0, 1.0])
            axes[3].set_ylabel("status")
            axes[3].set_xlabel("step / episode")
            for ax in axes:
                ax.grid(True, linestyle="--", alpha=0.35)
                ax.legend(loc="best")
            fig.tight_layout()
            fig.savefig(plot_path("reward_average_summary.png"), dpi=300, bbox_inches="tight")
            plt.close(fig)

            def _episode_sum(flags):
                flags = np.asarray(flags, dtype=float).reshape(-1)
                return np.asarray(
                    [
                        np.nansum(flags[idx * episode_len : min((idx + 1) * episode_len, flags.size)] > 0.5)
                        for idx in range(n_episodes)
                    ],
                    dtype=float,
                )

            actual_safety_count = _episode_sum(safety_active_flags)
            would_activate_count = _episode_sum(diagnostic_safety_active_flags)
            fallback_count = _episode_sum(fallback_mpc_active)
            contraction_failure_flags = np.asarray(candidate_first_step_lyap_ok_flags, dtype=float)
            contraction_failure_flags = np.where(
                np.isfinite(contraction_failure_flags),
                contraction_failure_flags < 0.5,
                diagnostic_safety_active_flags > 0.5,
            )
            contraction_failure_count = _episode_sum(contraction_failure_flags)

            fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
            axes[0].plot(ep_time + 1, would_activate_count, linewidth=1.8, color="tab:orange", label="would activate")
            axes[0].plot(
                ep_time + 1,
                _moving_average(would_activate_count, 10),
                linewidth=2.2,
                linestyle="--",
                color="tab:orange",
                label="would activate, 10-episode avg",
            )
            axes[0].plot(ep_time + 1, actual_safety_count, linewidth=1.8, color="tab:red", label="actual safety active")
            axes[0].plot(
                ep_time + 1,
                _moving_average(actual_safety_count, 10),
                linewidth=2.2,
                linestyle="--",
                color="tab:red",
                label="actual safety, 10-episode avg",
            )
            axes[0].set_ylabel("count / episode")
            axes[1].plot(ep_time + 1, contraction_failure_count, linewidth=1.8, color="tab:blue", label="contraction not satisfied")
            axes[1].plot(
                ep_time + 1,
                _moving_average(contraction_failure_count, 10),
                linewidth=2.2,
                linestyle="--",
                color="tab:blue",
                label="contraction failures, 10-episode avg",
            )
            axes[1].plot(ep_time + 1, fallback_count, linewidth=1.6, linestyle=":", color="tab:purple", label="actual fallback")
            axes[1].set_ylabel("count / episode")
            axes[1].set_xlabel("episode")
            for ax in axes:
                ax.grid(True, linestyle="--", alpha=0.35)
                ax.legend(loc="best")
            fig.tight_layout()
            fig.savefig(plot_path("activation_contraction_counts.png"), dpi=300, bbox_inches="tight")
            plt.close(fig)

            def _episode_plot_name(chosen_episode, block_start=None, block_end=None, *, last_episode=False):
                if last_episode:
                    if use_short_names:
                        return f"ep_{chosen_episode + 1:03d}_last.png"
                    return f"episode_{chosen_episode + 1:03d}_last.png"
                if use_short_names:
                    return f"ep_{chosen_episode + 1:03d}_{block_start + 1:03d}_{block_end:03d}.png"
                return f"episode_{chosen_episode + 1:03d}_from_{block_start + 1:03d}_{block_end:03d}.png"

            def _plot_episode_window(step_start, step_end, title, filepath):
                if step_start >= step_end:
                    return
                y_start = step_start
                y_end = min(step_end + 1, len(time_y))
                local_time_y = np.arange(y_end - y_start)
                local_time_u = np.arange(step_end - step_start)
                n_output_panels = min(n_y, 2)
                fig, axes = plt.subplots(n_output_panels + 3, 1, figsize=(10, 3.0 * (n_output_panels + 3)), sharex=False)
                axes = np.atleast_1d(axes)

                local_safety = safety_active_flags[step_start:step_end]
                local_diagnostic_safety = diagnostic_safety_active_flags[step_start:step_end]
                local_fallback = fallback_mpc_active[step_start:step_end]
                for idx in range(n_output_panels):
                    ax = axes[idx]
                    ax.plot(
                        local_time_y,
                        y_system[y_start:y_end, idx],
                        linewidth=2.0,
                        color="tab:blue",
                        label=f"output_{idx}",
                    )
                    ax.step(
                        local_time_u,
                        y_sp_plot[step_start:step_end, idx],
                        where="post",
                        linewidth=2.0,
                        linestyle="--",
                        color="tab:gray",
                        label=f"setpoint_{idx}",
                    )
                    _shade_active_intervals(ax, local_time_u, local_safety)
                    _shade_active_intervals(ax, local_time_u, local_diagnostic_safety, color="tab:orange", alpha=0.07)
                    ax.grid(True, linestyle="--", alpha=0.35)
                    ax.legend(loc="best")
                    ax.set_ylabel(f"y{idx}")

                input_ax = axes[n_output_panels]
                for idx in range(u_safe_dev.shape[1]):
                    input_ax.plot(
                        local_time_u,
                        u_cand_dev[step_start:step_end, idx],
                        linewidth=1.3,
                        linestyle="--",
                        label=f"candidate u{idx}",
                    )
                    input_ax.step(
                        local_time_u,
                        u_safe_dev[step_start:step_end, idx],
                        where="post",
                        linewidth=1.8,
                        label=f"executed u{idx}",
                    )
                fallback_dev = np.asarray(bundle.get("u_fallback_mpc_dev_store", []), dtype=float)
                if fallback_dev.ndim == 2 and fallback_dev.shape[0] >= step_end:
                    for idx in range(fallback_dev.shape[1]):
                        if np.any(np.isfinite(fallback_dev[step_start:step_end, idx])):
                            input_ax.step(
                                local_time_u,
                                fallback_dev[step_start:step_end, idx],
                                where="post",
                                linewidth=1.0,
                                linestyle=":",
                                label=f"fallback u{idx}",
                            )
                _shade_active_intervals(input_ax, local_time_u, local_safety)
                _shade_active_intervals(input_ax, local_time_u, local_diagnostic_safety, color="tab:orange", alpha=0.07)
                input_ax.set_ylabel("u dev")
                input_ax.grid(True, linestyle="--", alpha=0.35)
                input_ax.legend(loc="best", ncol=2)

                status_ax = axes[n_output_panels + 1]
                status_ax.step(local_time_u, local_safety, where="post", linewidth=2.0, color="tab:red", label="safety active")
                status_ax.step(
                    local_time_u,
                    local_diagnostic_safety,
                    where="post",
                    linewidth=1.8,
                    linestyle=":",
                    color="tab:orange",
                    label="diagnostic would activate",
                )
                status_ax.step(local_time_u, local_fallback, where="post", linewidth=1.5, linestyle="--", color="tab:purple", label="fallback active")
                status_ax.step(local_time_u, projection_active[step_start:step_end], where="post", linewidth=1.2, linestyle=":", color="tab:orange", label="optimized correction")
                status_ax.set_ylabel("safety")
                detail_ax = axes[n_output_panels + 2]
                if first_step_replacement_mode:
                    detail_ax.step(local_time_u, candidate_first_step_lyap_ok_flags[step_start:step_end], where="post", linewidth=2.0, color="tab:blue", label="candidate_first_step_ok")
                    detail_ax.step(local_time_u, first_step_contraction_triggered_flags[step_start:step_end], where="post", linewidth=2.0, color="tab:orange", label="replacement_triggered")
                    detail_ax.step(local_time_u, constrained_mpc_applied_flags[step_start:step_end], where="post", linewidth=2.0, color="tab:green", label="constrained_mpc_applied")
                    detail_ax.step(local_time_u, constrained_mpc_failed_applied_candidate_flags[step_start:step_end], where="post", linewidth=1.5, linestyle="--", color="tab:red", label="constrained_failed_candidate_applied")
                    detail_ax.set_ylabel("replacement")
                else:
                    detail_ax.step(local_time_u, qcqp_attempted_flags[step_start:step_end], where="post", linewidth=2.0, color="tab:orange", label="qcqp_attempted")
                    detail_ax.step(local_time_u, qcqp_hard_accepted_flags[step_start:step_end], where="post", linewidth=2.0, color="tab:green", label="qcqp_hard_accepted")
                    detail_ax.step(local_time_u, fallback_verified_flags[step_start:step_end], where="post", linewidth=1.5, linestyle="--", color="tab:purple", label="fallback_verified")
                    detail_ax.set_ylabel("gate detail")
                for ax in (status_ax, detail_ax):
                    ax.set_ylim(-0.05, 1.05)
                    ax.set_yticks([0.0, 1.0])
                    ax.grid(True, linestyle="--", alpha=0.35)
                    ax.legend(loc="best")
                detail_ax.set_xlabel("step in episode")
                fig.suptitle(title, fontsize=12)
                plt.tight_layout()
                plt.savefig(filepath, dpi=300, bbox_inches="tight")
                plt.close(fig)

            block_starts = list(range(0, n_episodes, 10))[:20]
            for block_start in block_starts:
                block_end = min(block_start + 10, n_episodes)
                if block_start >= block_end:
                    continue
                chosen_episode = int(rng.integers(block_start, block_end))
                step_start = chosen_episode * episode_len
                step_end = min((chosen_episode + 1) * episode_len, len(time_u))
                _plot_episode_window(
                    step_start,
                    step_end,
                    f"Episode {chosen_episode + 1} sampled from block {block_start + 1}-{block_end}",
                    os.path.join(
                        episode_plot_dir,
                        _episode_plot_name(
                            chosen_episode,
                            block_start,
                            block_end,
                        ),
                    ),
                )

            last_episode = max(n_episodes - 1, 0)
            step_start = last_episode * episode_len
            step_end = min((last_episode + 1) * episode_len, len(time_u))
            _plot_episode_window(
                step_start,
                step_end,
                f"Last episode ({last_episode + 1})",
                os.path.join(
                    last_episode_dir,
                    _episode_plot_name(last_episode, last_episode=True),
                ),
            )

    if not paper_style:
        mode_counts = bundle["summary"]["mode_counts"]
        if mode_counts:
            labels = list(mode_counts.keys())
            values = [mode_counts[k] for k in labels]
            plt.figure(figsize=(10, 4))
            plt.bar(labels, values)
            plt.xticks(rotation=30, ha="right")
            plt.tight_layout()
            plt.savefig(plot_path("correction_modes.png"), dpi=300, bbox_inches="tight")
            plt.close()

        solver_counts = bundle["summary"].get("solver_status_counts", {})
        if solver_counts:
            labels = list(solver_counts.keys())
            values = [solver_counts[k] for k in labels]
            plt.figure(figsize=(10, 4))
            plt.bar(labels, values)
            plt.xticks(rotation=30, ha="right")
            plt.tight_layout()
            plt.savefig(plot_path("solver_status_counts.png"), dpi=300, bbox_inches="tight")
            plt.close()

        fallback_counts = bundle["summary"].get("fallback_solver_status_counts", {})
        if fallback_counts:
            labels = list(fallback_counts.keys())
            values = [fallback_counts[k] for k in labels]
            plt.figure(figsize=(10, 4))
            plt.bar(labels, values)
            plt.xticks(rotation=30, ha="right")
            plt.tight_layout()
            plt.savefig(plot_path("fallback_solver_status_counts.png"), dpi=300, bbox_inches="tight")
            plt.close()


def plot_safety_filter_bundle(bundle, output_dir, paper_style=False):
    if not paper_style:
        return _plot_safety_filter_bundle_impl(bundle, output_dir, paper_style=False)
    with paper_plot_context():
        return _plot_safety_filter_bundle_impl(bundle, output_dir, paper_style=True)


def plot_safety_filter_diagnostic_only(bundle, output_dir):
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib is required for plotting debug artifacts.")

    os.makedirs(output_dir, exist_ok=True)

    y_system = np.asarray(bundle["y_system"], float)
    y_sp = np.asarray(bundle["y_sp"], float)
    rewards = np.asarray(bundle["rewards"], float).reshape(-1)
    reward_no_penalty = _bundle_reward_no_penalty_array(bundle, rewards)
    target_success_flags = np.asarray(bundle["target_success_flags"], float)
    target_failure_flags = np.asarray(bundle["target_failure_flags"], float)
    target_stage_code = np.asarray(bundle["target_stage_code"], float)
    effective_target_reused_flags = np.asarray(bundle["effective_target_reused_flags"], float)
    qcqp_attempted_flags = np.asarray(bundle["qcqp_attempted_flags"], float)
    qcqp_solved_flags = np.asarray(bundle["qcqp_solved_flags"], float)
    qcqp_hard_accepted_flags = np.asarray(bundle["qcqp_hard_accepted_flags"], float)
    candidate_first_step_lyap_ok_flags = np.asarray(bundle["candidate_first_step_lyap_ok_flags"], float)
    first_step_contraction_triggered_flags = np.asarray(bundle["first_step_contraction_triggered_flags"], float)
    constrained_mpc_attempted_flags = np.asarray(bundle["constrained_mpc_attempted_flags"], float)
    constrained_mpc_solved_flags = np.asarray(bundle["constrained_mpc_solved_flags"], float)
    constrained_mpc_applied_flags = np.asarray(bundle["constrained_mpc_applied_flags"], float)
    constrained_mpc_failed_applied_candidate_flags = np.asarray(bundle["constrained_mpc_failed_applied_candidate_flags"], float)
    first_step_contraction_satisfied_applied_flags = np.asarray(bundle["first_step_contraction_satisfied_applied_flags"], float)
    projection_active = np.asarray(bundle["projection_active_flags"], float)
    fallback_verified_flags = np.asarray(bundle["fallback_verified_flags"], float)

    time_y = np.arange(y_system.shape[0])
    time_u = np.arange(bundle["u_applied_phys"].shape[0])
    n_y = y_system.shape[1]
    first_step_replacement_mode = bool(
        "first_step" in str(bundle.get("source", "")).lower()
        and (
            np.any(constrained_mpc_attempted_flags > 0.5)
            or np.any(np.isfinite(candidate_first_step_lyap_ok_flags))
            or np.any(first_step_contraction_triggered_flags > 0.5)
        )
    )

    if bundle.get("steady_states") is not None and bundle.get("data_min") is not None and bundle.get("data_max") is not None:
        data_min = bundle["data_min"]
        data_max = bundle["data_max"]
        n_u = bundle["u_applied_phys"].shape[1]
        y_ss_scaled = apply_min_max(bundle["steady_states"]["y_ss"], data_min[n_u:], data_max[n_u:])
        y_sp_plot = reverse_min_max(y_sp + y_ss_scaled, data_min[n_u:], data_max[n_u:])
    else:
        y_sp_plot = y_sp

    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    axes[0].step(time_u, target_failure_flags, where="post", linewidth=2, label="target_selector_failed")
    axes[0].step(time_u, target_success_flags, where="post", linewidth=1.5, linestyle="--", label="target_selector_success")
    axes[0].set_ylim(-0.05, 1.05)
    axes[0].set_yticks([0.0, 1.0])
    axes[0].grid(True, linestyle="--", alpha=0.35)
    axes[0].legend()
    axes[1].step(time_u, target_stage_code, where="post", linewidth=2, label="target_stage_code")
    axes[1].set_ylim(-0.25, max(_TARGET_STAGE_TICKS) + 0.25)
    axes[1].set_yticks(_TARGET_STAGE_TICKS)
    axes[1].set_yticklabels(_TARGET_STAGE_LABELS)
    axes[1].grid(True, linestyle="--", alpha=0.35)
    axes[1].legend()
    axes[2].step(time_u, effective_target_reused_flags, where="post", linewidth=2, label="effective_target_reused")
    axes[2].set_ylim(-0.05, 1.05)
    axes[2].set_yticks([0.0, 1.0])
    axes[2].grid(True, linestyle="--", alpha=0.35)
    axes[2].legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "target_selector_status.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    if first_step_replacement_mode:
        axes[0].step(time_u, candidate_first_step_lyap_ok_flags, where="post", linewidth=2, label="candidate_first_step_ok")
        axes[0].step(time_u, first_step_contraction_triggered_flags, where="post", linewidth=1.5, linestyle="--", label="replacement_triggered")
    else:
        axes[0].step(time_u, qcqp_attempted_flags, where="post", linewidth=2, label="qcqp_attempted")
    axes[0].set_ylim(-0.05, 1.05)
    axes[0].set_yticks([0.0, 1.0])
    axes[0].grid(True, linestyle="--", alpha=0.35)
    axes[0].legend()
    if first_step_replacement_mode:
        axes[1].step(time_u, constrained_mpc_attempted_flags, where="post", linewidth=2, label="constrained_mpc_attempted")
        axes[1].step(time_u, constrained_mpc_solved_flags, where="post", linewidth=1.5, linestyle="--", label="constrained_mpc_solved")
    else:
        axes[1].step(time_u, qcqp_solved_flags, where="post", linewidth=2, label="qcqp_solved")
        axes[1].step(time_u, projection_active, where="post", linewidth=1.5, linestyle="--", label="qcqp_hard_accept")
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].set_yticks([0.0, 1.0])
    axes[1].grid(True, linestyle="--", alpha=0.35)
    axes[1].legend()
    if first_step_replacement_mode:
        axes[2].step(time_u, constrained_mpc_applied_flags, where="post", linewidth=2, label="constrained_mpc_applied")
        axes[2].step(
            time_u,
            constrained_mpc_failed_applied_candidate_flags,
            where="post",
            linewidth=1.5,
            linestyle="--",
            label="constrained_failed_candidate_applied",
        )
    else:
        axes[2].step(time_u, qcqp_hard_accepted_flags, where="post", linewidth=2, label="qcqp_hard_accepted")
        axes[2].step(time_u, fallback_verified_flags, where="post", linewidth=1.5, linestyle="--", label="fallback_verified")
    axes[2].set_ylim(-0.05, 1.05)
    axes[2].set_yticks([0.0, 1.0])
    axes[2].grid(True, linestyle="--", alpha=0.35)
    axes[2].legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "qcqp_status.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    plt.figure(figsize=(10, 5))
    plt.plot(time_u, rewards, linewidth=1.7, label="training reward")
    plt.plot(time_u, reward_no_penalty, linewidth=1.7, linestyle="-.", label="reward_no_penalty")
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "reward_trace.png"), dpi=300, bbox_inches="tight")
    plt.close()

    mode_counts = bundle["summary"]["mode_counts"]
    if mode_counts:
        labels = list(mode_counts.keys())
        values = [mode_counts[k] for k in labels]
        plt.figure(figsize=(10, 4))
        plt.bar(labels, values)
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "correction_modes.png"), dpi=300, bbox_inches="tight")
        plt.close()

    solver_counts = bundle["summary"].get("solver_status_counts", {})
    if solver_counts:
        labels = list(solver_counts.keys())
        values = [solver_counts[k] for k in labels]
        plt.figure(figsize=(10, 4))
        plt.bar(labels, values)
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "solver_status_counts.png"), dpi=300, bbox_inches="tight")
        plt.close()

    fallback_counts = bundle["summary"].get("fallback_solver_status_counts", {})
    if fallback_counts:
        labels = list(fallback_counts.keys())
        values = [fallback_counts[k] for k in labels]
        plt.figure(figsize=(10, 4))
        plt.bar(labels, values)
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "fallback_solver_status_counts.png"), dpi=300, bbox_inches="tight")
        plt.close()

def save_safety_filter_debug_artifacts(
    bundle,
    directory=None,
    prefix_name="safety_filter_debug",
    save_plots=True,
    export_profile="debug",
):
    export_profile = _normalize_export_profile(export_profile)
    if directory is None:
        directory = os.fspath(repo_root())
    else:
        directory = os.fspath(resolve_repo_path(directory))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = _select_safety_debug_output_dir(
        directory,
        prefix_name,
        timestamp=timestamp,
    )
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(_jsonable(bundle["summary"]), f, indent=2)

    summary_csv_records = [
        {"key": key, "value": json.dumps(_jsonable(value))}
        for key, value in bundle["summary"].items()
    ]
    _write_csv(os.path.join(out_dir, "summary.csv"), summary_csv_records)

    step_records = _filter_safety_step_records(
        make_safety_filter_step_records(bundle["lyap_info_storage"]),
        export_profile=export_profile,
    )
    _write_csv(os.path.join(out_dir, "step_table.csv"), step_records)

    episode_records = make_safety_filter_episode_records(bundle)
    _write_csv(os.path.join(out_dir, "episode_table.csv"), episode_records)

    _save_npz(os.path.join(out_dir, "arrays.npz"), bundle, export_profile=export_profile)

    if save_plots:
        plot_safety_filter_bundle(
            bundle,
            os.path.join(out_dir, "plots"),
            paper_style=(export_profile == "compact"),
        )

    return out_dir


def save_safety_filter_comparison_artifacts(
    records,
    bundles_by_case,
    study_root,
    *,
    save_plots=True,
):
    os.makedirs(study_root, exist_ok=True)
    records = [dict(record) for record in records]

    comparison_csv = os.path.join(study_root, "comparison_table.csv")
    _write_csv(comparison_csv, records)

    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "n_cases": len(records),
        "case_names": [str(record.get("case_name")) for record in records],
        "comparison_table_csv": comparison_csv,
        "case_debug_dirs": {
            str(record.get("case_name")): record.get("debug_dir")
            for record in records
        },
    }

    plot_paths = {}
    if save_plots:
        if not HAS_MATPLOTLIB:
            raise ImportError("matplotlib is required to save safety comparison plots.")
        plot_dir = os.path.join(study_root, "comparison_plots")
        plot_paths["reward_mean"] = _save_safety_comparison_bar(
            records,
            ["reward_mean"],
            ["reward mean"],
            "reward",
            "RL Safety Gate Reward Comparison",
            _comparison_plot_path(plot_dir, "comparison_reward_mean.png"),
        )
        if records and any(record.get("reward_no_penalty_mean") is not None for record in records):
            plot_paths["reward_no_penalty_mean"] = _save_safety_comparison_bar(
                records,
                ["reward_no_penalty_mean"],
                ["reward without penalty mean"],
                "reward without penalty",
                "RL Safety Gate Reward Without Fallback Penalty",
                _comparison_plot_path(plot_dir, "comparison_reward_no_penalty_mean.png"),
            )
        if records and any(record.get("episode_reward_mean") is not None for record in records):
            plot_paths["average_episode_reward"] = _save_safety_comparison_bar(
                records,
                ["episode_reward_mean"],
                ["episode average reward"],
                "reward",
                "RL Safety Gate Average Episode Reward",
                _comparison_plot_path(plot_dir, "comparison_average_episode_reward.png"),
            )
        output_rmse_keys = [
            key for key in records[0].keys()
            if key.startswith("output") and key.endswith("_rmse")
        ] if records else []
        if output_rmse_keys:
            plot_paths["output_rmse"] = _save_safety_comparison_bar(
                records,
                output_rmse_keys,
                output_rmse_keys,
                "RMSE (physical units)",
                "RL Safety Gate Output RMSE",
                _comparison_plot_path(plot_dir, "comparison_output_rmse.png"),
            )
        plot_paths["rates"] = _save_safety_comparison_bar(
            records,
            ["accepted_rate", "verified_rate", "fallback_rate"],
            ["accepted", "verified", "fallback"],
            "rate",
            "RL Safety Gate Acceptance And Fallback Rates",
            _comparison_plot_path(plot_dir, "comparison_rates.png"),
        )
        plot_paths["correction_modes"] = _save_safety_mode_stacked_bar(
            records,
            _comparison_plot_path(plot_dir, "comparison_correction_modes.png"),
        )
        gap_plot = _save_safety_distribution_boxplot(
            [
                (
                    case_name,
                    np.asarray(bundle.get("executed_action_gap_inf", []), dtype=float)[
                        np.isfinite(np.asarray(bundle.get("executed_action_gap_inf", []), dtype=float))
                    ],
                )
                for case_name, bundle in bundles_by_case.items()
            ],
            "||u_exec - u_rl||_inf",
            "Executed Action Gap Distribution",
            _comparison_plot_path(plot_dir, "comparison_executed_action_gap_box.png"),
        )
        if gap_plot is not None:
            plot_paths["executed_action_gap_box"] = gap_plot
        fallback_episode_plot = _save_safety_distribution_boxplot(
            [
                (case_name, _episode_fallback_or_would_activate_counts(case_name, bundle))
                for case_name, bundle in bundles_by_case.items()
            ],
            "fallback count per episode (MPC-only uses would-be gate activations)",
            "Fallback Count Per Episode And MPC-only Would-be Activations",
            _comparison_plot_path(plot_dir, "comparison_fallback_count_per_episode_box.png"),
        )
        if fallback_episode_plot is not None:
            plot_paths["fallback_count_per_episode_box"] = fallback_episode_plot
        if records and any(record.get("wall_clock_seconds_per_step") is not None for record in records):
            plot_paths["wall_clock"] = _save_safety_comparison_bar(
                records,
                ["wall_clock_seconds", "wall_clock_seconds_per_step"],
                ["total seconds", "seconds per step"],
                "wall-clock time",
                "Wall-clock Runtime Comparison",
                _comparison_plot_path(plot_dir, "comparison_wall_clock_runtime.png"),
            )
        plot_paths["target_diagnostics"] = _save_safety_comparison_bar(
            records,
            ["target_cond_M_max", "target_residual_total_norm_max"],
            ["cond_M max", "target residual max"],
            "diagnostic value",
            "Direct-Target Diagnostic Comparison",
            _comparison_plot_path(plot_dir, "comparison_target_diagnostics.png"),
        )
        plot_paths.update(_save_safety_last_episode_overlays(bundles_by_case, plot_dir))

    summary["plot_paths"] = plot_paths
    comparison_summary_json = os.path.join(study_root, "comparison_summary.json")
    with open(comparison_summary_json, "w", encoding="utf-8") as f:
        json.dump(_jsonable(summary), f, indent=2)

    return {
        "comparison_table_csv": comparison_csv,
        "comparison_summary_json": comparison_summary_json,
        "plot_paths": plot_paths,
    }


def save_lyap_debug_artifacts(
    bundle,
    directory=None,
    prefix_name="safety_filter_debug",
    save_plots=True,
):
    return save_safety_filter_debug_artifacts(
        bundle=bundle,
        directory=directory,
        prefix_name=prefix_name,
        save_plots=save_plots,
    )


def load_safety_filter_debug_bundle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def load_lyap_debug_artifacts(path):
    return load_safety_filter_debug_bundle(path)
