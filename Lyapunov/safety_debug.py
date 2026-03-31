import csv
import json
import os
import pickle
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

try:
    from Plotting_fns.rl_plots import save_rl_summary_plots_from_bundle

    HAS_RL_SUMMARY_PLOTS = True
except Exception:
    save_rl_summary_plots_from_bundle = None
    HAS_RL_SUMMARY_PLOTS = False


_TARGET_STAGE_CODE_MAP = {
    "failed": 0.0,
    "legacy_slack_target": 1.0,
    "refined_step_a": 2.0,
}

_TARGET_STAGE_TICKS = [0.0, 1.0, 2.0]
_TARGET_STAGE_LABELS = [
    "failed",
    "legacy_slack_target",
    "refined_step_a",
]


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


def make_safety_filter_step_records(lyap_info_storage):
    records = []
    for step_idx, info in enumerate(lyap_info_storage):
        target_info = _target_info(info)
        selector = _selector_debug(info)
        selector_terms = info.get("selector_objective_terms") or {}
        upstream_info = info.get("upstream_candidate_info", {})
        target_success = bool(info.get("target_success", False))
        row = {
            "step": int(step_idx),
            "step_idx": int(step_idx),
            "source": info.get("source"),
            "accepted": bool(info.get("accepted", False)),
            "verified": bool(info.get("verified", False)),
            "accept_reason": info.get("accept_reason"),
            "reject_reason": info.get("reject_reason"),
            "correction_mode": info.get("correction_mode"),
            "projection_active": bool(str(info.get("correction_mode", "")) == "optimized_correction"),
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
            "target_source": info.get("target_source"),
            "selector_mode": info.get("selector_mode"),
            "effective_selector_mode": info.get("effective_selector_mode"),
            "selector_name": info.get("selector_name"),
            "current_target_success": info.get("current_target_success"),
            "current_target_stage": info.get("current_target_stage"),
            "effective_target_success": info.get("effective_target_success"),
            "effective_target_source": info.get("effective_target_source"),
            "effective_target_stage": info.get("effective_target_stage"),
            "effective_target_reused": info.get("effective_target_reused"),
            "d_s_minus_dhat_inf": info.get("d_s_minus_dhat_inf"),
            "d_s_frozen": info.get("d_s_frozen"),
            "d_s_optimized": info.get("d_s_optimized"),
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
            "mpc_tracking_target": json.dumps(_jsonable(info.get("mpc_tracking_target"))),
            "qcqp_tracking_target": json.dumps(_jsonable(info.get("qcqp_tracking_target"))),
            "y_s": json.dumps(_jsonable(info.get("y_s"))),
            "r_s": json.dumps(_jsonable(info.get("r_s"))),
            "x_s": json.dumps(_jsonable(info.get("x_s"))),
            "d_s": json.dumps(_jsonable(info.get("d_s"))),
            "cx_s": json.dumps(_jsonable(info.get("cx_s"))),
            "cd_d_s": json.dumps(_jsonable(info.get("cd_d_s"))),
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
        "n_target_success": int(sum(bool(info.get("current_target_success", info.get("target_success", False))) for info in lyap_info_storage)),
        "n_effective_target_success": int(sum(bool(info.get("effective_target_success", False)) for info in lyap_info_storage)),
        "n_target_reused": int(sum(bool(info.get("effective_target_reused", False)) for info in lyap_info_storage)),
        "n_accepted_candidate": int(sum(mode == "accepted_candidate" for mode in modes)),
        "n_optimized_correction": int(sum(mode == "optimized_correction" for mode in modes)),
        "n_fallback_mpc_verified": int(sum(mode == "fallback_mpc_verified" for mode in modes)),
        "n_fallback_mpc_unverified": int(sum(mode == "fallback_mpc_unverified" for mode in modes)),
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
        "reward_min": float(np.min(bundle["rewards"])) if len(bundle["rewards"]) > 0 else None,
        "reward_max": float(np.max(bundle["rewards"])) if len(bundle["rewards"]) > 0 else None,
        "target_error_inf_max": float(np.nanmax(bundle["target_error_inf"])) if bundle["target_error_inf"].size > 0 else None,
        "target_slack_inf_max": float(np.nanmax(bundle["target_slack_inf"])) if bundle["target_slack_inf"].size > 0 else None,
        "lyapunov_margin_min": float(np.nanmin(bundle["lyapunov_margin"])) if bundle["lyapunov_margin"].size > 0 else None,
        "target_mismatch_inf_max": float(np.nanmax(bundle["target_mismatch_inf"])) if bundle["target_mismatch_inf"].size > 0 else None,
        "d_s_minus_dhat_inf_max": float(np.nanmax(bundle["d_s_minus_dhat_inf"])) if bundle["d_s_minus_dhat_inf"].size > 0 else None,
    }
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
        "u_target_dev_store": _stack_vectors(lyap_info_storage, "u_s", n_u),
        "u_fallback_mpc_dev_store": _stack_vectors(lyap_info_storage, "u_fallback_mpc", n_u),
        "x_target_store": _stack_vectors(lyap_info_storage, "x_s", xhatdhat.shape[0] - n_y),
        "d_target_store": _stack_vectors(lyap_info_storage, "d_s", n_y),
        "y_target_store": _stack_vectors(lyap_info_storage, "y_s", n_y),
        "r_target_store": _stack_vectors(lyap_info_storage, "r_s", y_sp.shape[1]),
        "cx_s_store": _stack_vectors(lyap_info_storage, "cx_s", n_y),
        "cd_d_s_store": _stack_vectors(lyap_info_storage, "cd_d_s", n_y),
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
        "warm_start_plot": None if extra is None else extra.get("warm_start_plot"),
        "start_plot_idx": 10 if extra is None else int(extra.get("start_plot_idx", 10)),
        "actor_losses": None if extra is None or extra.get("actor_losses") is None else np.asarray(extra.get("actor_losses"), float),
        "critic_losses": None if extra is None or extra.get("critic_losses") is None else np.asarray(extra.get("critic_losses"), float),
    }
    bundle["summary"] = summarize_safety_filter_bundle(bundle)
    return bundle


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


def _save_npz(path, bundle):
    np.savez_compressed(
        path,
        y_system=bundle["y_system"],
        u_applied_phys=bundle["u_applied_phys"],
        xhatdhat=bundle["xhatdhat"],
        y_sp=bundle["y_sp"],
        yhat=bundle["yhat"],
        e_store=bundle["e_store"],
        qi=bundle["qi"],
        qs=bundle["qs"],
        ha=bundle["ha"],
        u_safe_dev_store=bundle["u_safe_dev_store"],
        u_cand_dev_store=bundle["u_cand_dev_store"],
        u_prev_dev_store=bundle["u_prev_dev_store"],
        u_target_dev_store=bundle["u_target_dev_store"],
        u_fallback_mpc_dev_store=bundle["u_fallback_mpc_dev_store"],
        x_target_store=bundle["x_target_store"],
        d_target_store=bundle["d_target_store"],
        y_target_store=bundle["y_target_store"],
        r_target_store=bundle["r_target_store"],
        cx_s_store=bundle["cx_s_store"],
        cd_d_s_store=bundle["cd_d_s_store"],
        selector_target_tracking_term=bundle["selector_target_tracking_term"],
        selector_u_applied_anchor_term=bundle["selector_u_applied_anchor_term"],
        selector_u_prev_smoothing_term=bundle["selector_u_prev_smoothing_term"],
        selector_x_prev_smoothing_term=bundle["selector_x_prev_smoothing_term"],
        selector_xhat_anchor_term=bundle["selector_xhat_anchor_term"],
        selector_objective_value=bundle["selector_objective_value"],
        selector_x_s_minus_xhat_inf=bundle["selector_x_s_minus_xhat_inf"],
        selector_x_s_minus_xprev_inf=bundle["selector_x_s_minus_xprev_inf"],
        V_k=bundle["V_k"],
        V_next_first=bundle["V_next_first"],
        V_next_first_candidate=bundle["V_next_first_candidate"],
        V_next_first_applied=bundle["V_next_first_applied"],
        V_next_cand=bundle["V_next_cand"],
        V_bound=bundle["V_bound"],
        contraction_margin=bundle["contraction_margin"],
        contraction_margin_candidate=bundle["contraction_margin_candidate"],
        contraction_margin_applied=bundle["contraction_margin_applied"],
        first_step_contraction_satisfied_flags=bundle["first_step_contraction_satisfied_flags"],
        first_step_contraction_satisfied_applied_flags=bundle["first_step_contraction_satisfied_applied_flags"],
        final_lyap_value=bundle["final_lyap_value"],
        final_lyap_bound=bundle["final_lyap_bound"],
        final_lyap_margin=bundle["final_lyap_margin"],
        lyapunov_margin=bundle["lyapunov_margin"],
        target_error_inf=bundle["target_error_inf"],
        target_slack_inf=bundle["target_slack_inf"],
        selector_dyn_residual_inf=bundle["selector_dyn_residual_inf"],
        selector_bound_violation_inf=bundle["selector_bound_violation_inf"],
        target_mismatch_inf=bundle["target_mismatch_inf"],
        target_success_flags=bundle["target_success_flags"],
        target_failure_flags=bundle["target_failure_flags"],
        target_stage_code=bundle["target_stage_code"],
        effective_target_success_flags=bundle["effective_target_success_flags"],
        effective_target_reused_flags=bundle["effective_target_reused_flags"],
        d_s_minus_dhat_inf=bundle["d_s_minus_dhat_inf"],
        qcqp_attempted_flags=bundle["qcqp_attempted_flags"],
        qcqp_solved_flags=bundle["qcqp_solved_flags"],
        qcqp_hard_accepted_flags=bundle["qcqp_hard_accepted_flags"],
        candidate_first_step_lyap_ok_flags=bundle["candidate_first_step_lyap_ok_flags"],
        first_step_contraction_triggered_flags=bundle["first_step_contraction_triggered_flags"],
        constrained_mpc_attempted_flags=bundle["constrained_mpc_attempted_flags"],
        constrained_mpc_solved_flags=bundle["constrained_mpc_solved_flags"],
        constrained_mpc_applied_flags=bundle["constrained_mpc_applied_flags"],
        constrained_mpc_failed_applied_candidate_flags=bundle["constrained_mpc_failed_applied_candidate_flags"],
        verified_flags=bundle["verified_flags"],
        accepted_flags=bundle["accepted_flags"],
        fallback_verified_flags=bundle["fallback_verified_flags"],
        projection_active_flags=bundle["projection_active_flags"],
    )


def _plot_safety_filter_bundle_impl(bundle, output_dir):
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib is required for plotting debug artifacts.")

    os.makedirs(output_dir, exist_ok=True)
    paper_mode = "paper_plots" in os.path.normpath(output_dir).lower()

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
    rewards = bundle["rewards"]
    projection_active = np.asarray(bundle["projection_active_flags"], float)
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

    if bundle.get("steady_states") is not None and bundle.get("data_min") is not None and bundle.get("data_max") is not None:
        steady_states = bundle["steady_states"]
        data_min = bundle["data_min"]
        data_max = bundle["data_max"]
        u_ss_scaled = apply_min_max(steady_states["ss_inputs"], data_min[:u_applied_phys.shape[1]], data_max[:u_applied_phys.shape[1]])
        y_ss_scaled = apply_min_max(steady_states["y_ss"], data_min[u_applied_phys.shape[1]:], data_max[u_applied_phys.shape[1]:])
        u_target_plot = reverse_min_max(bundle["u_target_dev_store"] + u_ss_scaled, data_min[:u_applied_phys.shape[1]], data_max[:u_applied_phys.shape[1]])
        y_sp_plot = reverse_min_max(y_sp + y_ss_scaled, data_min[u_applied_phys.shape[1]:], data_max[u_applied_phys.shape[1]:])
        y_target_plot = reverse_min_max(y_target_store + y_ss_scaled, data_min[u_applied_phys.shape[1]:], data_max[u_applied_phys.shape[1]:])
        r_target_plot = reverse_min_max(r_target_store + y_ss_scaled, data_min[u_applied_phys.shape[1]:], data_max[u_applied_phys.shape[1]:])
        yhat_plot = reverse_min_max(bundle["yhat"].T + y_ss_scaled, data_min[u_applied_phys.shape[1]:], data_max[u_applied_phys.shape[1]:])
    else:
        u_target_plot = bundle["u_target_dev_store"]
        y_sp_plot = y_sp
        y_target_plot = y_target_store
        r_target_plot = r_target_store
        yhat_plot = bundle["yhat"].T

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
            ax.plot(time_vals, y_data[idx, :], linewidth=2.0, linestyle="--", color="tab:green", label=f"y_s_{idx}")
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
    state_target_dir = os.path.join(output_dir, "state_target_channels")
    _plot_state_target_channels(
        os.path.join(state_target_dir, "full_xhat_vs_xs"),
        xhatdhat[:n_x, :len(time_u)],
        x_target_store.T,
        "xhat_",
        "xs_",
        "tab:blue",
        "tab:red",
    )
    _plot_state_target_channels(
        os.path.join(state_target_dir, "full_dhat_vs_ds"),
        xhatdhat[n_x:, :len(time_u)],
        d_target_store.T,
        "dhat_",
        "ds_",
        "tab:orange",
        "tab:green",
    )
    _plot_target_vs_target_channels(
        os.path.join(state_target_dir, "full_ys_vs_ysp"),
        y_target_plot.T,
        y_sp_plot.T,
        "ys_",
        "ysp_",
        "tab:purple",
        "tab:gray",
    )
    _plot_target_vs_target_channels(
        os.path.join(state_target_dir, "full_rs_vs_ysp"),
        r_target_plot.T,
        y_sp_plot.T,
        "rs_",
        "ysp_",
        "tab:brown",
        "tab:gray",
    )
    _plot_decomposition_channels(
        os.path.join(state_target_dir, "full_ys_decomposition"),
        cx_s_store.T,
        cd_d_s_store.T,
        y_target_plot.T,
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
    axes[-1].set_xlabel("step")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "outputs_vs_ysp_vs_ys.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(n_y, 1, figsize=(10, 3.2 * n_y), sharex=True)
    axes = np.atleast_1d(axes)
    for idx in range(n_y):
        ax = axes[idx]
        ax.plot(time_u, cx_s_store[:, idx], linewidth=2.0, color="tab:blue", label=f"Cx_s_{idx}")
        ax.plot(time_u, cd_d_s_store[:, idx], linewidth=2.0, color="tab:orange", label=f"Cd_d_s_{idx}")
        ax.plot(time_u, y_target_plot[:, idx], linewidth=2.0, linestyle="--", color="tab:green", label=f"y_s_{idx}")
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.legend(loc="best")
    axes[-1].set_xlabel("step")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "ys_decomposition_summary.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    plt.figure(figsize=(10, 6))
    for idx in range(y_system.shape[1]):
        plt.subplot(y_system.shape[1], 1, idx + 1)
        plt.plot(time_y, y_system[:, idx], label="output", linewidth=2)
        plt.step(time_u, y_sp_plot[:, idx] if y_sp_plot.ndim == 2 else y_sp_plot, where="post", linestyle="--", label="setpoint", linewidth=2)
        plt.grid(True, linestyle="--", alpha=0.35)
        plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "outputs_vs_setpoint.png"), dpi=300, bbox_inches="tight")
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
    plt.savefig(os.path.join(output_dir, "outputs_vs_setpoint_projection.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    plt.figure(figsize=(10, 5))
    for idx in range(u_applied_phys.shape[1]):
        plt.step(time_u, u_applied_phys[:, idx], where="post", linewidth=2, label=f"u{idx}")
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "applied_inputs.png"), dpi=300, bbox_inches="tight")
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
    plt.savefig(os.path.join(output_dir, "u_applied_vs_us.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    plt.figure(figsize=(10, 5))
    for idx in range(u_safe_dev.shape[1]):
        plt.plot(time_u, u_cand_dev[:, idx], linestyle="--", linewidth=1.5, label=f"cand_{idx}")
        plt.plot(time_u, u_safe_dev[:, idx], linewidth=2.0, label=f"safe_{idx}")
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "candidate_vs_safe_dev.png"), dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(time_u, final_lyap_value, linewidth=2, label="final_lyap_value")
    plt.plot(time_u, final_lyap_bound, linewidth=2, linestyle="--", label="final_lyap_bound")
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "lyapunov_values.png"), dpi=300, bbox_inches="tight")
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
        plt.savefig(os.path.join(output_dir, "first_step_contraction_diagnostics.png"), dpi=300, bbox_inches="tight")
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
    cx_s_last = cx_s_store[start_idx:, :].T if cx_s_store.size > 0 else cx_s_store
    cd_d_s_last = cd_d_s_store[start_idx:, :].T if cd_d_s_store.size > 0 else cd_d_s_store
    _plot_augmented_states(xhatdhat_last, "xhatdhat_last_episode.png")
    _plot_augmented_states(xhatdhat_last, "dhat_last_episode.png", dhat_only=True)
    _plot_state_target_channels(
        os.path.join(state_target_dir, "last_episode_xhat_vs_xs"),
        xhatdhat[:n_x, start_idx:start_idx + x_target_last.shape[1]],
        x_target_last,
        "xhat_",
        "xs_",
        "tab:blue",
        "tab:red",
    )
    _plot_state_target_channels(
        os.path.join(state_target_dir, "last_episode_dhat_vs_ds"),
        xhatdhat[n_x:, start_idx:start_idx + d_target_last.shape[1]],
        d_target_last,
        "dhat_",
        "ds_",
        "tab:orange",
        "tab:green",
    )
    _plot_target_vs_target_channels(
        os.path.join(state_target_dir, "last_episode_ys_vs_ysp"),
        y_target_last,
        y_sp_last,
        "ys_",
        "ysp_",
        "tab:purple",
        "tab:gray",
    )
    _plot_target_vs_target_channels(
        os.path.join(state_target_dir, "last_episode_rs_vs_ysp"),
        r_target_last,
        y_sp_last,
        "rs_",
        "ysp_",
        "tab:brown",
        "tab:gray",
    )
    _plot_decomposition_channels(
        os.path.join(state_target_dir, "last_episode_ys_decomposition"),
        cx_s_last,
        cd_d_s_last,
        y_target_last,
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
    axes[-1].set_xlabel("step")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "outputs_vs_ysp_vs_ys_last_episode.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(n_y, 1, figsize=(10, 3.2 * n_y), sharex=True)
    axes = np.atleast_1d(axes)
    for idx in range(n_y):
        ax = axes[idx]
        ax.plot(last_steps, cx_s_last[idx, :], linewidth=2.0, color="tab:blue", label=f"Cx_s_{idx}")
        ax.plot(last_steps, cd_d_s_last[idx, :], linewidth=2.0, color="tab:orange", label=f"Cd_d_s_{idx}")
        ax.plot(last_steps, y_target_last[idx, :], linewidth=2.0, linestyle="--", color="tab:green", label=f"y_s_{idx}")
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.legend(loc="best")
    axes[-1].set_xlabel("step")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "ys_decomposition_summary_last_episode.png"), dpi=300, bbox_inches="tight")
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
    plt.savefig(os.path.join(output_dir, "y_minus_ys_last_episode.png"), dpi=300, bbox_inches="tight")
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
    plt.savefig(os.path.join(output_dir, "u_applied_vs_us_last_episode.png"), dpi=300, bbox_inches="tight")
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
    plt.savefig(os.path.join(output_dir, "lyapunov_last_episode.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    plt.figure(figsize=(10, 5))
    plt.plot(time_u, final_lyap_margin, linewidth=2, label="final_lyap_margin")
    plt.plot(time_u, lyapunov_margin, linewidth=1.5, linestyle=":", label="candidate_lyap_margin")
    plt.axhline(0.0, color="black", linestyle="--", linewidth=1.0)
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "lyapunov_margin.png"), dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(time_u, target_error_inf, linewidth=2, label="target_error_inf")
    plt.plot(time_u, target_slack_inf, linewidth=2, linestyle="--", label="target_slack_inf")
    if np.any(np.isfinite(target_mismatch_inf)):
        plt.plot(time_u, target_mismatch_inf, linewidth=2, linestyle=":", label="target_mismatch_inf")
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "target_selector_metrics.png"), dpi=300, bbox_inches="tight")
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
    plt.savefig(os.path.join(output_dir, "selector_objective_terms.png"), dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.semilogy(time_u, np.maximum(np.abs(selector_target_tracking_term), 1e-16), linewidth=2, label="target_tracking")
    plt.semilogy(time_u, np.maximum(np.abs(selector_x_prev_smoothing_term), 1e-16), linewidth=2, label="x_prev_smoothing")
    plt.semilogy(time_u, np.maximum(np.abs(selector_xhat_anchor_term), 1e-16), linewidth=2, label="xhat_anchor")
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "selector_x_penalties.png"), dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(time_u, selector_x_s_minus_xhat_inf, linewidth=2, label="||x_s - xhat||_inf")
    plt.plot(time_u, selector_x_s_minus_xprev_inf, linewidth=2, label="||x_s - x_s_prev||_inf")
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "selector_x_gaps.png"), dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(time_u, selector_dyn_residual_inf, linewidth=2, label="dyn_residual_inf")
    plt.plot(time_u, selector_bound_violation_inf, linewidth=2, linestyle="--", label="bound_violation_inf")
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "selector_residuals.png"), dpi=300, bbox_inches="tight")
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
    plt.savefig(os.path.join(output_dir, "y_minus_ys.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    if not paper_mode:
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
            axes[2].step(time_u, bundle["fallback_verified_flags"], where="post", linewidth=1.5, linestyle="--", label="fallback_verified")
        axes[2].set_ylim(-0.05, 1.05)
        axes[2].set_yticks([0.0, 1.0])
        axes[2].grid(True, linestyle="--", alpha=0.35)
        axes[2].legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "qcqp_status.png"), dpi=300, bbox_inches="tight")
        plt.close(fig)

        plt.figure(figsize=(10, 5))
        plt.plot(time_u, rewards, linewidth=2, label="reward")
        plt.grid(True, linestyle="--", alpha=0.35)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "reward_trace.png"), dpi=300, bbox_inches="tight")
        plt.close()

    episode_len = int(bundle.get("time_in_sub_episodes", len(time_u)))
    if (not paper_mode) and episode_len > 0 and len(time_u) > 0:
        n_episodes = int(np.ceil(len(time_u) / float(episode_len)))
        rng = np.random.default_rng(0)
        episode_plot_dir = os.path.join(output_dir, "episode_samples_by_tens")
        os.makedirs(episode_plot_dir, exist_ok=True)

        fallback_mpc_active = np.array(
            [1.0 if str(mode).startswith("fallback_mpc") else 0.0 for mode in bundle["correction_modes"]],
            dtype=float,
        )
        fallback_mpc_verified = np.asarray(bundle["fallback_verified_flags"], float)

        block_starts = list(range(0, n_episodes, 10))[:20]
        for block_start in block_starts:
            block_end = min(block_start + 10, n_episodes)
            if block_start >= block_end:
                continue
            chosen_episode = int(rng.integers(block_start, block_end))

            step_start = chosen_episode * episode_len
            step_end = min((chosen_episode + 1) * episode_len, len(time_u))
            if step_start >= step_end:
                continue

            y_start = step_start
            y_end = min(step_end + 1, len(time_y))

            local_time_y = np.arange(y_end - y_start)
            local_time_u = np.arange(step_end - step_start)

            fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=False)
            axes = np.atleast_1d(axes)

            for idx in range(min(n_y, 2)):
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
                ax.grid(True, linestyle="--", alpha=0.35)
                ax.legend(loc="best")
                ax.set_ylabel(f"y{idx}")

            if first_step_replacement_mode:
                axes[2].step(local_time_u, candidate_first_step_lyap_ok_flags[step_start:step_end], where="post", linewidth=2.0, color="tab:blue", label="candidate_first_step_ok")
                axes[2].step(local_time_u, first_step_contraction_triggered_flags[step_start:step_end], where="post", linewidth=2.0, color="tab:orange", label="replacement_triggered")
                axes[2].step(local_time_u, constrained_mpc_applied_flags[step_start:step_end], where="post", linewidth=2.0, color="tab:green", label="constrained_mpc_applied")
                axes[2].step(local_time_u, constrained_mpc_failed_applied_candidate_flags[step_start:step_end], where="post", linewidth=1.5, linestyle="--", color="tab:red", label="constrained_failed_candidate_applied")
                axes[2].set_ylabel("replacement")
                axes[3].step(local_time_u, first_step_contraction_satisfied_applied_flags[step_start:step_end], where="post", linewidth=2.0, color="tab:green", label="applied_first_step_ok")
                axes[3].step(local_time_u, constrained_mpc_solved_flags[step_start:step_end], where="post", linewidth=1.5, linestyle="--", color="tab:purple", label="constrained_mpc_solved")
                axes[3].set_ylabel("lyap")
            else:
                axes[2].step(
                    local_time_u,
                    qcqp_attempted_flags[step_start:step_end],
                    where="post",
                    linewidth=2.0,
                    color="tab:orange",
                    label="qcqp_attempted",
                )
                axes[2].step(
                    local_time_u,
                    qcqp_hard_accepted_flags[step_start:step_end],
                    where="post",
                    linewidth=2.0,
                    color="tab:green",
                    label="qcqp_hard_accepted",
                )
                axes[2].step(
                    local_time_u,
                    projection_active[step_start:step_end],
                    where="post",
                    linewidth=1.5,
                    linestyle="--",
                    color="tab:red",
                    label="optimized_correction",
                )
                axes[2].set_ylabel("projection")
                axes[3].step(
                    local_time_u,
                    fallback_mpc_active[step_start:step_end],
                    where="post",
                    linewidth=2.0,
                    color="tab:purple",
                    label="fallback_mpc_active",
                )
                axes[3].step(
                    local_time_u,
                    fallback_mpc_verified[step_start:step_end],
                    where="post",
                    linewidth=1.5,
                    linestyle="--",
                    color="tab:green",
                    label="fallback_mpc_verified",
                )
                axes[3].set_ylabel("fallback")
            axes[2].set_ylim(-0.05, 1.05)
            axes[2].set_yticks([0.0, 1.0])
            axes[2].grid(True, linestyle="--", alpha=0.35)
            axes[2].legend(loc="best")
            axes[3].set_ylim(-0.05, 1.05)
            axes[3].set_yticks([0.0, 1.0])
            axes[3].grid(True, linestyle="--", alpha=0.35)
            axes[3].legend(loc="best")
            axes[3].set_xlabel("step in episode")

            fig.suptitle(
                f"Episode {chosen_episode + 1} sampled from block {block_start + 1}-{block_end}",
                fontsize=12,
            )
            plt.tight_layout()
            plt.savefig(
                os.path.join(
                    episode_plot_dir,
                    f"episode_{chosen_episode + 1:03d}_from_{block_start + 1:03d}_{block_end:03d}.png",
                ),
                dpi=300,
                bbox_inches="tight",
            )
            plt.close(fig)

        last_episode_dir = os.path.join(output_dir, "last_episode_summary")
        os.makedirs(last_episode_dir, exist_ok=True)
        last_episode = max(n_episodes - 1, 0)
        step_start = last_episode * episode_len
        step_end = min((last_episode + 1) * episode_len, len(time_u))
        if step_start < step_end:
            y_start = step_start
            y_end = min(step_end + 1, len(time_y))
            local_time_y = np.arange(y_end - y_start)
            local_time_u = np.arange(step_end - step_start)

            fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=False)
            axes = np.atleast_1d(axes)

            for idx in range(min(n_y, 2)):
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
                ax.grid(True, linestyle="--", alpha=0.35)
                ax.legend(loc="best")
                ax.set_ylabel(f"y{idx}")

            if first_step_replacement_mode:
                axes[2].step(local_time_u, candidate_first_step_lyap_ok_flags[step_start:step_end], where="post", linewidth=2.0, color="tab:blue", label="candidate_first_step_ok")
                axes[2].step(local_time_u, first_step_contraction_triggered_flags[step_start:step_end], where="post", linewidth=2.0, color="tab:orange", label="replacement_triggered")
                axes[2].step(local_time_u, constrained_mpc_applied_flags[step_start:step_end], where="post", linewidth=2.0, color="tab:green", label="constrained_mpc_applied")
                axes[2].step(local_time_u, constrained_mpc_failed_applied_candidate_flags[step_start:step_end], where="post", linewidth=1.5, linestyle="--", color="tab:red", label="constrained_failed_candidate_applied")
                axes[2].set_ylabel("replacement")
                axes[3].step(local_time_u, first_step_contraction_satisfied_applied_flags[step_start:step_end], where="post", linewidth=2.0, color="tab:green", label="applied_first_step_ok")
                axes[3].step(local_time_u, constrained_mpc_solved_flags[step_start:step_end], where="post", linewidth=1.5, linestyle="--", color="tab:purple", label="constrained_mpc_solved")
                axes[3].set_ylabel("lyap")
            else:
                axes[2].step(
                    local_time_u,
                    qcqp_attempted_flags[step_start:step_end],
                    where="post",
                    linewidth=2.0,
                    color="tab:orange",
                    label="qcqp_attempted",
                )
                axes[2].step(
                    local_time_u,
                    qcqp_hard_accepted_flags[step_start:step_end],
                    where="post",
                    linewidth=2.0,
                    color="tab:green",
                    label="qcqp_hard_accepted",
                )
                axes[2].step(
                    local_time_u,
                    projection_active[step_start:step_end],
                    where="post",
                    linewidth=1.5,
                    linestyle="--",
                    color="tab:red",
                    label="optimized_correction",
                )
                axes[2].set_ylabel("projection")
                axes[3].step(
                    local_time_u,
                    fallback_mpc_active[step_start:step_end],
                    where="post",
                    linewidth=2.0,
                    color="tab:purple",
                    label="fallback_mpc_active",
                )
                axes[3].step(
                    local_time_u,
                    fallback_mpc_verified[step_start:step_end],
                    where="post",
                    linewidth=1.5,
                    linestyle="--",
                    color="tab:green",
                    label="fallback_mpc_verified",
                )
                axes[3].set_ylabel("fallback")
            axes[2].set_ylim(-0.05, 1.05)
            axes[2].set_yticks([0.0, 1.0])
            axes[2].grid(True, linestyle="--", alpha=0.35)
            axes[2].legend(loc="best")
            axes[3].set_ylim(-0.05, 1.05)
            axes[3].set_yticks([0.0, 1.0])
            axes[3].grid(True, linestyle="--", alpha=0.35)
            axes[3].legend(loc="best")
            axes[3].set_xlabel("step in episode")

            fig.suptitle(f"Last episode ({last_episode + 1})", fontsize=12)
            plt.tight_layout()
            plt.savefig(
                os.path.join(last_episode_dir, f"episode_{last_episode + 1:03d}_last.png"),
                dpi=300,
                bbox_inches="tight",
            )
            plt.close(fig)

    if not paper_mode:
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


def plot_safety_filter_bundle(bundle, output_dir, paper_style=False):
    if not paper_style:
        return _plot_safety_filter_bundle_impl(bundle, output_dir)
    with paper_plot_context():
        return _plot_safety_filter_bundle_impl(bundle, output_dir)


def plot_safety_filter_diagnostic_only(bundle, output_dir):
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib is required for plotting debug artifacts.")

    os.makedirs(output_dir, exist_ok=True)

    y_system = np.asarray(bundle["y_system"], float)
    y_sp = np.asarray(bundle["y_sp"], float)
    rewards = np.asarray(bundle["rewards"], float)
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
    plt.plot(time_u, rewards, linewidth=2, label="reward")
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

    episode_len = int(bundle.get("time_in_sub_episodes", len(time_u)))
    if episode_len <= 0 or len(time_u) <= 0:
        return

    n_episodes = int(np.ceil(len(time_u) / float(episode_len)))
    rng = np.random.default_rng(0)
    fallback_mpc_active = np.array(
        [1.0 if str(mode).startswith("fallback_mpc") else 0.0 for mode in bundle["correction_modes"]],
        dtype=float,
    )

    def _plot_episode_window(step_start, step_end, title, filepath):
        if step_start >= step_end:
            return
        y_start = step_start
        y_end = min(step_end + 1, len(time_y))
        local_time_y = np.arange(y_end - y_start)
        local_time_u = np.arange(step_end - step_start)
        fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=False)
        axes = np.atleast_1d(axes)
        for idx in range(min(n_y, 2)):
            ax = axes[idx]
            ax.plot(local_time_y, y_system[y_start:y_end, idx], linewidth=2.0, color="tab:blue", label=f"output_{idx}")
            ax.step(local_time_u, y_sp_plot[step_start:step_end, idx], where="post", linewidth=2.0, linestyle="--", color="tab:gray", label=f"setpoint_{idx}")
            ax.grid(True, linestyle="--", alpha=0.35)
            ax.legend(loc="best")
            ax.set_ylabel(f"y{idx}")
        if first_step_replacement_mode:
            axes[2].step(local_time_u, candidate_first_step_lyap_ok_flags[step_start:step_end], where="post", linewidth=2.0, color="tab:blue", label="candidate_first_step_ok")
            axes[2].step(local_time_u, first_step_contraction_triggered_flags[step_start:step_end], where="post", linewidth=2.0, color="tab:orange", label="replacement_triggered")
            axes[2].step(local_time_u, constrained_mpc_applied_flags[step_start:step_end], where="post", linewidth=2.0, color="tab:green", label="constrained_mpc_applied")
            axes[2].step(local_time_u, constrained_mpc_failed_applied_candidate_flags[step_start:step_end], where="post", linewidth=1.5, linestyle="--", color="tab:red", label="constrained_failed_candidate_applied")
            axes[2].set_ylabel("replacement")
            axes[3].step(local_time_u, first_step_contraction_satisfied_applied_flags[step_start:step_end], where="post", linewidth=2.0, color="tab:green", label="applied_first_step_ok")
            axes[3].step(local_time_u, constrained_mpc_solved_flags[step_start:step_end], where="post", linewidth=1.5, linestyle="--", color="tab:purple", label="constrained_mpc_solved")
            axes[3].set_ylabel("lyap")
        else:
            axes[2].step(local_time_u, qcqp_attempted_flags[step_start:step_end], where="post", linewidth=2.0, color="tab:orange", label="qcqp_attempted")
            axes[2].step(local_time_u, qcqp_hard_accepted_flags[step_start:step_end], where="post", linewidth=2.0, color="tab:green", label="qcqp_hard_accepted")
            axes[2].step(local_time_u, projection_active[step_start:step_end], where="post", linewidth=1.5, linestyle="--", color="tab:red", label="optimized_correction")
            axes[2].set_ylabel("projection")
            axes[3].step(local_time_u, fallback_mpc_active[step_start:step_end], where="post", linewidth=2.0, color="tab:purple", label="fallback_mpc_active")
            axes[3].step(local_time_u, fallback_verified_flags[step_start:step_end], where="post", linewidth=1.5, linestyle="--", color="tab:green", label="fallback_mpc_verified")
            axes[3].set_ylabel("fallback")
        axes[2].set_ylim(-0.05, 1.05)
        axes[2].set_yticks([0.0, 1.0])
        axes[2].grid(True, linestyle="--", alpha=0.35)
        axes[2].legend(loc="best")
        axes[3].set_ylim(-0.05, 1.05)
        axes[3].set_yticks([0.0, 1.0])
        axes[3].grid(True, linestyle="--", alpha=0.35)
        axes[3].legend(loc="best")
        axes[3].set_xlabel("step in episode")
        fig.suptitle(title, fontsize=12)
        plt.tight_layout()
        plt.savefig(filepath, dpi=300, bbox_inches="tight")
        plt.close(fig)

    episode_plot_dir = os.path.join(output_dir, "episode_samples_by_tens")
    os.makedirs(episode_plot_dir, exist_ok=True)
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
                f"episode_{chosen_episode + 1:03d}_from_{block_start + 1:03d}_{block_end:03d}.png",
            ),
        )

    last_episode_dir = os.path.join(output_dir, "last_episode_summary")
    os.makedirs(last_episode_dir, exist_ok=True)
    last_episode = max(n_episodes - 1, 0)
    step_start = last_episode * episode_len
    step_end = min((last_episode + 1) * episode_len, len(time_u))
    _plot_episode_window(
        step_start,
        step_end,
        f"Last episode ({last_episode + 1})",
        os.path.join(last_episode_dir, f"episode_{last_episode + 1:03d}_last.png"),
    )


def save_safety_filter_debug_artifacts(
    bundle,
    directory=None,
    prefix_name="safety_filter_debug",
    save_plots=True,
    save_paper_plots=True,
    save_rl_summary_plots=True,
    paper_plot_subdir="paper_plots",
):
    if directory is None:
        directory = os.getcwd()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(directory, prefix_name, timestamp)
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(out_dir, "bundle.pkl"), "wb") as f:
        pickle.dump(bundle, f)

    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(_jsonable(bundle["summary"]), f, indent=2)

    summary_csv_records = [
        {"key": key, "value": json.dumps(_jsonable(value))}
        for key, value in bundle["summary"].items()
    ]
    _write_csv(os.path.join(out_dir, "summary.csv"), summary_csv_records)

    step_records = make_safety_filter_step_records(bundle["lyap_info_storage"])
    _write_csv(os.path.join(out_dir, "step_table.csv"), step_records)

    _save_npz(os.path.join(out_dir, "arrays.npz"), bundle)

    if HAS_PANDAS:
        df = pd.DataFrame(step_records)
        df.to_pickle(os.path.join(out_dir, "step_table.pkl"))

    if save_plots:
        if save_paper_plots:
            plot_safety_filter_diagnostic_only(bundle, out_dir)
        else:
            plot_safety_filter_bundle(bundle, out_dir)
        if save_paper_plots:
            paper_root = os.path.join(out_dir, str(paper_plot_subdir))
            plot_safety_filter_bundle(bundle, os.path.join(paper_root, "safety_selector"), paper_style=True)
            if (
                str(bundle.get("source", "")).lower().startswith("rl")
                and save_rl_summary_plots
                and HAS_RL_SUMMARY_PLOTS
            ):
                save_rl_summary_plots_from_bundle(
                    bundle=bundle,
                    output_dir=os.path.join(paper_root, "rl_summary"),
                    paper_style=True,
                    save_input_data=False,
                )

    return out_dir


def save_lyap_debug_artifacts(
    bundle,
    directory=None,
    prefix_name="safety_filter_debug",
    save_plots=True,
    save_paper_plots=True,
    save_rl_summary_plots=True,
    paper_plot_subdir="paper_plots",
):
    return save_safety_filter_debug_artifacts(
        bundle=bundle,
        directory=directory,
        prefix_name=prefix_name,
        save_plots=save_plots,
        save_paper_plots=save_paper_plots,
        save_rl_summary_plots=save_rl_summary_plots,
        paper_plot_subdir=paper_plot_subdir,
    )


def load_safety_filter_debug_bundle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def load_lyap_debug_artifacts(path):
    return load_safety_filter_debug_bundle(path)
