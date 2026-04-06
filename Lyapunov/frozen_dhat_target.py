from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Optional

import numpy as np

from analysis.steady_state_debug_analysis import (
    DEFAULT_ANALYSIS_CONFIG,
    check_box_bounds,
    solve_bounded_steady_state_least_squares,
    solve_legacy_ss_exact,
)


FROZEN_DHAT_SELECTOR_NAME = "bounded_frozen_dhat"

DEFAULT_FROZEN_DHAT_TARGET_CONFIG: Dict[str, Any] = {
    "solver_mode": DEFAULT_ANALYSIS_CONFIG["solver_mode"],
    "cond_warn_threshold": DEFAULT_ANALYSIS_CONFIG["cond_warn_threshold"],
    "residual_warn_threshold": DEFAULT_ANALYSIS_CONFIG["residual_warn_threshold"],
    "rank_tol": DEFAULT_ANALYSIS_CONFIG["rank_tol"],
    "box_bound_tol": DEFAULT_ANALYSIS_CONFIG["box_bound_tol"],
    "box_use_reduced_first": DEFAULT_ANALYSIS_CONFIG["box_use_reduced_first"],
    "u_ref_weight": DEFAULT_ANALYSIS_CONFIG["u_ref_weight"],
    "integrator_tol": 1.0e-9,
    "cd_identity_tol": 1.0e-9,
    "zero_block_tol": 1.0e-9,
}


def _as_float_array(value: Any, name: str, ndim: Optional[int] = None) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if ndim is not None and array.ndim != ndim:
        raise ValueError(f"{name} must have ndim={ndim}, got shape {array.shape}.")
    return array


def _merge_config(config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    merged = deepcopy(DEFAULT_FROZEN_DHAT_TARGET_CONFIG)
    if config:
        merged.update(dict(config))
    return merged


def _norm_sq(value: np.ndarray) -> float:
    flat = np.asarray(value, dtype=float).reshape(-1)
    return float(flat @ flat)


def _inf_norm(value: Any) -> float:
    if value is None:
        return float("nan")
    flat = np.asarray(value, dtype=float).reshape(-1)
    if flat.size == 0:
        return 0.0
    return float(np.max(np.abs(flat)))


def _extract_prev_target(prev_target: Optional[Dict[str, Any]]) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    if not isinstance(prev_target, dict):
        return None, None
    x_prev = prev_target.get("x_s")
    u_prev = prev_target.get("u_s")
    x_prev_arr = None if x_prev is None else np.asarray(x_prev, dtype=float).reshape(-1).copy()
    u_prev_arr = None if u_prev is None else np.asarray(u_prev, dtype=float).reshape(-1).copy()
    return x_prev_arr, u_prev_arr


def _recover_unaugmented_model(
    A_aug: np.ndarray,
    B_aug: np.ndarray,
    C_aug: np.ndarray,
    cfg: Dict[str, Any],
) -> Dict[str, Any]:
    A_aug = _as_float_array(A_aug, "A_aug", ndim=2)
    B_aug = _as_float_array(B_aug, "B_aug", ndim=2)
    C_aug = _as_float_array(C_aug, "C_aug", ndim=2)

    if A_aug.shape[0] != A_aug.shape[1]:
        raise ValueError("A_aug must be square.")
    if B_aug.shape[0] != A_aug.shape[0]:
        raise ValueError("B_aug has incompatible shape.")
    if C_aug.shape[1] != A_aug.shape[0]:
        raise ValueError("C_aug has incompatible shape.")

    n_aug = int(A_aug.shape[0])
    n_y = int(C_aug.shape[0])
    n_x = n_aug - n_y
    if n_x <= 0:
        raise ValueError("Invalid augmentation: inferred physical-state dimension must be positive.")

    A = np.asarray(A_aug[:n_x, :n_x], dtype=float)
    B = np.asarray(B_aug[:n_x, :], dtype=float)
    C = np.asarray(C_aug[:, :n_x], dtype=float)

    A_xd = np.asarray(A_aug[:n_x, n_x:], dtype=float)
    A_dx = np.asarray(A_aug[n_x:, :n_x], dtype=float)
    A_dd = np.asarray(A_aug[n_x:, n_x:], dtype=float)
    B_d = np.asarray(B_aug[n_x:, :], dtype=float)
    C_d = np.asarray(C_aug[:, n_x:], dtype=float)

    zero_tol = float(cfg["zero_block_tol"])
    integrator_tol = float(cfg["integrator_tol"])
    cd_identity_tol = float(cfg["cd_identity_tol"])

    if np.max(np.abs(A_dx)) > zero_tol:
        raise ValueError("bounded_frozen_dhat expects the augmentation lower-left block to be zero.")
    if np.max(np.abs(B_d)) > zero_tol:
        raise ValueError("bounded_frozen_dhat expects zero input action on the disturbance integrator rows.")
    if not np.allclose(A_dd, np.eye(n_y, dtype=float), atol=integrator_tol, rtol=0.0):
        raise ValueError("bounded_frozen_dhat expects disturbance integrator dynamics d_{k+1} = d_k.")
    if not np.allclose(C_d, np.eye(n_y, dtype=float), atol=cd_identity_tol, rtol=0.0):
        raise ValueError("bounded_frozen_dhat currently supports only C_d = I output-offset augmentation.")

    return {
        "A": A,
        "B": B,
        "C": C,
        "Bd_aug": A_xd,
        "n_x": n_x,
        "n_y": n_y,
        "n_u": int(B.shape[1]),
    }


def _failure_target_info(
    *,
    y_sp: np.ndarray,
    message: str,
    warm_start_enabled: bool,
    warm_start_available: bool,
) -> Dict[str, Any]:
    selector_debug = {
        "status": "failed",
        "solver": None,
        "solver_error": str(message),
        "objective_value": None,
        "objective_terms": {
            "target_tracking": np.nan,
            "u_applied_anchor": np.nan,
            "u_prev_smoothing": np.nan,
            "x_prev_smoothing": np.nan,
            "xhat_anchor": np.nan,
        },
        "warm_start_enabled": bool(warm_start_enabled),
        "warm_start_available": bool(warm_start_available),
        "warm_start_used": False,
        "prev_input_term_active": False,
        "prev_state_term_active": False,
        "box_solve_mode": "failed",
        "exact_within_bounds": None,
        "exact_bound_violation_inf": None,
        "bounded_residual_norm": None,
    }
    return {
        "success": False,
        "selector_mode": FROZEN_DHAT_SELECTOR_NAME,
        "selector_name": FROZEN_DHAT_SELECTOR_NAME,
        "solve_stage": "failed",
        "x_s": None,
        "u_s": None,
        "d_s": None,
        "x_s_aug": None,
        "y_s": None,
        "yc_s": None,
        "r_s": None,
        "requested_y_sp": np.asarray(y_sp, dtype=float).reshape(-1).copy(),
        "objective_value": None,
        "objective": None,
        "objective_terms": selector_debug["objective_terms"],
        "target_error": None,
        "target_error_inf": None,
        "target_error_norm": None,
        "target_slack": None,
        "target_slack_inf": None,
        "target_slack_2": None,
        "target_eq_residual_inf": None,
        "dyn_residual_inf": None,
        "bound_violation_inf": None,
        "input_bound_violation_inf": None,
        "output_bound_violation_inf": 0.0,
        "d_s_minus_dhat_inf": None,
        "d_s_frozen": True,
        "d_s_optimized": False,
        "warm_start": {
            "enabled": bool(warm_start_enabled),
            "available": bool(warm_start_available),
            "used": False,
        },
        "status": "failed",
        "solver": None,
        "selector_debug": selector_debug,
        "box_solve_mode": "failed",
        "exact_within_bounds": None,
        "exact_bound_violation_inf": None,
        "bounded_residual_norm": None,
    }


def prepare_filter_target_from_bounded_frozen_dhat(
    A_aug,
    B_aug,
    C_aug,
    xhat_aug,
    y_sp,
    u_min,
    u_max,
    *,
    prev_target=None,
    warm_start=True,
    u_applied_k=None,
    config=None,
    return_debug=False,
    H=None,
):
    cfg = _merge_config(config)
    x_prev, u_prev = _extract_prev_target(prev_target)
    warm_start_enabled = bool(warm_start)
    warm_start_available = bool(isinstance(prev_target, dict) and prev_target.get("success", False))

    model = _recover_unaugmented_model(A_aug, B_aug, C_aug, cfg)
    n_x = int(model["n_x"])
    n_y = int(model["n_y"])
    n_u = int(model["n_u"])

    xhat_aug = _as_float_array(xhat_aug, "xhat_aug", ndim=1)
    if xhat_aug.size != (n_x + n_y):
        raise ValueError(f"xhat_aug has incorrect size. Expected {n_x + n_y}, got {xhat_aug.size}.")
    xhat_k = xhat_aug[:n_x].copy()
    d_hat_k = xhat_aug[n_x:].copy()

    if H is not None:
        H_arr = _as_float_array(H, "H", ndim=2)
        if H_arr.shape != (n_y, n_y) or not np.allclose(H_arr, np.eye(n_y), atol=1.0e-12, rtol=0.0):
            raise ValueError(
                "bounded_frozen_dhat currently supports only full-output targets with H = None or H = I."
            )

    y_sp = _as_float_array(y_sp, "y_sp", ndim=1)
    if y_sp.size != n_y:
        raise ValueError(f"y_sp has incorrect size. Expected {n_y}, got {y_sp.size}.")

    u_min = _as_float_array(u_min, "u_min", ndim=1)
    u_max = _as_float_array(u_max, "u_max", ndim=1)
    if u_min.size != n_u or u_max.size != n_u:
        raise ValueError(f"u_min and u_max must have size {n_u}.")

    if u_applied_k is None:
        u_applied = np.zeros(n_u, dtype=float)
    else:
        u_applied = _as_float_array(u_applied_k, "u_applied_k", ndim=1)
        if u_applied.size != n_u:
            raise ValueError(f"u_applied_k must have size {n_u}.")

    exact_info = solve_legacy_ss_exact(
        model["A"],
        model["B"],
        model["C"],
        y_sp_k=y_sp,
        d_hat_k=d_hat_k,
        solver_mode=cfg["solver_mode"],
        cond_warn_threshold=float(cfg["cond_warn_threshold"]),
        residual_warn_threshold=float(cfg["residual_warn_threshold"]),
        rank_tol=cfg["rank_tol"],
    )

    exact_bounds = check_box_bounds(
        exact_info["u_s"],
        u_min,
        u_max,
        tol=float(cfg["box_bound_tol"]),
    )
    exact_success = bool(exact_info["is_exact_solution"]) and np.all(np.isfinite(exact_info["u_s"]))
    exact_within_bounds = bool(exact_success and exact_bounds["within_bounds"])
    exact_bound_violation_inf = float(exact_bounds["violation_inf"])

    bounded_info = None
    box_solve_mode = "exact_bounded"
    solve_stage = "frozen_dhat_exact"
    chosen = exact_info
    bounded_residual_norm = 0.0

    if not exact_within_bounds:
        box_solve_mode = "exact_unbounded_fallback_bounded_ls" if exact_success else "exact_unsolved_fallback_bounded_ls"
        bounded_info = solve_bounded_steady_state_least_squares(
            model["A"],
            model["B"],
            model["C"],
            y_sp_k=y_sp,
            d_hat_k=d_hat_k,
            u_min=u_min,
            u_max=u_max,
            cond_warn_threshold=float(cfg["cond_warn_threshold"]),
            rank_tol=cfg["rank_tol"],
            box_bound_tol=float(cfg["box_bound_tol"]),
            use_reduced_first=bool(cfg["box_use_reduced_first"]),
            u_ref=u_applied,
            u_ref_weight=cfg.get("u_ref_weight", 0.0),
        )
        if not bounded_info.get("solve_success", False):
            failure = _failure_target_info(
                y_sp=y_sp,
                message=bounded_info.get("message", "bounded frozen-dhat solve failed."),
                warm_start_enabled=warm_start_enabled,
                warm_start_available=warm_start_available,
            )
            failure["exact_within_bounds"] = bool(exact_within_bounds)
            failure["exact_bound_violation_inf"] = float(exact_bound_violation_inf)
            failure["bounded_residual_norm"] = bounded_info.get("residual_norm")
            failure["box_solve_mode"] = box_solve_mode
            failure["selector_debug"]["exact_within_bounds"] = bool(exact_within_bounds)
            failure["selector_debug"]["exact_bound_violation_inf"] = float(exact_bound_violation_inf)
            failure["selector_debug"]["bounded_residual_norm"] = bounded_info.get("residual_norm")
            failure["selector_debug"]["box_solve_mode"] = box_solve_mode
            failure["selector_debug"]["solver"] = bounded_info.get("solver_name")
            failure["selector_debug"]["status"] = bounded_info.get("status")
            failure["selector_debug"]["solver_error"] = bounded_info.get("message")
            if return_debug:
                return failure, dict(failure["selector_debug"])
            return failure
        chosen = bounded_info
        solve_stage = "frozen_dhat_bounded_fallback"
        bounded_residual_norm = float(bounded_info.get("residual_norm", np.nan))

    x_s = np.asarray(chosen["x_s"], dtype=float).reshape(n_x)
    u_s = np.asarray(chosen["u_s"], dtype=float).reshape(n_u)
    d_s = np.asarray(chosen["d_s"], dtype=float).reshape(n_y)
    y_s = np.asarray(chosen["y_s"], dtype=float).reshape(n_y)
    r_s = y_s.copy()

    dyn_residual = (np.eye(n_x, dtype=float) - model["A"]) @ x_s - model["B"] @ u_s
    target_error = r_s - y_sp
    target_error_inf = _inf_norm(target_error)
    target_error_norm = float(np.linalg.norm(target_error))
    bounds_info = check_box_bounds(u_s, u_min, u_max, tol=float(cfg["box_bound_tol"]))

    objective_terms = {
        "target_tracking": 0.5 * _norm_sq(target_error),
        "u_applied_anchor": 0.0,
        "u_prev_smoothing": np.nan,
        "x_prev_smoothing": np.nan,
        "xhat_anchor": np.nan,
    }
    u_ref_weight = np.asarray(cfg.get("u_ref_weight", 0.0), dtype=float).reshape(-1)
    if u_ref_weight.size == 0:
        u_ref_weight = np.zeros(n_u, dtype=float)
    elif u_ref_weight.size == 1:
        u_ref_weight = np.full(n_u, float(u_ref_weight.item()), dtype=float)
    elif u_ref_weight.size != n_u:
        raise ValueError("u_ref_weight must be scalar or match the number of inputs.")
    u_ref_weight = np.maximum(u_ref_weight, 0.0)
    objective_terms["u_applied_anchor"] = float(
        chosen.get("u_ref_penalty", np.sum(u_ref_weight * np.square(u_s - u_applied)))
    )
    objective_value = float(chosen.get("cost", objective_terms["target_tracking"]))
    solve_status = exact_info["solver_mode_used"] if solve_stage == "frozen_dhat_exact" else chosen.get("status", "bounded_success")

    selector_debug = {
        "status": solve_status,
        "solver": exact_info["solver_mode_used"] if solve_stage == "frozen_dhat_exact" else chosen.get("solver_name"),
        "solver_error": None if solve_stage == "frozen_dhat_exact" else chosen.get("message"),
        "objective_value": objective_value,
        "objective_terms": objective_terms,
        "warm_start_enabled": warm_start_enabled,
        "warm_start_available": warm_start_available,
        "warm_start_used": False,
        "prev_input_term_active": bool(np.any(u_ref_weight > 0.0)),
        "prev_state_term_active": False,
        "use_output_bounds_in_selector": False,
        "x_weight_base": "frozen_dhat_projection",
        "Qr_diag_used": None,
        "R_u_ref_diag_used": u_ref_weight.copy(),
        "R_delta_u_sel_diag_used": None,
        "Q_delta_x_diag_used": None,
        "Q_x_ref_diag_used": None,
        "Qx_base_diag_used": None,
        "Rdu_diag_used": u_ref_weight.copy(),
        "exact_solver_mode_requested": exact_info["solver_mode_requested"],
        "exact_solver_mode_used": exact_info["solver_mode_used"],
        "exact_residual_total_norm": exact_info["residual_total_norm"],
        "exact_residual_dyn_norm": exact_info["residual_dyn_norm"],
        "exact_residual_out_norm": exact_info["residual_out_norm"],
        "exact_within_bounds": exact_within_bounds,
        "exact_bound_violation_inf": exact_bound_violation_inf,
        "box_solve_mode": box_solve_mode,
        "bounded_residual_norm": bounded_residual_norm,
        "bounded_solver_name": None if bounded_info is None else bounded_info.get("solver_name"),
        "bounded_status": None if bounded_info is None else bounded_info.get("status"),
        "bounded_message": None if bounded_info is None else bounded_info.get("message"),
        "bounded_solve_form": None if bounded_info is None else bounded_info.get("solve_form"),
        "active_lower_mask": bounds_info["active_lower_mask"].copy(),
        "active_upper_mask": bounds_info["active_upper_mask"].copy(),
    }

    target_info = {
        "success": True,
        "selector_mode": FROZEN_DHAT_SELECTOR_NAME,
        "selector_name": FROZEN_DHAT_SELECTOR_NAME,
        "solve_stage": solve_stage,
        "x_s": x_s.copy(),
        "u_s": u_s.copy(),
        "d_s": d_s.copy(),
        "x_s_aug": np.concatenate([x_s, d_s]).copy(),
        "y_s": y_s.copy(),
        "yc_s": r_s.copy(),
        "r_s": r_s.copy(),
        "requested_y_sp": y_sp.copy(),
        "objective_value": objective_value,
        "objective": objective_value,
        "objective_terms": objective_terms,
        "target_error": target_error.copy(),
        "target_error_inf": target_error_inf,
        "target_error_norm": target_error_norm,
        "target_slack": target_error.copy(),
        "target_slack_inf": target_error_inf,
        "target_slack_2": target_error_norm,
        "target_eq_residual_inf": target_error_inf,
        "dyn_residual_inf": _inf_norm(dyn_residual),
        "bound_violation_inf": float(bounds_info["violation_inf"]),
        "input_bound_violation_inf": float(bounds_info["violation_inf"]),
        "output_bound_violation_inf": 0.0,
        "d_s_minus_dhat_inf": 0.0,
        "d_s_frozen": True,
        "d_s_optimized": False,
        "warm_start": {
            "enabled": warm_start_enabled,
            "available": warm_start_available,
            "used": False,
        },
        "status": solve_status,
        "solver": selector_debug["solver"],
        "selector_debug": selector_debug,
        "margin_to_u_min": (u_s - u_min).copy(),
        "margin_to_u_max": (u_max - u_s).copy(),
        "y_s_minus_y_sp": (y_s - y_sp).copy(),
        "r_s_minus_y_sp": target_error.copy(),
        "u_s_minus_u_applied": (u_s - u_applied).copy(),
        "u_s_minus_u_prev": None if u_prev is None else (u_s - u_prev).copy(),
        "x_s_minus_x_prev": None if x_prev is None else (x_s - x_prev).copy(),
        "x_s_minus_xhat": (x_s - xhat_k).copy(),
        "box_solve_mode": box_solve_mode,
        "exact_within_bounds": exact_within_bounds,
        "exact_bound_violation_inf": exact_bound_violation_inf,
        "bounded_residual_norm": bounded_residual_norm,
    }

    if return_debug:
        return target_info, dict(selector_debug)
    return target_info
