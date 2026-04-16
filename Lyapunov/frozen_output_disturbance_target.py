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


DEFAULT_FROZEN_OUTPUT_DISTURBANCE_TARGET_CONFIG: Dict[str, Any] = {
    "solver_mode": DEFAULT_ANALYSIS_CONFIG["solver_mode"],
    "cond_warn_threshold": DEFAULT_ANALYSIS_CONFIG["cond_warn_threshold"],
    "residual_warn_threshold": DEFAULT_ANALYSIS_CONFIG["residual_warn_threshold"],
    "rank_tol": DEFAULT_ANALYSIS_CONFIG["rank_tol"],
    "box_bound_tol": DEFAULT_ANALYSIS_CONFIG["box_bound_tol"],
    "box_use_reduced_first": DEFAULT_ANALYSIS_CONFIG["box_use_reduced_first"],
    "zero_block_tol": 1.0e-9,
    "integrator_tol": 1.0e-9,
    "cd_identity_tol": 1.0e-9,
}


def _as_float_array(value: Any, name: str, ndim: Optional[int] = None) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if ndim is not None and array.ndim != ndim:
        raise ValueError(f"{name} must have ndim={ndim}, got shape {array.shape}.")
    return array


def _merge_config(config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    merged = deepcopy(DEFAULT_FROZEN_OUTPUT_DISTURBANCE_TARGET_CONFIG)
    if config:
        merged.update(dict(config))
    return merged


def _recover_output_disturbance_model(
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
        raise ValueError("B_aug row dimension must match A_aug.")
    if C_aug.shape[1] != A_aug.shape[0]:
        raise ValueError("C_aug column dimension must match A_aug.")

    n_aug = int(A_aug.shape[0])
    n_y = int(C_aug.shape[0])
    n_x = n_aug - n_y
    if n_x <= 0:
        raise ValueError("Invalid augmentation: physical-state dimension must be positive.")

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

    if np.max(np.abs(A_xd)) > zero_tol:
        raise ValueError(
            "frozen_output_disturbance_target expects no disturbance term in the state dynamics."
        )
    if np.max(np.abs(A_dx)) > zero_tol:
        raise ValueError(
            "frozen_output_disturbance_target expects zero lower-left augmentation block."
        )
    if np.max(np.abs(B_d)) > zero_tol:
        raise ValueError(
            "frozen_output_disturbance_target expects zero input action on disturbance rows."
        )
    if not np.allclose(A_dd, np.eye(n_y, dtype=float), atol=integrator_tol, rtol=0.0):
        raise ValueError(
            "frozen_output_disturbance_target expects disturbance integrator dynamics d_{k+1} = d_k."
        )
    if not np.allclose(C_d, np.eye(n_y, dtype=float), atol=cd_identity_tol, rtol=0.0):
        raise ValueError(
            "frozen_output_disturbance_target currently supports only C_d = I augmentation."
        )

    return {
        "A": A,
        "B": B,
        "C": C,
        "n_x": n_x,
        "n_y": n_y,
        "n_u": int(B.shape[1]),
    }


def _validate_target_inputs(
    *,
    model: Dict[str, Any],
    xhat_aug: np.ndarray,
    y_sp: np.ndarray,
    H: Optional[np.ndarray],
) -> Dict[str, Any]:
    n_x = int(model["n_x"])
    n_y = int(model["n_y"])
    n_u = int(model["n_u"])

    xhat_aug = _as_float_array(xhat_aug, "xhat_aug", ndim=1)
    if xhat_aug.size != (n_x + n_y):
        raise ValueError(f"xhat_aug has incorrect size. Expected {n_x + n_y}, got {xhat_aug.size}.")

    y_sp = _as_float_array(y_sp, "y_sp", ndim=1)
    if y_sp.size != n_y:
        raise ValueError(f"y_sp has incorrect size. Expected {n_y}, got {y_sp.size}.")

    if H is not None:
        H_arr = _as_float_array(H, "H", ndim=2)
        if H_arr.shape != (n_y, n_y) or not np.allclose(H_arr, np.eye(n_y), atol=1.0e-12, rtol=0.0):
            raise ValueError(
                "frozen_output_disturbance_target currently supports only full-output targets with H = None or H = I."
            )

    return {
        "xhat_k": np.asarray(xhat_aug[:n_x], dtype=float).reshape(-1),
        "d_hat_k": np.asarray(xhat_aug[n_x:], dtype=float).reshape(-1),
        "y_sp": y_sp,
        "n_u": n_u,
    }


def _exact_bounds_info(
    u_s: np.ndarray,
    u_min: Optional[np.ndarray],
    u_max: Optional[np.ndarray],
    box_bound_tol: float,
) -> Dict[str, Any]:
    if u_min is None or u_max is None:
        return {
            "within_bounds": None,
            "lower_violation": None,
            "upper_violation": None,
            "lower_violation_inf": None,
            "upper_violation_inf": None,
            "violation_inf": None,
            "active_lower_mask": None,
            "active_upper_mask": None,
        }
    return check_box_bounds(
        _as_float_array(u_s, "u_s", ndim=1),
        _as_float_array(u_min, "u_min", ndim=1),
        _as_float_array(u_max, "u_max", ndim=1),
        tol=float(box_bound_tol),
    )


def _base_result_dict(
    *,
    mode: str,
    target_variant: str,
    model: Dict[str, Any],
    exact_info: Dict[str, Any],
    d_hat_k: np.ndarray,
    bounds_info: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "success": True,
        "mode": str(mode),
        "target_variant": str(target_variant),
        "x_s": np.asarray(exact_info["x_s"], dtype=float).reshape(model["n_x"]),
        "u_s": np.asarray(exact_info["u_s"], dtype=float).reshape(model["n_u"]),
        "d_s": np.asarray(d_hat_k, dtype=float).reshape(model["n_y"]).copy(),
        "y_s": np.asarray(exact_info["y_s"], dtype=float).reshape(model["n_y"]),
        "x_s_aug": np.concatenate(
            [
                np.asarray(exact_info["x_s"], dtype=float).reshape(model["n_x"]),
                np.asarray(d_hat_k, dtype=float).reshape(model["n_y"]),
            ]
        ),
        "residual_dyn": np.asarray(exact_info["residual_dyn"], dtype=float).reshape(model["n_x"]),
        "residual_out": np.asarray(exact_info["residual_out"], dtype=float).reshape(model["n_y"]),
        "residual_total": np.asarray(exact_info["residual_total"], dtype=float),
        "residual_dyn_norm": float(exact_info["residual_dyn_norm"]),
        "residual_out_norm": float(exact_info["residual_out_norm"]),
        "residual_total_norm": float(exact_info["residual_total_norm"]),
        "rank_M": int(exact_info["rank_M"]),
        "rank_G": int(exact_info["rank_G"]),
        "rank_solver_matrix": None
        if exact_info["rank_solver_matrix"] is None
        else int(exact_info["rank_solver_matrix"]),
        "cond_M": float(exact_info["cond_M"]),
        "cond_G": float(exact_info["cond_G"]),
        "cond_I_minus_A": float(exact_info["cond_I_minus_A"]),
        "smallest_sv_M": float(exact_info["smallest_sv_M"]),
        "smallest_sv_G": float(exact_info["smallest_sv_G"]),
        "solver_mode_requested": str(exact_info["solver_mode_requested"]),
        "solver_mode_used": str(exact_info["solver_mode_used"]),
        "requested_mode_fallback": bool(exact_info["requested_mode_fallback"]),
        "fallback_reason": exact_info["fallback_reason"],
        "used_lstsq": bool(exact_info["used_lstsq"]),
        "is_exact_solution": bool(exact_info["is_exact_solution"]),
        "rhs_output": np.asarray(exact_info["rhs_output"], dtype=float).reshape(model["n_y"]),
        "invertible_I_minus_A": bool(exact_info["invertible_I_minus_A"]),
        "reduced_exact_available": bool(exact_info["reduced_exact_available"]),
        "reduced_lstsq_available": bool(exact_info["reduced_lstsq_available"]),
        "exact_within_bounds": bounds_info["within_bounds"],
        "exact_bound_violation_inf": bounds_info["violation_inf"],
        "exact_bound_violation_lower_inf": bounds_info["lower_violation_inf"],
        "exact_bound_violation_upper_inf": bounds_info["upper_violation_inf"],
        "exact_active_lower_mask": None
        if bounds_info["active_lower_mask"] is None
        else np.asarray(bounds_info["active_lower_mask"], dtype=bool),
        "exact_active_upper_mask": None
        if bounds_info["active_upper_mask"] is None
        else np.asarray(bounds_info["active_upper_mask"], dtype=bool),
    }


def solve_target_unbounded_output_disturbance(
    A_aug: np.ndarray,
    B_aug: np.ndarray,
    C_aug: np.ndarray,
    xhat_aug: np.ndarray,
    y_sp: np.ndarray,
    *,
    u_min: Optional[np.ndarray] = None,
    u_max: Optional[np.ndarray] = None,
    config: Optional[Dict[str, Any]] = None,
    H: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    cfg = _merge_config(config)
    model = _recover_output_disturbance_model(A_aug, B_aug, C_aug, cfg)
    validated = _validate_target_inputs(
        model=model,
        xhat_aug=xhat_aug,
        y_sp=y_sp,
        H=H,
    )

    exact_info = solve_legacy_ss_exact(
        model["A"],
        model["B"],
        model["C"],
        validated["y_sp"],
        validated["d_hat_k"],
        solver_mode=str(cfg["solver_mode"]),
        cond_warn_threshold=float(cfg["cond_warn_threshold"]),
        residual_warn_threshold=float(cfg["residual_warn_threshold"]),
        rank_tol=cfg["rank_tol"],
    )
    bounds_info = _exact_bounds_info(
        exact_info["u_s"],
        u_min=u_min,
        u_max=u_max,
        box_bound_tol=float(cfg["box_bound_tol"]),
    )
    result = _base_result_dict(
        mode="unbounded",
        target_variant="unbounded",
        model=model,
        exact_info=exact_info,
        d_hat_k=validated["d_hat_k"],
        bounds_info=bounds_info,
    )
    result.update(
        {
            "solve_stage": "frozen_output_disturbance_unbounded",
            "bounded_solution_used": False,
            "bounded_solver_name": None,
            "bounded_solve_form": None,
            "bounded_residual_norm": None,
            "bounded_state_residual_inf": None,
            "bounded_output_residual_inf": None,
            "bounded_active_lower_mask": None,
            "bounded_active_upper_mask": None,
        }
    )
    return result


def solve_target_bounded_output_disturbance(
    A_aug: np.ndarray,
    B_aug: np.ndarray,
    C_aug: np.ndarray,
    xhat_aug: np.ndarray,
    y_sp: np.ndarray,
    u_min: np.ndarray,
    u_max: np.ndarray,
    *,
    config: Optional[Dict[str, Any]] = None,
    H: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    cfg = _merge_config(config)
    model = _recover_output_disturbance_model(A_aug, B_aug, C_aug, cfg)
    validated = _validate_target_inputs(
        model=model,
        xhat_aug=xhat_aug,
        y_sp=y_sp,
        H=H,
    )

    u_min = _as_float_array(u_min, "u_min", ndim=1)
    u_max = _as_float_array(u_max, "u_max", ndim=1)
    if u_min.size != model["n_u"] or u_max.size != model["n_u"]:
        raise ValueError(f"u_min and u_max must have size {model['n_u']}.")

    exact_info = solve_legacy_ss_exact(
        model["A"],
        model["B"],
        model["C"],
        validated["y_sp"],
        validated["d_hat_k"],
        solver_mode=str(cfg["solver_mode"]),
        cond_warn_threshold=float(cfg["cond_warn_threshold"]),
        residual_warn_threshold=float(cfg["residual_warn_threshold"]),
        rank_tol=cfg["rank_tol"],
    )
    bounds_info = _exact_bounds_info(
        exact_info["u_s"],
        u_min=u_min,
        u_max=u_max,
        box_bound_tol=float(cfg["box_bound_tol"]),
    )

    exact_result = _base_result_dict(
        mode="bounded",
        target_variant="bounded",
        model=model,
        exact_info=exact_info,
        d_hat_k=validated["d_hat_k"],
        bounds_info=bounds_info,
    )
    exact_result.update(
        {
            "exact_x_s": np.asarray(exact_info["x_s"], dtype=float).reshape(model["n_x"]),
            "exact_u_s": np.asarray(exact_info["u_s"], dtype=float).reshape(model["n_u"]),
            "exact_y_s": np.asarray(exact_info["y_s"], dtype=float).reshape(model["n_y"]),
            "exact_d_s": np.asarray(validated["d_hat_k"], dtype=float).reshape(model["n_y"]),
        }
    )

    if bool(bounds_info["within_bounds"]):
        exact_result.update(
            {
                "solve_stage": "frozen_output_disturbance_exact_bounded",
                "bounded_solution_used": False,
                "bounded_solver_name": None,
                "bounded_solve_form": None,
                "bounded_residual_norm": 0.0,
                "bounded_state_residual_inf": float(exact_info["residual_dyn_norm"]),
                "bounded_output_residual_inf": float(exact_info["residual_out_norm"]),
                "bounded_active_lower_mask": np.asarray(bounds_info["active_lower_mask"], dtype=bool),
                "bounded_active_upper_mask": np.asarray(bounds_info["active_upper_mask"], dtype=bool),
            }
        )
        return exact_result

    bounded_info = solve_bounded_steady_state_least_squares(
        model["A"],
        model["B"],
        model["C"],
        validated["y_sp"],
        validated["d_hat_k"],
        u_min,
        u_max,
        cond_warn_threshold=float(cfg["cond_warn_threshold"]),
        rank_tol=cfg["rank_tol"],
        box_bound_tol=float(cfg["box_bound_tol"]),
        use_reduced_first=bool(cfg["box_use_reduced_first"]),
        u_ref=None,
        u_ref_weight=0.0,
    )

    if not bool(bounded_info["solve_success"]):
        exact_result.update(
            {
                "success": False,
                "solve_stage": "frozen_output_disturbance_bounded_failed",
                "bounded_solution_used": True,
                "bounded_solver_name": bounded_info["solver_name"],
                "bounded_solve_form": bounded_info["solve_form"],
                "bounded_status": bounded_info["status"],
                "bounded_message": bounded_info["message"],
                "bounded_residual_norm": bounded_info["residual_norm"],
                "bounded_state_residual_inf": bounded_info["state_residual_inf"],
                "bounded_output_residual_inf": bounded_info["output_residual_inf"],
                "bounded_active_lower_mask": np.asarray(bounded_info["active_lower_mask"], dtype=bool),
                "bounded_active_upper_mask": np.asarray(bounded_info["active_upper_mask"], dtype=bool),
            }
        )
        return exact_result

    exact_result.update(
        {
            "x_s": np.asarray(bounded_info["x_s"], dtype=float).reshape(model["n_x"]),
            "u_s": np.asarray(bounded_info["u_s"], dtype=float).reshape(model["n_u"]),
            "d_s": np.asarray(bounded_info["d_s"], dtype=float).reshape(model["n_y"]),
            "y_s": np.asarray(bounded_info["y_s"], dtype=float).reshape(model["n_y"]),
            "x_s_aug": np.concatenate(
                [
                    np.asarray(bounded_info["x_s"], dtype=float).reshape(model["n_x"]),
                    np.asarray(bounded_info["d_s"], dtype=float).reshape(model["n_y"]),
                ]
            ),
            "residual_dyn": np.asarray(bounded_info["residual_dyn"], dtype=float).reshape(model["n_x"]),
            "residual_out": np.asarray(bounded_info["residual_out"], dtype=float).reshape(model["n_y"]),
            "residual_total": np.asarray(bounded_info["residual_total"], dtype=float),
            "residual_dyn_norm": float(bounded_info["state_residual_inf"]),
            "residual_out_norm": float(bounded_info["output_residual_inf"]),
            "residual_total_norm": float(bounded_info["residual_norm"]),
            "solve_stage": "frozen_output_disturbance_bounded_ls",
            "bounded_solution_used": True,
            "bounded_solver_name": bounded_info["solver_name"],
            "bounded_solve_form": bounded_info["solve_form"],
            "bounded_status": bounded_info["status"],
            "bounded_message": bounded_info["message"],
            "bounded_residual_norm": float(bounded_info["residual_norm"]),
            "bounded_state_residual_inf": float(bounded_info["state_residual_inf"]),
            "bounded_output_residual_inf": float(bounded_info["output_residual_inf"]),
            "bounded_active_lower_mask": np.asarray(bounded_info["active_lower_mask"], dtype=bool),
            "bounded_active_upper_mask": np.asarray(bounded_info["active_upper_mask"], dtype=bool),
        }
    )
    return exact_result


def solve_output_disturbance_target(
    A_aug: np.ndarray,
    B_aug: np.ndarray,
    C_aug: np.ndarray,
    xhat_aug: np.ndarray,
    y_sp: np.ndarray,
    *,
    target_mode: str = "unbounded",
    u_min: Optional[np.ndarray] = None,
    u_max: Optional[np.ndarray] = None,
    config: Optional[Dict[str, Any]] = None,
    H: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    mode = str(target_mode).strip().lower()
    if mode == "unbounded":
        return solve_target_unbounded_output_disturbance(
            A_aug,
            B_aug,
            C_aug,
            xhat_aug,
            y_sp,
            u_min=u_min,
            u_max=u_max,
            config=config,
            H=H,
        )
    if mode == "bounded":
        if u_min is None or u_max is None:
            raise ValueError("bounded target mode requires u_min and u_max.")
        return solve_target_bounded_output_disturbance(
            A_aug,
            B_aug,
            C_aug,
            xhat_aug,
            y_sp,
            u_min,
            u_max,
            config=config,
            H=H,
        )
    raise ValueError("target_mode must be 'unbounded' or 'bounded'.")
