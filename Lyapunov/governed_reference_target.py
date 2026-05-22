from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np

try:
    import cvxpy as cp

    HAS_CVXPY = True
except Exception:
    cp = None
    HAS_CVXPY = False

from utils.lyapunov_utils import DEFAULT_CVXPY_SOLVERS


_OPTIMAL_STATUSES = {"optimal", "optimal_inaccurate"}

DEFAULT_GOVERNED_REFERENCE_CONFIG: Dict[str, Any] = {
    "governed_reference_enabled": True,
    "lambda_cmd_move": 1.0,
    "Qr_diag": None,
    "W_r_diag": None,
    "u_ref_weight": 0.1,
    "x_ref_weight": 0.1,
    "input_headroom_frac": 0.03,
    "one_step_probe": True,
    "governor_active_tol": 1.0e-8,
    "active_bound_tol": 1.0e-8,
    "solver_pref": DEFAULT_CVXPY_SOLVERS,
}


def _as_float_array(value: Any, name: str, ndim: Optional[int] = None) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if ndim is not None and arr.ndim != ndim:
        raise ValueError(f"{name} must have ndim={ndim}, got shape {arr.shape}.")
    return arr


def _merge_config(config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    merged = deepcopy(DEFAULT_GOVERNED_REFERENCE_CONFIG)
    if not config:
        return merged

    raw = dict(config)
    nested = raw.pop("governed_reference", None)
    if isinstance(nested, dict):
        merged.update(nested)

    for key in DEFAULT_GOVERNED_REFERENCE_CONFIG:
        if key in raw:
            merged[key] = raw[key]
    return merged


def _solver_sequence(solver_pref: Any) -> Tuple[Any, ...]:
    if solver_pref is None:
        return tuple(DEFAULT_CVXPY_SOLVERS)
    if isinstance(solver_pref, str):
        return (solver_pref,)
    return tuple(solver_pref)


def _solve_problem(problem: Any, variables: Iterable[Any], solver_pref: Any, *, warm_start: bool = True) -> Dict[str, Any]:
    last_status = None
    last_solver = None
    last_error = None
    last_value = None

    for solver_name in _solver_sequence(solver_pref):
        try:
            for var in variables:
                var.value = None
            problem.solve(solver=solver_name, warm_start=bool(warm_start), verbose=False)
            last_status = problem.status
            last_solver = solver_name
            last_value = problem.value
            if problem.status in _OPTIMAL_STATUSES and all(var.value is not None for var in variables):
                return {
                    "success": True,
                    "status": problem.status,
                    "solver": solver_name,
                    "error": None,
                    "objective_value": None if problem.value is None else float(problem.value),
                }
        except Exception as exc:
            last_error = repr(exc)

    return {
        "success": False,
        "status": last_status,
        "solver": last_solver,
        "error": last_error,
        "objective_value": None if last_value is None else float(last_value),
    }


def _recover_model(A_aug: Any, B_aug: Any, C_aug: Any, xhat_aug: Any) -> Dict[str, Any]:
    A_aug = _as_float_array(A_aug, "A_aug", ndim=2)
    B_aug = _as_float_array(B_aug, "B_aug", ndim=2)
    C_aug = _as_float_array(C_aug, "C_aug", ndim=2)
    xhat_aug = _as_float_array(xhat_aug, "xhat_aug", ndim=1)

    if A_aug.shape[0] != A_aug.shape[1]:
        raise ValueError("A_aug must be square.")
    if B_aug.shape[0] != A_aug.shape[0]:
        raise ValueError("B_aug row dimension must match A_aug.")
    if C_aug.shape[1] != A_aug.shape[0]:
        raise ValueError("C_aug column dimension must match A_aug.")
    if xhat_aug.size != A_aug.shape[0]:
        raise ValueError("xhat_aug has incorrect size.")

    n_aug = int(A_aug.shape[0])
    n_y = int(C_aug.shape[0])
    n_x = n_aug - n_y
    if n_x <= 0:
        raise ValueError("Expected augmented state ordered as [x; d].")

    return {
        "A_aug": A_aug,
        "B_aug": B_aug,
        "C_aug": C_aug,
        "xhat_aug": xhat_aug,
        "A": np.asarray(A_aug[:n_x, :n_x], dtype=float),
        "Bd": np.asarray(A_aug[:n_x, n_x:], dtype=float),
        "B": np.asarray(B_aug[:n_x, :], dtype=float),
        "C": np.asarray(C_aug[:, :n_x], dtype=float),
        "Cd": np.asarray(C_aug[:, n_x:], dtype=float),
        "xhat": np.asarray(xhat_aug[:n_x], dtype=float).reshape(n_x),
        "d_hat": np.asarray(xhat_aug[n_x:], dtype=float).reshape(n_y),
        "n_x": n_x,
        "n_y": n_y,
        "n_u": int(B_aug.shape[1]),
    }


def _validate_full_output_target(H: Optional[np.ndarray], n_y: int) -> Optional[np.ndarray]:
    if H is None:
        return None
    H_arr = _as_float_array(H, "H", ndim=2)
    if H_arr.shape != (n_y, n_y) or not np.allclose(H_arr, np.eye(n_y), atol=1.0e-12, rtol=0.0):
        raise ValueError("governed_reference currently supports only full-output targets with H=None or H=I.")
    return H_arr


def _diag_matrix(value: Any, size: int, *, default: float = 1.0) -> np.ndarray:
    if value is None:
        vals = np.full(size, float(default), dtype=float)
    else:
        vals = np.asarray(value, dtype=float).reshape(-1)
        if vals.size == 0:
            vals = np.full(size, float(default), dtype=float)
        elif vals.size == 1:
            vals = np.full(size, float(vals.item()), dtype=float)
        elif vals.size != size:
            raise ValueError(f"Expected scalar or vector of length {size}, got {vals.size}.")
    return np.diag(np.maximum(vals, 0.0))


def _weight_vector(value: Any, size: int, *, default: float = 0.0) -> np.ndarray:
    if value is None:
        return np.full(size, float(default), dtype=float)
    vals = np.asarray(value, dtype=float).reshape(-1)
    if vals.size == 0:
        return np.full(size, float(default), dtype=float)
    if vals.size == 1:
        return np.full(size, float(vals.item()), dtype=float)
    if vals.size != size:
        raise ValueError(f"Expected scalar or vector of length {size}, got {vals.size}.")
    return vals.copy()


def _headroom_bounds(u_min: Any, u_max: Any, cfg: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    u_min = _as_float_array(u_min, "u_min", ndim=1)
    u_max = _as_float_array(u_max, "u_max", ndim=1)
    if u_min.size != u_max.size:
        raise ValueError("u_min and u_max must have the same length.")
    width = u_max - u_min
    if np.any(width <= 0.0):
        raise ValueError("Input bounds must satisfy u_min < u_max.")
    frac = max(float(cfg.get("input_headroom_frac", 0.0)), 0.0)
    margin = frac * width
    u_lo = u_min + margin
    u_hi = u_max - margin
    if np.any(u_lo > u_hi):
        raise ValueError("input_headroom_frac makes the input bounds infeasible.")
    return u_lo, u_hi, margin


def _inf_norm(value: Any) -> float:
    arr = np.asarray(value, dtype=float).reshape(-1)
    if arr.size == 0:
        return 0.0
    return float(np.max(np.abs(arr)))


def _objective_quad(value: np.ndarray, weight: np.ndarray) -> float:
    arr = np.asarray(value, dtype=float).reshape(-1)
    return float(arr.T @ np.asarray(weight, dtype=float) @ arr)


def _lyapunov_probe(
    *,
    model: Dict[str, Any],
    x_s: np.ndarray,
    u_s: np.ndarray,
    u_min: np.ndarray,
    u_max: np.ndarray,
    P_x: Optional[np.ndarray],
    rho_lyap: Optional[float],
    eps_lyap: Optional[float],
    solver_pref: Any,
) -> Dict[str, Any]:
    if not HAS_CVXPY or P_x is None or rho_lyap is None or eps_lyap is None:
        return {
            "governor_probe_available": False,
            "governor_probe_success": None,
            "governor_probe_margin": None,
            "governor_probe_min_value": None,
            "governor_probe_bound": None,
            "governor_probe_status": None,
        }

    P_x = np.asarray(P_x, dtype=float)
    x_s = np.asarray(x_s, dtype=float).reshape(model["n_x"])
    u_s = np.asarray(u_s, dtype=float).reshape(model["n_u"])
    e_k = model["xhat"] - x_s
    V_k = float(e_k.T @ P_x @ e_k)
    V_bound = float(rho_lyap) * V_k + float(eps_lyap)

    u_var = cp.Variable(model["n_u"])
    e_next = model["A"] @ e_k + model["B"] @ (u_var - u_s)
    problem = cp.Problem(
        cp.Minimize(cp.quad_form(e_next, cp.psd_wrap(P_x))),
        [u_var >= u_min, u_var <= u_max],
    )
    solve_info = _solve_problem(problem, [u_var], solver_pref, warm_start=True)
    if not solve_info["success"]:
        return {
            "governor_probe_available": True,
            "governor_probe_success": False,
            "governor_probe_margin": None,
            "governor_probe_min_value": None,
            "governor_probe_bound": V_bound,
            "governor_probe_status": solve_info.get("status"),
        }

    u_probe = np.asarray(u_var.value, dtype=float).reshape(model["n_u"])
    e_next_value = model["A"] @ e_k + model["B"] @ (u_probe - u_s)
    V_next_min = float(e_next_value.T @ P_x @ e_next_value)
    margin = V_next_min - V_bound
    return {
        "governor_probe_available": True,
        "governor_probe_success": bool(margin <= 0.0),
        "governor_probe_margin": float(margin),
        "governor_probe_min_value": float(V_next_min),
        "governor_probe_bound": float(V_bound),
        "governor_probe_status": solve_info.get("status"),
        "governor_probe_u": u_probe.copy(),
    }


def solve_governed_command(
    A_aug: Any,
    B_aug: Any,
    C_aug: Any,
    xhat_aug: Any,
    y_sp: Any,
    *,
    u_min: Any,
    u_max: Any,
    config: Optional[Dict[str, Any]] = None,
    H: Optional[np.ndarray] = None,
    r_prev: Optional[np.ndarray] = None,
    P_x: Optional[np.ndarray] = None,
    rho_lyap: Optional[float] = None,
    eps_lyap: Optional[float] = None,
) -> Dict[str, Any]:
    cfg = _merge_config(config)
    if not HAS_CVXPY:
        return {
            "success": False,
            "status": "cvxpy_missing",
            "message": "CVXPY is required for the governed command solve.",
        }

    try:
        model = _recover_model(A_aug, B_aug, C_aug, xhat_aug)
        _validate_full_output_target(H, model["n_y"])
        y_sp = _as_float_array(y_sp, "y_sp", ndim=1)
        if y_sp.size != model["n_y"]:
            raise ValueError(f"y_sp must have length {model['n_y']}.")
        u_min_arr = _as_float_array(u_min, "u_min", ndim=1)
        u_max_arr = _as_float_array(u_max, "u_max", ndim=1)
        u_lo, u_hi, headroom = _headroom_bounds(u_min_arr, u_max_arr, cfg)

        W_r = _diag_matrix(cfg.get("W_r_diag", cfg.get("Qr_diag")), model["n_y"], default=1.0)
        lambda_cmd = max(float(cfg.get("lambda_cmd_move", 0.0)), 0.0)
        r_prev_arr = None if r_prev is None else _as_float_array(r_prev, "r_prev", ndim=1)
        if r_prev_arr is not None and r_prev_arr.size != model["n_y"]:
            raise ValueError(f"r_prev must have length {model['n_y']}.")

        x_var = cp.Variable(model["n_x"])
        u_var = cp.Variable(model["n_u"])
        y_expr = model["C"] @ x_var + model["Cd"] @ model["d_hat"]
        objective = cp.quad_form(y_expr - y_sp, cp.psd_wrap(W_r))
        if r_prev_arr is not None and lambda_cmd > 0.0:
            objective += lambda_cmd * cp.sum_squares(y_expr - r_prev_arr)
        constraints = [
            x_var == model["A"] @ x_var + model["Bd"] @ model["d_hat"] + model["B"] @ u_var,
            u_var >= u_lo,
            u_var <= u_hi,
        ]
        problem = cp.Problem(cp.Minimize(objective), constraints)
        solve_info = _solve_problem(problem, [x_var, u_var], cfg.get("solver_pref"), warm_start=True)
        if not solve_info["success"]:
            return {
                "success": False,
                "status": solve_info.get("status"),
                "solver": solve_info.get("solver"),
                "message": solve_info.get("error") or "governed command solve failed",
                "objective_value": solve_info.get("objective_value"),
            }

        x_cmd = np.asarray(x_var.value, dtype=float).reshape(model["n_x"])
        u_cmd = np.asarray(u_var.value, dtype=float).reshape(model["n_u"])
        r_cmd = np.asarray(model["C"] @ x_cmd + model["Cd"] @ model["d_hat"], dtype=float).reshape(model["n_y"])
        r_cmd_minus_y_sp = r_cmd - y_sp
        command_move = None if r_prev_arr is None else r_cmd - r_prev_arr
        input_headroom_min = float(np.min(np.minimum(u_cmd - u_min_arr, u_max_arr - u_cmd)))
        active_tol = float(cfg.get("active_bound_tol", 1.0e-8))
        probe = {}
        if bool(cfg.get("one_step_probe", True)):
            probe = _lyapunov_probe(
                model=model,
                x_s=x_cmd,
                u_s=u_cmd,
                u_min=u_min_arr,
                u_max=u_max_arr,
                P_x=P_x,
                rho_lyap=rho_lyap,
                eps_lyap=eps_lyap,
                solver_pref=cfg.get("solver_pref"),
            )

        return {
            "success": True,
            "status": solve_info.get("status"),
            "solver": solve_info.get("solver"),
            "message": "ok",
            "objective_value": solve_info.get("objective_value"),
            "x_cmd": x_cmd.copy(),
            "u_cmd": u_cmd.copy(),
            "r_cmd": r_cmd.copy(),
            "r_cmd_minus_y_sp": r_cmd_minus_y_sp.copy(),
            "governor_active": bool(_inf_norm(r_cmd_minus_y_sp) > float(cfg.get("governor_active_tol", 1.0e-8))),
            "command_move": None if command_move is None else command_move.copy(),
            "command_move_inf": None if command_move is None else _inf_norm(command_move),
            "input_headroom_min": input_headroom_min,
            "input_headroom_frac": float(cfg.get("input_headroom_frac", 0.0)),
            "input_headroom_margin": headroom.copy(),
            "governor_active_lower_mask": np.asarray(u_cmd <= u_lo + active_tol, dtype=bool),
            "governor_active_upper_mask": np.asarray(u_cmd >= u_hi - active_tol, dtype=bool),
            **probe,
        }
    except Exception as exc:
        return {
            "success": False,
            "status": "failed",
            "solver": None,
            "message": repr(exc),
            "objective_value": None,
        }


def solve_target_for_governed_command(
    A_aug: Any,
    B_aug: Any,
    C_aug: Any,
    xhat_aug: Any,
    r_cmd: Any,
    *,
    u_min: Any,
    u_max: Any,
    u_ref: Optional[np.ndarray] = None,
    x_ref: Optional[np.ndarray] = None,
    config: Optional[Dict[str, Any]] = None,
    H: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    cfg = _merge_config(config)
    if not HAS_CVXPY:
        return {
            "success": False,
            "status": "cvxpy_missing",
            "message": "CVXPY is required for the governed target solve.",
        }

    try:
        model = _recover_model(A_aug, B_aug, C_aug, xhat_aug)
        _validate_full_output_target(H, model["n_y"])
        r_cmd = _as_float_array(r_cmd, "r_cmd", ndim=1)
        if r_cmd.size != model["n_y"]:
            raise ValueError(f"r_cmd must have length {model['n_y']}.")
        u_min_arr = _as_float_array(u_min, "u_min", ndim=1)
        u_max_arr = _as_float_array(u_max, "u_max", ndim=1)
        u_lo, u_hi, _ = _headroom_bounds(u_min_arr, u_max_arr, cfg)

        Q_r = _diag_matrix(cfg.get("Qr_diag", cfg.get("W_r_diag")), model["n_y"], default=1.0)
        u_ref_arr = np.zeros(model["n_u"], dtype=float) if u_ref is None else _as_float_array(u_ref, "u_ref", ndim=1)
        if u_ref_arr.size != model["n_u"]:
            raise ValueError(f"u_ref must have length {model['n_u']}.")
        x_ref_arr = None if x_ref is None else _as_float_array(x_ref, "x_ref", ndim=1)
        if x_ref_arr is not None and x_ref_arr.size != model["n_x"]:
            raise ValueError(f"x_ref must have length {model['n_x']}.")
        u_weight = _weight_vector(cfg.get("u_ref_weight", 0.0), model["n_u"], default=0.0)
        x_weight = _weight_vector(cfg.get("x_ref_weight", 0.0), model["n_x"], default=0.0)

        x_var = cp.Variable(model["n_x"])
        u_var = cp.Variable(model["n_u"])
        y_expr = model["C"] @ x_var + model["Cd"] @ model["d_hat"]
        objective = cp.quad_form(y_expr - r_cmd, cp.psd_wrap(Q_r))
        if np.any(u_weight > 0.0):
            objective += cp.sum(cp.multiply(u_weight, cp.square(u_var - u_ref_arr)))
        if x_ref_arr is not None and np.any(x_weight > 0.0):
            objective += cp.sum(cp.multiply(x_weight, cp.square(x_var - x_ref_arr)))
        constraints = [
            x_var == model["A"] @ x_var + model["Bd"] @ model["d_hat"] + model["B"] @ u_var,
            u_var >= u_lo,
            u_var <= u_hi,
        ]
        problem = cp.Problem(cp.Minimize(objective), constraints)
        solve_info = _solve_problem(problem, [x_var, u_var], cfg.get("solver_pref"), warm_start=True)
        if not solve_info["success"]:
            return {
                "success": False,
                "status": solve_info.get("status"),
                "solver": solve_info.get("solver"),
                "message": solve_info.get("error") or "governed target solve failed",
                "objective_value": solve_info.get("objective_value"),
            }

        x_s = np.asarray(x_var.value, dtype=float).reshape(model["n_x"])
        u_s = np.asarray(u_var.value, dtype=float).reshape(model["n_u"])
        d_s = model["d_hat"].copy()
        y_s = np.asarray(model["C"] @ x_s + model["Cd"] @ d_s, dtype=float).reshape(model["n_y"])
        dyn_residual = x_s - (model["A"] @ x_s + model["Bd"] @ d_s + model["B"] @ u_s)
        output_residual = y_s - r_cmd
        residual_total = np.concatenate([dyn_residual, output_residual])
        active_tol = float(cfg.get("active_bound_tol", 1.0e-8))
        u_diff = u_s - u_ref_arr
        x_diff = None if x_ref_arr is None else x_s - x_ref_arr

        return {
            "success": True,
            "status": solve_info.get("status"),
            "solver": solve_info.get("solver"),
            "message": "ok",
            "objective_value": solve_info.get("objective_value"),
            "x_s": x_s.copy(),
            "u_s": u_s.copy(),
            "d_s": d_s.copy(),
            "y_s": y_s.copy(),
            "x_s_aug": np.concatenate([x_s, d_s]),
            "residual_dyn": dyn_residual.copy(),
            "residual_out": output_residual.copy(),
            "residual_total": residual_total.copy(),
            "residual_dyn_norm": _inf_norm(dyn_residual),
            "residual_out_norm": _inf_norm(output_residual),
            "residual_total_norm": float(np.linalg.norm(residual_total)),
            "u_ref": u_ref_arr.copy(),
            "u_ref_weight": u_weight.copy(),
            "u_ref_active": bool(np.any(u_weight > 0.0)),
            "u_ref_penalty": float(np.sum(u_weight * np.square(u_diff))),
            "us_u_ref_inf": _inf_norm(u_diff),
            "x_ref": None if x_ref_arr is None else x_ref_arr.copy(),
            "x_ref_weight": None if x_ref_arr is None else x_weight.copy(),
            "x_ref_active": bool(x_ref_arr is not None and np.any(x_weight > 0.0)),
            "x_ref_penalty": None if x_diff is None else float(np.sum(x_weight * np.square(x_diff))),
            "xs_x_ref_inf": None if x_diff is None else _inf_norm(x_diff),
            "bounded_solution_used": True,
            "bounded_solver_name": solve_info.get("solver"),
            "bounded_solve_form": "governed_reference_cvxpy",
            "bounded_status": solve_info.get("status"),
            "bounded_message": "ok",
            "bounded_residual_norm": float(np.linalg.norm(residual_total)),
            "bounded_state_residual_inf": _inf_norm(dyn_residual),
            "bounded_output_residual_inf": _inf_norm(output_residual),
            "bounded_active_lower_mask": np.asarray(u_s <= u_lo + active_tol, dtype=bool),
            "bounded_active_upper_mask": np.asarray(u_s >= u_hi - active_tol, dtype=bool),
            "input_headroom_min": float(np.min(np.minimum(u_s - u_min_arr, u_max_arr - u_s))),
        }
    except Exception as exc:
        return {
            "success": False,
            "status": "failed",
            "solver": None,
            "message": repr(exc),
            "objective_value": None,
        }


def solve_governed_reference_target(
    A_aug: Any,
    B_aug: Any,
    C_aug: Any,
    xhat_aug: Any,
    y_sp: Any,
    *,
    u_min: Any,
    u_max: Any,
    config: Optional[Dict[str, Any]] = None,
    H: Optional[np.ndarray] = None,
    u_ref: Optional[np.ndarray] = None,
    x_ref: Optional[np.ndarray] = None,
    r_prev: Optional[np.ndarray] = None,
    P_x: Optional[np.ndarray] = None,
    rho_lyap: Optional[float] = None,
    eps_lyap: Optional[float] = None,
) -> Dict[str, Any]:
    y_sp_arr = _as_float_array(y_sp, "y_sp", ndim=1)
    command_info = solve_governed_command(
        A_aug,
        B_aug,
        C_aug,
        xhat_aug,
        y_sp_arr,
        u_min=u_min,
        u_max=u_max,
        config=config,
        H=H,
        r_prev=r_prev,
        P_x=P_x,
        rho_lyap=rho_lyap,
        eps_lyap=eps_lyap,
    )
    if not bool(command_info.get("success", False)):
        return {
            "success": False,
            "mode": "governed_reference",
            "target_variant": "governed_reference",
            "solve_stage": "governed_command_failed",
            "status": command_info.get("status"),
            "solver": command_info.get("solver"),
            "message": command_info.get("message"),
            "objective_value": command_info.get("objective_value"),
            "x_s": None,
            "u_s": None,
            "d_s": None,
            "x_s_aug": None,
            "y_s": None,
            "r_cmd": None,
            "r_cmd_minus_y_sp": None,
            "y_s_minus_r_cmd": None,
            "y_s_minus_y_sp": None,
            "governor_active": None,
            "governor_probe_margin": command_info.get("governor_probe_margin"),
            "input_headroom_min": command_info.get("input_headroom_min"),
            "disturbance_model_mode": "output",
        }

    target_info = solve_target_for_governed_command(
        A_aug,
        B_aug,
        C_aug,
        xhat_aug,
        command_info["r_cmd"],
        u_min=u_min,
        u_max=u_max,
        u_ref=u_ref,
        x_ref=x_ref,
        config=config,
        H=H,
    )
    target_info = dict(target_info)
    r_cmd = np.asarray(command_info["r_cmd"], dtype=float).reshape(-1)
    y_s = None if target_info.get("y_s") is None else np.asarray(target_info["y_s"], dtype=float).reshape(-1)
    y_s_minus_r_cmd = None if y_s is None else y_s - r_cmd
    y_s_minus_y_sp = None if y_s is None else y_s - y_sp_arr

    target_info.update(
        {
            "mode": "governed_reference",
            "target_variant": "governed_reference",
            "solve_stage": "governed_reference_target"
            if bool(target_info.get("success", False))
            else "governed_reference_target_failed",
            "r_s": None if y_s is None else y_s.copy(),
            "target_error": None if y_s is None else y_s_minus_y_sp.copy(),
            "target_error_inf": None if y_s is None else _inf_norm(y_s_minus_y_sp),
            "target_error_norm": None if y_s is None else float(np.linalg.norm(y_s_minus_y_sp)),
            "target_eq_residual_inf": None if y_s_minus_r_cmd is None else _inf_norm(y_s_minus_r_cmd),
            "target_slack": None if y_s is None else y_s_minus_y_sp.copy(),
            "target_slack_inf": None if y_s is None else _inf_norm(y_s_minus_y_sp),
            "target_slack_2": None if y_s is None else float(np.linalg.norm(y_s_minus_y_sp)),
            "r_cmd": r_cmd.copy(),
            "r_cmd_minus_y_sp": np.asarray(command_info["r_cmd_minus_y_sp"], dtype=float).reshape(-1).copy(),
            "y_s_minus_r_cmd": None if y_s_minus_r_cmd is None else y_s_minus_r_cmd.copy(),
            "y_s_minus_y_sp": None if y_s_minus_y_sp is None else y_s_minus_y_sp.copy(),
            "governor_active": command_info.get("governor_active"),
            "command_move": command_info.get("command_move"),
            "command_move_inf": command_info.get("command_move_inf"),
            "input_headroom_min": target_info.get("input_headroom_min", command_info.get("input_headroom_min")),
            "input_headroom_frac": command_info.get("input_headroom_frac"),
            "governor_status": command_info.get("status"),
            "governor_solver": command_info.get("solver"),
            "governor_objective_value": command_info.get("objective_value"),
            "governor_probe_available": command_info.get("governor_probe_available"),
            "governor_probe_success": command_info.get("governor_probe_success"),
            "governor_probe_margin": command_info.get("governor_probe_margin"),
            "governor_probe_min_value": command_info.get("governor_probe_min_value"),
            "governor_probe_bound": command_info.get("governor_probe_bound"),
            "governor_probe_status": command_info.get("governor_probe_status"),
            "governor_active_lower_mask": command_info.get("governor_active_lower_mask"),
            "governor_active_upper_mask": command_info.get("governor_active_upper_mask"),
            "disturbance_model_mode": "output",
            "rank_M": None,
            "rank_G": None,
            "cond_M": None,
            "cond_G": None,
            "exact_within_bounds": None,
            "exact_active_lower_mask": None,
            "exact_active_upper_mask": None,
        }
    )
    return target_info
