from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import numpy as np

try:
    import cvxpy as cp

    HAS_CVXPY = True
except Exception:
    cp = None
    HAS_CVXPY = False

from Lyapunov.gart_target import GARTTargetResult, jsonable
from Lyapunov.lyapunov_core import (
    _OPTIMAL_STATUSES,
    _TRACKING_TOL_BY_STATUS,
    _bounds_to_horizon_matrices,
    _extract_num_iters,
    compute_terminal_alpha_input_only,
    first_step_contraction_metrics,
    lyapunov_bound,
    lyapunov_value,
)
from utils.lyapunov_utils import reshape_u_sequence, shift_input_guess, tracking_solver_sequence


@dataclass
class GARTMPCConfig:
    Q_raw_diag: np.ndarray
    Q_target_diag: np.ndarray
    R_us_diag: np.ndarray
    Rdu_diag: np.ndarray
    eta_y: float = 0.1
    eta_u: float = 0.1
    slack_penalty: float = 1.0e6
    terminal_set_on: bool = True
    first_step_contraction_on: bool = True
    lyapunov_mode: str = "soft"
    rho: float = 0.99
    eps: float = 1.0e-3
    alpha_terminal_min: float = 1.0e-8
    target_term_gate_enabled: bool = True
    target_term_gate_delta_y: float = 0.5
    target_term_gate_min_alpha: float = 0.5
    target_term_gate_disable_on_hold: bool = True
    eta_y_when_gated: float | None = None
    eta_u_when_gated: float | None = None
    solver_options: dict[str, Any] | None = None


def _as_vector(value: Any, name: str, size: int | None = None) -> np.ndarray:
    arr = np.asarray(value, dtype=float).reshape(-1)
    if size is not None and arr.size != size:
        raise ValueError(f"{name} must have length {size}, got {arr.size}.")
    return arr


def _diag_matrix(value: Any, size: int, *, default: float = 0.0) -> np.ndarray:
    arr = np.asarray(value if value is not None else default, dtype=float).reshape(-1)
    if arr.size == 0:
        arr = np.full(size, float(default), dtype=float)
    elif arr.size == 1:
        arr = np.full(size, float(arr.item()), dtype=float)
    elif arr.size != size:
        raise ValueError(f"Expected scalar or vector length {size}, got {arr.size}.")
    return np.diag(np.maximum(arr, 0.0))


def _as_mode(value: str, allowed: tuple[str, ...], name: str) -> str:
    mode = str(value or "").strip().lower()
    if mode not in allowed:
        raise ValueError(f"{name} must be one of {allowed}, got {value!r}.")
    return mode


def _target_attr(target: GARTTargetResult | dict[str, Any], name: str) -> Any:
    if isinstance(target, dict):
        return target.get(name)
    return getattr(target, name, None)


def effective_target_objective_weights(
    *,
    target: GARTTargetResult | dict[str, Any],
    config: GARTMPCConfig,
) -> tuple[float, float, dict[str, Any]]:
    eta_y = float(config.eta_y_when_gated if config.eta_y_when_gated is not None else config.eta_y)
    eta_u = float(config.eta_u_when_gated if config.eta_u_when_gated is not None else config.eta_u)

    if not bool(config.target_term_gate_enabled):
        return eta_y, eta_u, {
            "target_term_gate_active": False,
            "target_terms_enabled": True,
            "target_term_gate_reason": "disabled",
            "eta_y_eff": eta_y,
            "eta_u_eff": eta_u,
        }

    mismatch = _target_attr(target, "target_error_inf")
    alpha = _target_attr(target, "governor_alpha")
    hold = bool(_target_attr(target, "hold_previous"))

    reasons: list[str] = []
    enabled = True
    if mismatch is None or float(mismatch) > float(config.target_term_gate_delta_y):
        enabled = False
        reasons.append("target_setpoint_mismatch")
    if alpha is not None and float(alpha) < float(config.target_term_gate_min_alpha):
        enabled = False
        reasons.append("governor_alpha_too_small")
    if config.target_term_gate_disable_on_hold and hold:
        enabled = False
        reasons.append("hold_previous")

    if not enabled:
        return 0.0, 0.0, {
            "target_term_gate_active": True,
            "target_terms_enabled": False,
            "target_term_gate_reason": ",".join(reasons),
            "ungated_eta_y": eta_y,
            "ungated_eta_u": eta_u,
            "eta_y_eff": 0.0,
            "eta_u_eff": 0.0,
        }

    return eta_y, eta_u, {
        "target_term_gate_active": True,
        "target_terms_enabled": True,
        "target_term_gate_reason": "ok",
        "ungated_eta_y": eta_y,
        "ungated_eta_u": eta_u,
        "eta_y_eff": eta_y,
        "eta_u_eff": eta_u,
    }


def _make_solution_report(
    *,
    LMPC_obj: Any,
    x_opt_flat: np.ndarray,
    x_pred: np.ndarray,
    x0_aug: np.ndarray,
    x_s: np.ndarray,
    u_s: np.ndarray,
    y_sp: np.ndarray,
    y_s: np.ndarray,
    u_prev_dev: np.ndarray,
    alpha_terminal: float | None,
    rho_lyap: float,
    lyap_eps: float,
    first_step_contraction_on: bool,
    lyapunov_mode: str,
    slack_lyap: float,
    slack_penalty: float,
) -> dict[str, Any]:
    if hasattr(LMPC_obj, "standard_tracking_report"):
        report = LMPC_obj.standard_tracking_report(
            x_opt=x_opt_flat,
            x0_aug=x0_aug,
            x_s=x_s,
            u_s=u_s,
            y_target=y_sp,
            u_prev_dev=u_prev_dev,
            alpha_terminal=alpha_terminal,
            rho_lyap=rho_lyap,
            eps_lyap=lyap_eps,
            first_step_contraction_on=first_step_contraction_on,
            lyapunov_mode=lyapunov_mode,
            slack_lyap=slack_lyap,
            slack_penalty=slack_penalty,
        )
    else:
        report = {}
    if first_step_contraction_on:
        contraction = first_step_contraction_metrics(
            x0_aug=x0_aug,
            x_pred=x_pred,
            x_s=x_s,
            P_x=LMPC_obj.P_x,
            rho=rho_lyap,
            eps_lyap=lyap_eps,
        )
        report.update(contraction)
    mpc_contraction_violation = report.get("contraction_margin")
    mpc_contraction_margin_good = None if mpc_contraction_violation is None else -float(mpc_contraction_violation)
    y_pred = (LMPC_obj.C @ x_pred[:, 1:]).T
    report.update(
        {
            "mpc_contraction_violation": mpc_contraction_violation,
            "mpc_contraction_margin_good": mpc_contraction_margin_good,
            "y_target": y_sp.copy(),
            "y_s": y_s.copy(),
            "y_s_minus_y_sp": y_s - y_sp,
            "y_target_minus_y_sp": np.zeros_like(y_sp),
            "gart_y_pred_minus_y_sp_rmse": float(np.sqrt(np.mean((y_pred - y_sp.reshape(1, -1)) ** 2))),
            "gart_y_pred_minus_y_s_rmse": float(np.sqrt(np.mean((y_pred - y_s.reshape(1, -1)) ** 2))),
            "lyapunov_mode": lyapunov_mode,
            "slack_lyap": float(slack_lyap),
            "slack_penalty": float(slack_penalty),
        }
    )
    return report


def solve_gart_lmpc_step(
    LMPC_obj: Any,
    x0_aug: np.ndarray,
    y_sp: np.ndarray,
    target: GARTTargetResult | dict[str, Any],
    u_prev_dev: np.ndarray,
    IC_opt: np.ndarray,
    bnds: Any,
    u_dev_min: np.ndarray,
    u_dev_max: np.ndarray,
    config: GARTMPCConfig,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if not HAS_CVXPY:
        raise ImportError("CVXPY is required for GART-LMPC.")

    n_u = int(LMPC_obj.B.shape[1])
    n_y = int(LMPC_obj.C.shape[0])
    n_aug = int(LMPC_obj.A.shape[0])
    n_x = int(n_aug - n_y)
    NP = int(LMPC_obj.NP)
    NC = int(LMPC_obj.NC)

    x0_aug = _as_vector(x0_aug, "x0_aug", n_aug)
    y_sp = _as_vector(y_sp, "y_sp", n_y)
    u_prev_dev = _as_vector(u_prev_dev, "u_prev_dev", n_u)
    IC_opt = np.asarray(IC_opt, dtype=float).reshape(-1)
    u_dev_min = _as_vector(u_dev_min, "u_dev_min", n_u)
    u_dev_max = _as_vector(u_dev_max, "u_dev_max", n_u)
    lyapunov_mode = _as_mode(config.lyapunov_mode, ("hard", "soft"), "lyapunov_mode")

    step_info: dict[str, Any] = {
        "success": False,
        "method": "gart_lmpc",
        "lyapunov_mode": lyapunov_mode,
        "target_success": bool(_target_attr(target, "success")),
        "target_solve_success": _target_attr(target, "solve_success"),
        "target_accepted": _target_attr(target, "accepted"),
        "target_usable_for_lmpc": _target_attr(target, "usable_for_lmpc"),
        "target_rejection_reason": _target_attr(target, "rejection_reason"),
        "target_status": _target_attr(target, "status"),
        "target_stage": _target_attr(target, "stage"),
        "target_error_inf": _target_attr(target, "target_error_inf"),
        "governor_alpha": _target_attr(target, "governor_alpha"),
        "governor_active": _target_attr(target, "governor_active"),
        "hold_previous": _target_attr(target, "hold_previous"),
        "contraction_probe_success": _target_attr(target, "contraction_probe_success"),
        "contraction_probe_margin_good": _target_attr(target, "contraction_probe_margin_good"),
        "contraction_probe_margin": _target_attr(target, "contraction_probe_margin"),
        "input_headroom_min": _target_attr(target, "input_headroom_min"),
        "u_prev_dev": u_prev_dev.copy(),
        "y_sp": y_sp.copy(),
        "slack_lyap": 0.0,
    }

    usable_attr = _target_attr(target, "usable_for_lmpc")
    accepted_attr = _target_attr(target, "accepted")
    target_usable = bool(usable_attr if usable_attr is not None else (accepted_attr if accepted_attr is not None else _target_attr(target, "success")))

    if not target_usable:
        u_hold = np.clip(u_prev_dev, u_dev_min, u_dev_max)
        step_info.update(
            {
                "method": "gart_target_not_usable_hold_prev",
                "u_apply": u_hold.copy(),
                "message": _target_attr(target, "rejection_reason") or "target not usable for LMPC",
                "target_solve_success": _target_attr(target, "solve_success"),
                "target_accepted": _target_attr(target, "accepted"),
                "target_usable_for_lmpc": _target_attr(target, "usable_for_lmpc"),
            }
        )
        return u_hold, np.tile(u_hold, NC), step_info

    x_s = _as_vector(_target_attr(target, "x_s"), "target.x_s", n_x)
    u_s = _as_vector(_target_attr(target, "u_s"), "target.u_s", n_u)
    y_s = _as_vector(_target_attr(target, "y_s"), "target.y_s", n_y)
    step_info.update({"x_s": x_s.copy(), "u_s": u_s.copy(), "y_s": y_s.copy(), "y_s_minus_y_sp": y_s - y_sp})

    alpha_terminal_raw = compute_terminal_alpha_input_only(
        P_x=LMPC_obj.P_x,
        K_x=LMPC_obj.K_x,
        u_s=u_s,
        u_min=u_dev_min,
        u_max=u_dev_max,
        alpha_scale=1.0,
    )
    alpha_terminal = compute_terminal_alpha_input_only(
        P_x=LMPC_obj.P_x,
        K_x=LMPC_obj.K_x,
        u_s=u_s,
        u_min=u_dev_min,
        u_max=u_dev_max,
        alpha_scale=getattr(LMPC_obj, "terminal_alpha_scale", 1.0),
    )
    active_terminal_constraint = bool(
        config.terminal_set_on
        and getattr(LMPC_obj, "terminal_set_on", True)
        and np.isfinite(float(alpha_terminal))
        and float(alpha_terminal) > float(config.alpha_terminal_min)
    )
    alpha_for_solver = float(alpha_terminal) if active_terminal_constraint else None

    lower, upper = _bounds_to_horizon_matrices(bnds, n_u, NC)
    Q_raw = _diag_matrix(config.Q_raw_diag, n_y, default=1.0)
    Q_target = _diag_matrix(config.Q_target_diag, n_y, default=1.0)
    R_us = _diag_matrix(config.R_us_diag, n_u, default=1.0)
    Rdu = _diag_matrix(config.Rdu_diag, n_u, default=0.0)
    use_soft_slack = bool(lyapunov_mode == "soft" and config.first_step_contraction_on)
    eta_y_eff, eta_u_eff, target_gate_info = effective_target_objective_weights(target=target, config=config)
    step_info.update(target_gate_info)

    u_var = cp.Variable((NC, n_u))
    x_var = cp.Variable((n_aug, NP + 1))
    lyap_slack = cp.Variable(nonneg=True) if use_soft_slack else None
    constraints = [x_var[:, 0] == x0_aug]
    if lower is not None:
        lower_rows, lower_cols = np.where(np.isfinite(lower))
        for row_idx, col_idx in zip(lower_rows, lower_cols):
            constraints.append(u_var[row_idx, col_idx] >= float(lower[row_idx, col_idx]))
    if upper is not None:
        upper_rows, upper_cols = np.where(np.isfinite(upper))
        for row_idx, col_idx in zip(upper_rows, upper_cols):
            constraints.append(u_var[row_idx, col_idx] <= float(upper[row_idx, col_idx]))

    objective = 0.0
    for step_idx in range(NP):
        ctrl_idx = step_idx if step_idx < NC else NC - 1
        constraints.append(x_var[:, step_idx + 1] == LMPC_obj.A @ x_var[:, step_idx] + LMPC_obj.B @ u_var[ctrl_idx, :])
        y_expr = LMPC_obj.C @ x_var[:, step_idx + 1]
        if getattr(LMPC_obj, "D", None) is not None:
            y_expr = y_expr + LMPC_obj.D @ u_var[ctrl_idx, :]
        objective += cp.quad_form(y_expr - y_sp, Q_raw)
        if float(eta_y_eff) != 0.0:
            objective += float(eta_y_eff) * cp.quad_form(y_expr - y_s, Q_target)

    if float(eta_u_eff) != 0.0:
        for ctrl_idx in range(NC):
            objective += float(eta_u_eff) * cp.quad_form(u_var[ctrl_idx, :] - u_s, R_us)
    if np.any(np.diag(Rdu) > 0.0):
        objective += cp.quad_form(u_var[0, :] - u_prev_dev, Rdu)
        for ctrl_idx in range(1, NC):
            objective += cp.quad_form(u_var[ctrl_idx, :] - u_var[ctrl_idx - 1, :], Rdu)

    terminal_error = x_var[:n_x, NP] - x_s
    terminal_value_expr = cp.quad_form(terminal_error, cp.psd_wrap(LMPC_obj.P_x))
    if active_terminal_constraint:
        constraints.append(terminal_value_expr <= float(alpha_for_solver))

    if config.first_step_contraction_on:
        V_k = lyapunov_value(x0_aug[:n_x] - x_s, LMPC_obj.P_x)
        V_bound = float(lyapunov_bound(V_k, rho=config.rho, eps_lyap=config.eps))
        first_step_error = x_var[:n_x, 1] - x_s
        first_step_value_expr = cp.quad_form(first_step_error, cp.psd_wrap(LMPC_obj.P_x))
        if use_soft_slack:
            constraints.append(first_step_value_expr <= V_bound + lyap_slack)
        else:
            constraints.append(first_step_value_expr <= V_bound)

    if use_soft_slack:
        objective += float(config.slack_penalty) * lyap_slack

    problem = cp.Problem(cp.Minimize(objective), constraints)
    if IC_opt.size == n_u * NC:
        try:
            u_guess = reshape_u_sequence(IC_opt, n_u, NC)
            u_var.value = u_guess
            if hasattr(LMPC_obj, "_predict_from_sequence"):
                x_guess, _ = LMPC_obj._predict_from_sequence(u_guess, x0_aug)
                x_var.value = x_guess
        except Exception:
            pass

    options = {} if config.solver_options is None else dict(config.solver_options)
    solver_pref_override = options.pop("solver_pref", None)
    warm_start = bool(options.pop("warm_start", True))
    verbose = bool(options.pop("verbose", False))
    solve_kwargs = dict(options.pop("solve_kwargs", {}))
    if solver_pref_override is None:
        needs_conic = bool(active_terminal_constraint or config.first_step_contraction_on)
        solver_pref = LMPC_obj.solver_pref_conic if needs_conic else LMPC_obj.solver_pref_qp
    else:
        solver_pref = solver_pref_override
    solver_sequence = tracking_solver_sequence(bool(active_terminal_constraint or config.first_step_contraction_on), solver_pref=solver_pref)

    last_status = None
    last_solver = None
    last_error = None
    last_objective = None
    last_nit = None
    last_slack = 0.0
    for solver_name in solver_sequence:
        try:
            problem.solve(solver=solver_name, warm_start=warm_start, verbose=verbose, **solve_kwargs)
            last_status = problem.status
            last_solver = solver_name
            last_nit = _extract_num_iters(problem)
            if problem.value is not None:
                last_objective = float(problem.value)
            if problem.status not in _OPTIMAL_STATUSES or u_var.value is None or x_var.value is None:
                continue
            if use_soft_slack and lyap_slack is not None and lyap_slack.value is None:
                continue
            u_value = np.asarray(u_var.value, dtype=float)
            x_value = np.asarray(x_var.value, dtype=float)
            last_slack = float(np.asarray(lyap_slack.value).item()) if use_soft_slack and lyap_slack is not None else 0.0
            report = _make_solution_report(
                LMPC_obj=LMPC_obj,
                x_opt_flat=u_value.reshape(-1),
                x_pred=x_value,
                x0_aug=x0_aug,
                x_s=x_s,
                u_s=u_s,
                y_sp=y_sp,
                y_s=y_s,
                u_prev_dev=u_prev_dev,
                alpha_terminal=alpha_for_solver,
                rho_lyap=config.rho,
                lyap_eps=config.eps,
                first_step_contraction_on=config.first_step_contraction_on,
                lyapunov_mode=lyapunov_mode,
                slack_lyap=last_slack,
                slack_penalty=config.slack_penalty,
            )
            accepted = True
            reject_reason = None
            if report.get("bound_violation_inf") is not None and float(report["bound_violation_inf"]) > _TRACKING_TOL_BY_STATUS.get(problem.status, 1.0e-5):
                accepted = False
                reject_reason = "bound_violation"
            if lyapunov_mode == "hard" and config.first_step_contraction_on and report.get("first_step_contraction_satisfied") is False:
                accepted = False
                reject_reason = "first_step_contraction"
            if lyapunov_mode == "soft" and config.first_step_contraction_on:
                margin = report.get("contraction_margin")
                relaxed_violation = None if margin is None else max(float(margin) - last_slack, 0.0)
                relaxed_ok = None if relaxed_violation is None else bool(relaxed_violation <= _TRACKING_TOL_BY_STATUS.get(problem.status, 1.0e-5))
                report["relaxed_contraction_satisfied"] = relaxed_ok
                report["relaxed_contraction_violation"] = relaxed_violation
                if relaxed_ok is False:
                    accepted = False
                    reject_reason = "soft_contraction_constraint"
            if not accepted:
                last_error = reject_reason
                continue
            u_dev_apply = np.clip(u_value[0, :], u_dev_min, u_dev_max)
            IC_opt_next = shift_input_guess(u_value.reshape(-1), n_u, NC)
            step_info.update(
                {
                    "success": True,
                    "status": problem.status,
                    "message": "optimal",
                    "fun": last_objective,
                    "solver_nit": last_nit,
                    "tracking_solver": solver_name,
                    "objective_value": last_objective,
                    "alpha_terminal_raw": float(alpha_terminal_raw),
                    "alpha_terminal": float(alpha_terminal),
                    "alpha_terminal_used": None if alpha_for_solver is None else float(alpha_for_solver),
                    "terminal_constraint_skipped": not active_terminal_constraint,
                    "first_step_contraction_on": bool(config.first_step_contraction_on),
                    "u_apply": u_dev_apply.copy(),
                    "u_sequence": u_value.copy(),
                    "x_pred_path": x_value.copy(),
                    **report,
                }
            )
            return u_dev_apply, IC_opt_next, step_info
        except Exception as exc:
            last_error = repr(exc)

    u_hold = np.clip(u_prev_dev, u_dev_min, u_dev_max)
    step_info.update(
        {
            "method": "gart_solver_fail_hold_prev",
            "status": last_status,
            "message": last_error or "solver_status",
            "fun": last_objective,
            "solver_nit": last_nit,
            "tracking_solver": last_solver,
            "slack_lyap": last_slack,
            "u_apply": u_hold.copy(),
            "alpha_terminal_raw": float(alpha_terminal_raw),
            "alpha_terminal": float(alpha_terminal),
            "alpha_terminal_used": None if alpha_for_solver is None else float(alpha_for_solver),
            "terminal_constraint_skipped": not active_terminal_constraint,
        }
    )
    return u_hold, np.tile(u_hold, NC), step_info


__all__ = ["GARTMPCConfig", "effective_target_objective_weights", "solve_gart_lmpc_step", "jsonable"]
