from dataclasses import asdict, dataclass

import numpy as np

try:
    import cvxpy as cp

    HAS_CVXPY = True
except Exception:
    HAS_CVXPY = False

from utils.lyapunov_utils import DEFAULT_CVXPY_SOLVERS, diag_psd_from_vector, vector_or_zeros


_OPTIMAL_STATUSES = {"optimal", "optimal_inaccurate"}
_DYN_TOL_BY_STATUS = {
    "optimal": 1e-6,
    "optimal_inaccurate": 1e-5,
}

TARGET_SELECTOR_MODES = (
    "current_exact_fallback_frozen_d",
    "free_disturbance_prior",
    "compromised_reference",
    "single_stage_robust_sstp",
)


@dataclass
class TargetSelectorConfig:
    selector_mode: str = "current_exact_fallback_frozen_d"
    u_nom: object = None
    Ty_diag: object = None
    Qr_diag: object = None
    Ru_diag: object = None
    Qx_diag: object = None
    w_x: float = 1e-6
    Qdx_diag: object = None
    Rdu_diag: object = None
    Qd_diag: object = None
    delta_d_inf: object = None
    rho_x: float = 1e5
    rho_y: float = 1e5
    soft_output_bounds: bool = True
    Wy_low_diag: object = None
    Wy_high_diag: object = None
    u_tight: object = None
    y_tight: object = None
    freeze_d_at_estimate: bool = True
    solver_pref: object = None
    accept_statuses: tuple = ("optimal", "optimal_inaccurate")
    tol_optimal: float = 1e-6
    tol_optimal_inaccurate: float = 1e-5


def _solver_sequence(solver_pref):
    if solver_pref is None:
        return DEFAULT_CVXPY_SOLVERS
    if isinstance(solver_pref, str):
        return (solver_pref,)
    return tuple(solver_pref)


def _reset_variable_values(variables):
    for var in variables:
        if var is not None:
            var.value = None


def _apply_initial_values(variables, initial_values):
    if initial_values is None:
        return
    for var, init in zip(variables, initial_values):
        if var is None or init is None:
            continue
        var.value = np.asarray(init, float).copy()


def _solve_problem_with_preferences(problem, variables, solver_pref, warm_start=False, initial_values=None):
    last_status = None
    last_solver = None
    last_err = None

    for solver_name in _solver_sequence(solver_pref):
        try:
            _reset_variable_values(variables)
            _apply_initial_values(variables, initial_values)
            problem.solve(solver=solver_name, warm_start=bool(warm_start), verbose=False)
            last_status = problem.status
            last_solver = solver_name

            if any(var is not None and var.value is None for var in variables):
                continue
            if problem.status in _OPTIMAL_STATUSES:
                return {
                    "accepted_by_status": True,
                    "status": problem.status,
                    "solver": solver_name,
                    "error": None,
                    "objective_value": float(problem.value) if problem.value is not None else None,
                }
        except Exception as exc:
            last_err = repr(exc)

    return {
        "accepted_by_status": False,
        "status": last_status,
        "solver": last_solver,
        "error": last_err,
        "objective_value": float(problem.value) if problem.value is not None else None,
    }


def _bound_violation_inf(value, lower=None, upper=None):
    violation = 0.0
    if lower is not None:
        violation = max(violation, float(np.max(np.maximum(lower - value, 0.0))))
    if upper is not None:
        violation = max(violation, float(np.max(np.maximum(value - upper, 0.0))))
    return violation


def _output_slack_value(var, size):
    if var is None or var.value is None:
        return np.zeros(size, dtype=float)
    return np.asarray(var.value, float).reshape(-1)


def compute_ss_target_refined_rawlings(
    A_aug,
    B_aug,
    C_aug,
    xhat_aug,
    y_sp,
    u_min,
    u_max,
    u_nom=None,
    Ty_diag=None,
    Ru_diag=None,
    Qx_diag=None,
    w_x=1e-6,
    x_s_prev=None,
    u_s_prev=None,
    Qdx_diag=None,
    Rdu_diag=None,
    y_min=None,
    y_max=None,
    u_tight=None,
    y_tight=None,
    soft_output_bounds=True,
    Wy_low_diag=None,
    Wy_high_diag=None,
    solver_pref=DEFAULT_CVXPY_SOLVERS,
    warm_start=True,
    return_debug=False,
    H=None,
):
    """
    Compute a steady-state offset-free target using a two-stage Rawlings-style selector.

    The expected convention is that all quantities are expressed in the same coordinates.
    In this codebase that should usually mean scaled deviation coordinates, so `u_nom`
    is typically the zero vector.

    Stage 1 solves an exact steady-state target problem with hard equality on the
    controlled outputs. Stage 2 is only used if Stage 1 is infeasible or numerically
    unacceptable; it computes the closest feasible equilibrium target. When
    `soft_output_bounds` is enabled, only output inequality bounds are softened. The
    exact target equality in Stage 1 remains hard.
    """
    if not HAS_CVXPY:
        raise ImportError("CVXPY is required for the refined target selector.")

    A_aug = np.asarray(A_aug, float)
    B_aug = np.asarray(B_aug, float)
    C_aug = np.asarray(C_aug, float)
    xhat_aug = np.asarray(xhat_aug, float).reshape(-1)
    y_sp = np.asarray(y_sp, float).reshape(-1)
    u_min = np.asarray(u_min, float).reshape(-1)
    u_max = np.asarray(u_max, float).reshape(-1)

    if A_aug.ndim != 2 or A_aug.shape[0] != A_aug.shape[1]:
        raise ValueError("A_aug must be square.")
    if B_aug.ndim != 2 or B_aug.shape[0] != A_aug.shape[0]:
        raise ValueError("B_aug has incompatible shape.")
    if C_aug.ndim != 2 or C_aug.shape[1] != A_aug.shape[0]:
        raise ValueError("C_aug has incompatible shape.")
    if xhat_aug.size != A_aug.shape[0]:
        raise ValueError("xhat_aug has incorrect size.")

    n_aug = A_aug.shape[0]
    n_y = C_aug.shape[0]
    n_x = n_aug - n_y
    n_u = B_aug.shape[1]

    if n_x <= 0:
        raise ValueError("Invalid augmented model: inferred physical state dimension must be positive.")
    if u_min.size != n_u or u_max.size != n_u:
        raise ValueError("u_min and u_max must both have size n_u.")

    if H is None:
        H_arr = None
        n_c = n_y
    else:
        H_arr = np.asarray(H, float)
        if H_arr.ndim != 2 or H_arr.shape[1] != n_y:
            raise ValueError("H must have shape (n_c, n_y).")
        n_c = H_arr.shape[0]

    if y_sp.size != n_c:
        raise ValueError(f"y_sp has incorrect size. Expected {n_c}, got {y_sp.size}.")

    A = A_aug[:n_x, :n_x]
    Bd = A_aug[:n_x, n_x:]
    B = B_aug[:n_x, :]
    C = C_aug[:, :n_x]
    Cd = C_aug[:, n_x:]
    d_hat = xhat_aug[n_x:].copy()

    if d_hat.size != n_y:
        raise ValueError(
            "This selector assumes the augmented state is ordered as [x; d] with len(d) == n_y."
        )

    if u_nom is None:
        u_nom = np.zeros(n_u, dtype=float)
    else:
        u_nom = np.asarray(u_nom, float).reshape(-1)
        if u_nom.size != n_u:
            raise ValueError("u_nom has incorrect size.")

    u_tight = np.maximum(vector_or_zeros(u_tight, n_u), 0.0)
    y_tight = np.maximum(vector_or_zeros(y_tight, n_y), 0.0)

    u_lo = u_min + u_tight
    u_hi = u_max - u_tight
    if np.any(u_lo > u_hi):
        raise ValueError("Input tightening is too large. Tightened bounds are infeasible.")

    use_y_lo = y_min is not None
    use_y_hi = y_max is not None
    if use_y_lo:
        y_min = np.asarray(y_min, float).reshape(-1)
        if y_min.size != n_y:
            raise ValueError("y_min has incorrect size.")
        y_lo = y_min + y_tight
    else:
        y_lo = None
    if use_y_hi:
        y_max = np.asarray(y_max, float).reshape(-1)
        if y_max.size != n_y:
            raise ValueError("y_max has incorrect size.")
        y_hi = y_max - y_tight
    else:
        y_hi = None
    if y_lo is not None and y_hi is not None and np.any(y_lo > y_hi):
        raise ValueError("Output tightening is too large. Tightened output bounds are infeasible.")

    Ty, Ty_used = diag_psd_from_vector(Ty_diag, n_c, eps=1e-12, default=1.0)
    Ru, Ru_used = diag_psd_from_vector(Ru_diag, n_u, eps=1e-12, default=1.0)

    if Qx_diag is None:
        Qx = float(w_x) * np.eye(n_x, dtype=float)
        Qx_used = np.full(n_x, float(w_x), dtype=float)
    else:
        Qx, Qx_used = diag_psd_from_vector(Qx_diag, n_x, eps=1e-12, default=max(float(w_x), 1e-12))

    use_x_prev = x_s_prev is not None and Qdx_diag is not None
    use_u_prev = u_s_prev is not None and Rdu_diag is not None

    if use_x_prev:
        x_s_prev = np.asarray(x_s_prev, float).reshape(-1)
        if x_s_prev.size != n_x:
            raise ValueError("x_s_prev has incorrect size.")
        Qdx, Qdx_used = diag_psd_from_vector(Qdx_diag, n_x, eps=1e-12, default=0.0)
    else:
        x_s_prev = None
        Qdx = None
        Qdx_used = None

    if use_u_prev:
        u_s_prev = np.asarray(u_s_prev, float).reshape(-1)
        if u_s_prev.size != n_u:
            raise ValueError("u_s_prev has incorrect size.")
        Rdu, Rdu_used = diag_psd_from_vector(Rdu_diag, n_u, eps=1e-12, default=0.0)
    else:
        u_s_prev = None
        Rdu = None
        Rdu_used = None

    I_minus_A = np.eye(n_x, dtype=float) - A

    def build_stage_problem(stage_name):
        x = cp.Variable(n_x)
        u = cp.Variable(n_u)
        y_expr = C @ x + Cd @ d_hat
        yc_expr = y_expr if H_arr is None else H_arr @ y_expr

        constraints = [
            I_minus_A @ x - B @ u - Bd @ d_hat == 0.0,
            u >= u_lo,
            u <= u_hi,
        ]

        objective = cp.quad_form(u - u_nom, Ru) + cp.quad_form(x, Qx)
        s_y_low = None
        s_y_high = None
        Wy_low_used = None
        Wy_high_used = None

        if stage_name == "exact":
            constraints.append(yc_expr == y_sp)
        else:
            objective += cp.quad_form(yc_expr - y_sp, Ty)
            if use_x_prev:
                objective += cp.quad_form(x - x_s_prev, Qdx)
            if use_u_prev:
                objective += cp.quad_form(u - u_s_prev, Rdu)

        if y_lo is not None:
            if soft_output_bounds:
                s_y_low = cp.Variable(n_y, nonneg=True)
                Wy_low, Wy_low_used = diag_psd_from_vector(Wy_low_diag, n_y, eps=1e-12, default=1e3)
                objective += cp.quad_form(s_y_low, Wy_low)
                constraints.append(y_expr + s_y_low >= y_lo)
            else:
                constraints.append(y_expr >= y_lo)

        if y_hi is not None:
            if soft_output_bounds:
                s_y_high = cp.Variable(n_y, nonneg=True)
                Wy_high, Wy_high_used = diag_psd_from_vector(Wy_high_diag, n_y, eps=1e-12, default=1e3)
                objective += cp.quad_form(s_y_high, Wy_high)
                constraints.append(y_expr - s_y_high <= y_hi)
            else:
                constraints.append(y_expr <= y_hi)

        return {
            "stage_name": stage_name,
            "problem": cp.Problem(cp.Minimize(objective), constraints),
            "x": x,
            "u": u,
            "y_expr": y_expr,
            "yc_expr": yc_expr,
            "s_y_low": s_y_low,
            "s_y_high": s_y_high,
            "Wy_low_diag_used": Wy_low_used,
            "Wy_high_diag_used": Wy_high_used,
        }

    def evaluate_stage(stage_bundle, solve_info):
        status = solve_info["status"]
        tol = _DYN_TOL_BY_STATUS.get(status)
        result = {
            "solve_info": solve_info,
            "accepted": False,
            "reject_reason": None,
            "x_s": None,
            "u_s": None,
            "y_s": None,
            "yc_s": None,
            "s_y_low": np.zeros(n_y, dtype=float),
            "s_y_high": np.zeros(n_y, dtype=float),
            "dyn_residual_inf": None,
            "target_eq_residual_inf": None,
            "bound_violation_inf": None,
        }

        if not solve_info["accepted_by_status"]:
            result["reject_reason"] = "solver_status"
            return result
        if tol is None:
            result["reject_reason"] = "unsupported_status"
            return result

        x_s = np.asarray(stage_bundle["x"].value, float).reshape(-1)
        u_s = np.asarray(stage_bundle["u"].value, float).reshape(-1)
        y_s = np.asarray(C @ x_s + Cd @ d_hat, float).reshape(-1)
        yc_s = np.asarray(y_s if H_arr is None else H_arr @ y_s, float).reshape(-1)

        dyn_residual = I_minus_A @ x_s - B @ u_s - Bd @ d_hat
        dyn_residual_inf = float(np.max(np.abs(dyn_residual)))
        target_eq_residual = yc_s - y_sp
        target_eq_residual_inf = float(np.max(np.abs(target_eq_residual)))

        bound_violation_inf = _bound_violation_inf(u_s, lower=u_lo, upper=u_hi)
        if not soft_output_bounds:
            bound_violation_inf = max(
                bound_violation_inf,
                _bound_violation_inf(y_s, lower=y_lo, upper=y_hi),
            )

        if dyn_residual_inf > tol:
            result["reject_reason"] = "dyn_residual"
        elif stage_bundle["stage_name"] == "exact" and target_eq_residual_inf > tol:
            result["reject_reason"] = "target_eq_residual"
        elif bound_violation_inf > tol:
            result["reject_reason"] = "bound_violation"
        else:
            result["accepted"] = True

        result.update({
            "x_s": x_s,
            "u_s": u_s,
            "y_s": y_s,
            "yc_s": yc_s,
            "s_y_low": _output_slack_value(stage_bundle["s_y_low"], n_y),
            "s_y_high": _output_slack_value(stage_bundle["s_y_high"], n_y),
            "dyn_residual_inf": dyn_residual_inf,
            "target_eq_residual_inf": target_eq_residual_inf,
            "bound_violation_inf": float(bound_violation_inf),
        })
        return result

    warm_start_enabled = bool(warm_start)
    warm_start_available = bool(x_s_prev is not None or u_s_prev is not None)
    warm_start_used = bool(warm_start_enabled and warm_start_available)

    def stage_warm_values(stage_bundle):
        return [
            None if x_s_prev is None else np.asarray(x_s_prev, float).reshape(-1).copy(),
            None if u_s_prev is None else np.asarray(u_s_prev, float).reshape(-1).copy(),
            None if stage_bundle["s_y_low"] is None else np.zeros(n_y, dtype=float),
            None if stage_bundle["s_y_high"] is None else np.zeros(n_y, dtype=float),
        ]

    exact_stage = build_stage_problem("exact")
    exact_solve = _solve_problem_with_preferences(
        exact_stage["problem"],
        [exact_stage["x"], exact_stage["u"], exact_stage["s_y_low"], exact_stage["s_y_high"]],
        solver_pref,
        warm_start=warm_start_used,
        initial_values=stage_warm_values(exact_stage),
    )
    exact_eval = evaluate_stage(exact_stage, exact_solve)

    fallback_stage = None
    fallback_solve = None
    fallback_eval = None
    final_stage = None
    final_eval = None
    final_stage_name = None

    if exact_eval["accepted"]:
        final_stage = exact_stage
        final_eval = exact_eval
        final_stage_name = "exact"
    else:
        fallback_stage = build_stage_problem("fallback")
        fallback_solve = _solve_problem_with_preferences(
            fallback_stage["problem"],
            [fallback_stage["x"], fallback_stage["u"], fallback_stage["s_y_low"], fallback_stage["s_y_high"]],
            solver_pref,
            warm_start=warm_start_used,
            initial_values=stage_warm_values(fallback_stage),
        )
        fallback_eval = evaluate_stage(fallback_stage, fallback_solve)
        if fallback_eval["accepted"]:
            final_stage = fallback_stage
            final_eval = fallback_eval
            final_stage_name = "fallback"

    primary_failure = fallback_eval if fallback_eval is not None else exact_eval
    primary_solve = primary_failure["solve_info"]

    dbg = {
        "success": bool(final_eval is not None),
        "status": primary_solve["status"],
        "solver": primary_solve["solver"],
        "error": primary_solve["error"],
        "solve_stage": final_stage_name,
        "assumed_augmented_state_order": "[x; d]",
        "assumed_disturbance_dimension_equals_ny": True,
        "Ty_diag_used": Ty_used,
        "Ru_diag_used": Ru_used,
        "Qx_diag_used": Qx_used,
        "Qdx_diag_used": Qdx_used,
        "Rdu_diag_used": Rdu_used,
        "u_tight": u_tight.copy(),
        "y_tight": y_tight.copy(),
        "soft_output_bounds": bool(soft_output_bounds),
        "warm_start_enabled": warm_start_enabled,
        "warm_start_available": warm_start_available,
        "warm_start_used": warm_start_used,
        "state_smoothing_active": bool(use_x_prev),
        "input_smoothing_active": bool(use_u_prev),
        "stage1_status": exact_solve["status"],
        "stage1_solver": exact_solve["solver"],
        "stage1_error": exact_solve["error"],
        "stage1_objective_value": exact_solve["objective_value"],
        "stage1_reject_reason": exact_eval["reject_reason"],
        "stage2_status": None if fallback_solve is None else fallback_solve["status"],
        "stage2_solver": None if fallback_solve is None else fallback_solve["solver"],
        "stage2_error": None if fallback_solve is None else fallback_solve["error"],
        "stage2_objective_value": None if fallback_solve is None else fallback_solve["objective_value"],
        "stage2_reject_reason": None if fallback_eval is None else fallback_eval["reject_reason"],
    }

    if final_eval is None:
        dbg.update({
            "x_s": None,
            "u_s": None,
            "d_s": d_hat.copy(),
            "x_s_aug": None,
            "y_s": None,
            "objective_value": primary_solve["objective_value"],
            "objective": primary_solve["objective_value"],
            "target_error": None,
            "target_error_inf": None,
            "target_error_norm": None,
            "target_slack": None,
            "target_slack_inf": None,
            "target_slack_2": None,
            "target_eq_residual_inf": primary_failure["target_eq_residual_inf"],
            "dyn_residual_inf": primary_failure["dyn_residual_inf"],
            "bound_violation_inf": primary_failure["bound_violation_inf"],
            "target_move_u_inf": None,
            "target_move_x_inf": None,
            "margin_to_u_min": None,
            "margin_to_u_max": None,
            "tight_margin_to_u_min": None,
            "tight_margin_to_u_max": None,
            "s_y_low": None if primary_failure is None else primary_failure["s_y_low"].copy(),
            "s_y_high": None if primary_failure is None else primary_failure["s_y_high"].copy(),
            "Wy_low_diag_used": None if fallback_stage is None else fallback_stage["Wy_low_diag_used"],
            "Wy_high_diag_used": None if fallback_stage is None else fallback_stage["Wy_high_diag_used"],
            "slack_y_inf": None,
            "slack_inf": None,
        })
        if y_min is not None:
            dbg["margin_to_y_min"] = None
            dbg["tight_margin_to_y_min"] = None
        if y_max is not None:
            dbg["margin_to_y_max"] = None
            dbg["tight_margin_to_y_max"] = None
        if return_debug:
            return None, None, d_hat.copy(), dbg
        return None, None, d_hat.copy()

    x_s = final_eval["x_s"]
    u_s = final_eval["u_s"]
    y_s = final_eval["y_s"]
    target_err = final_eval["yc_s"] - y_sp
    target_err_inf = float(np.max(np.abs(target_err)))
    target_err_norm = float(np.linalg.norm(target_err))
    target_slack_2 = float(np.linalg.norm(target_err))
    x_s_aug = np.concatenate([x_s, d_hat])

    final_solve = final_eval["solve_info"]
    final_stage_bundle = exact_stage if final_stage_name == "exact" else fallback_stage

    dbg.update({
        "success": True,
        "status": final_solve["status"],
        "solver": final_solve["solver"],
        "error": final_solve["error"],
        "x_s": x_s.copy(),
        "u_s": u_s.copy(),
        "d_s": d_hat.copy(),
        "x_s_aug": x_s_aug.copy(),
        "y_s": y_s.copy(),
        "objective_value": final_solve["objective_value"],
        "objective": final_solve["objective_value"],
        "target_error": target_err.copy(),
        "target_error_inf": target_err_inf,
        "target_error_norm": target_err_norm,
        "target_slack": target_err.copy(),
        "target_slack_inf": target_err_inf,
        "target_slack_2": target_slack_2,
        "target_eq_residual_inf": final_eval["target_eq_residual_inf"],
        "dyn_residual_inf": final_eval["dyn_residual_inf"],
        "bound_violation_inf": final_eval["bound_violation_inf"],
        "target_move_x_inf": None if x_s_prev is None else float(np.max(np.abs(x_s - x_s_prev))),
        "target_move_u_inf": None if u_s_prev is None else float(np.max(np.abs(u_s - u_s_prev))),
        "margin_to_u_min": (u_s - u_min).copy(),
        "margin_to_u_max": (u_max - u_s).copy(),
        "tight_margin_to_u_min": (u_s - u_lo).copy(),
        "tight_margin_to_u_max": (u_hi - u_s).copy(),
        "s_y_low": final_eval["s_y_low"].copy(),
        "s_y_high": final_eval["s_y_high"].copy(),
        "Wy_low_diag_used": final_stage_bundle["Wy_low_diag_used"],
        "Wy_high_diag_used": final_stage_bundle["Wy_high_diag_used"],
        "slack_y_inf": target_err_inf,
        "slack_inf": target_err_inf,
    })

    if y_min is not None:
        dbg["margin_to_y_min"] = (y_s - y_min).copy()
        dbg["tight_margin_to_y_min"] = (y_s - y_lo).copy()
    if y_max is not None:
        dbg["margin_to_y_max"] = (y_max - y_s).copy()
        dbg["tight_margin_to_y_max"] = (y_hi - y_s).copy()

    if return_debug:
        return x_s, u_s, d_hat.copy(), dbg
    return x_s, u_s, d_hat.copy()


def prepare_filter_target_from_refined_selector(
    A_aug,
    B_aug,
    C_aug,
    xhat_aug,
    y_sp,
    u_min,
    u_max,
    u_nom=None,
    Ty_diag=None,
    Ru_diag=None,
    Qx_diag=None,
    w_x=1e-6,
    prev_target=None,
    x_s_prev=None,
    u_s_prev=None,
    Qdx_diag=None,
    Rdu_diag=None,
    y_min=None,
    y_max=None,
    u_tight=None,
    y_tight=None,
    soft_output_bounds=True,
    Wy_low_diag=None,
    Wy_high_diag=None,
    solver_pref=DEFAULT_CVXPY_SOLVERS,
    warm_start=True,
    return_debug=False,
    H=None,
):
    return prepare_filter_target(
        selector_mode="current_exact_fallback_frozen_d",
        A_aug=A_aug,
        B_aug=B_aug,
        C_aug=C_aug,
        xhat_aug=xhat_aug,
        y_sp=y_sp,
        u_min=u_min,
        u_max=u_max,
        config=TargetSelectorConfig(
            selector_mode="current_exact_fallback_frozen_d",
            u_nom=u_nom,
            Ty_diag=Ty_diag,
            Ru_diag=Ru_diag,
            Qx_diag=Qx_diag,
            w_x=w_x,
            Qdx_diag=Qdx_diag,
            Rdu_diag=Rdu_diag,
            soft_output_bounds=bool(soft_output_bounds),
            Wy_low_diag=Wy_low_diag,
            Wy_high_diag=Wy_high_diag,
            u_tight=u_tight,
            y_tight=y_tight,
            freeze_d_at_estimate=True,
            solver_pref=solver_pref,
        ),
        prev_target=prev_target,
        x_s_prev=x_s_prev,
        u_s_prev=u_s_prev,
        y_min=y_min,
        y_max=y_max,
        warm_start=warm_start,
        return_debug=return_debug,
        H=H,
    )


def _as_config_dict(config):
    if config is None:
        return {}
    if isinstance(config, TargetSelectorConfig):
        return asdict(config)
    if isinstance(config, dict):
        return dict(config)
    raise TypeError("config must be a TargetSelectorConfig, dict, or None.")


def _coerce_mode(selector_mode):
    mode = str(selector_mode).strip()
    if mode not in TARGET_SELECTOR_MODES:
        raise ValueError(
            f"Unsupported selector_mode '{selector_mode}'. Supported modes: {TARGET_SELECTOR_MODES}."
        )
    return mode


def _diag_like(source, size, scale=1.0, default=1.0):
    if source is None:
        return np.full(size, float(scale * default), dtype=float)
    arr = np.asarray(source, float).reshape(-1)
    if arr.size == 1:
        arr = np.full(size, float(arr.item()), dtype=float)
    if arr.size != size:
        raise ValueError(f"Expected vector of size {size}, got {arr.size}.")
    return float(scale) * arr.copy()


def build_target_selector_config(
    selector_mode="current_exact_fallback_frozen_d",
    user_overrides=None,
    n_x=None,
    n_u=None,
    n_y=None,
    n_d=None,
    Q_out=None,
    Rmove_diag=None,
):
    mode = _coerce_mode(selector_mode)
    if None in (n_x, n_u, n_y):
        raise ValueError("n_x, n_u, and n_y must be provided to build_target_selector_config.")
    if n_d is None:
        n_d = n_y

    q_out_diag = _diag_like(Q_out, n_y, scale=1.0, default=1.0)
    rmove_diag = _diag_like(Rmove_diag, n_u, scale=1.0, default=1.0)
    ones_u = np.ones(n_u, dtype=float)
    ones_y = np.ones(n_y, dtype=float)
    ones_x = np.ones(n_x, dtype=float)
    ones_d = np.ones(n_d, dtype=float)

    mode_defaults = {
        "current_exact_fallback_frozen_d": {
            "u_nom": None,
            "Ty_diag": 1e8 * q_out_diag,
            "Ru_diag": ones_u.copy(),
            "Qx_diag": None,
            "w_x": 1e-6,
            "Qdx_diag": None,
            "Rdu_diag": rmove_diag.copy(),
            "soft_output_bounds": True,
            "Wy_low_diag": 1e3 * ones_y.copy(),
            "Wy_high_diag": 1e3 * ones_y.copy(),
            "u_tight": np.zeros(n_u, dtype=float),
            "y_tight": np.zeros(n_y, dtype=float),
            "freeze_d_at_estimate": True,
            "solver_pref": DEFAULT_CVXPY_SOLVERS,
            "accept_statuses": ("optimal", "optimal_inaccurate"),
            "tol_optimal": 1e-6,
            "tol_optimal_inaccurate": 1e-5,
        },
        "free_disturbance_prior": {
            "u_nom": None,
            "Qr_diag": 1e8 * q_out_diag,
            "Ru_diag": ones_u.copy(),
            "Qd_diag": 1e2 * ones_d.copy(),
            "Qdx_diag": 1e-3 * ones_x.copy(),
            "Rdu_diag": rmove_diag.copy(),
            "soft_output_bounds": True,
            "Wy_low_diag": 1e3 * ones_y.copy(),
            "Wy_high_diag": 1e3 * ones_y.copy(),
            "u_tight": np.zeros(n_u, dtype=float),
            "y_tight": np.zeros(n_y, dtype=float),
            "delta_d_inf": None,
            "freeze_d_at_estimate": False,
            "solver_pref": DEFAULT_CVXPY_SOLVERS,
            "accept_statuses": ("optimal", "optimal_inaccurate"),
            "tol_optimal": 1e-6,
            "tol_optimal_inaccurate": 1e-5,
        },
        "compromised_reference": {
            "u_nom": None,
            "Qr_diag": 1e8 * q_out_diag,
            "Ru_diag": ones_u.copy(),
            "Qdx_diag": 1e-3 * ones_x.copy(),
            "Rdu_diag": rmove_diag.copy(),
            "soft_output_bounds": True,
            "Wy_low_diag": 1e3 * ones_y.copy(),
            "Wy_high_diag": 1e3 * ones_y.copy(),
            "u_tight": np.zeros(n_u, dtype=float),
            "y_tight": np.zeros(n_y, dtype=float),
            "freeze_d_at_estimate": True,
            "solver_pref": DEFAULT_CVXPY_SOLVERS,
            "accept_statuses": ("optimal", "optimal_inaccurate"),
            "tol_optimal": 1e-6,
            "tol_optimal_inaccurate": 1e-5,
        },
        "single_stage_robust_sstp": {
            "u_nom": None,
            "Qr_diag": 1e8 * q_out_diag,
            "Ru_diag": ones_u.copy(),
            "Qdx_diag": 1e-3 * ones_x.copy(),
            "Rdu_diag": rmove_diag.copy(),
            "rho_x": 1e5,
            "rho_y": 1e5,
            "soft_output_bounds": True,
            "Wy_low_diag": 1e3 * ones_y.copy(),
            "Wy_high_diag": 1e3 * ones_y.copy(),
            "u_tight": np.zeros(n_u, dtype=float),
            "y_tight": np.zeros(n_y, dtype=float),
            "freeze_d_at_estimate": True,
            "solver_pref": DEFAULT_CVXPY_SOLVERS,
            "accept_statuses": ("optimal", "optimal_inaccurate"),
            "tol_optimal": 1e-6,
            "tol_optimal_inaccurate": 1e-5,
        },
    }

    merged = dict(mode_defaults[mode])
    merged["selector_mode"] = mode
    if user_overrides is not None:
        merged.update(_as_config_dict(user_overrides))
    merged["selector_mode"] = mode
    return TargetSelectorConfig(**merged)


def _resolve_prev_target_vectors(prev_target, x_s_prev=None, u_s_prev=None, d_s_prev=None, r_s_prev=None):
    if isinstance(prev_target, dict):
        if x_s_prev is None and prev_target.get("x_s") is not None:
            x_s_prev = prev_target.get("x_s")
        if u_s_prev is None and prev_target.get("u_s") is not None:
            u_s_prev = prev_target.get("u_s")
        if d_s_prev is None and prev_target.get("d_s") is not None:
            d_s_prev = prev_target.get("d_s")
        if r_s_prev is None and prev_target.get("r_s") is not None:
            r_s_prev = prev_target.get("r_s")
    return x_s_prev, u_s_prev, d_s_prev, r_s_prev


def _selector_context(
    A_aug,
    B_aug,
    C_aug,
    xhat_aug,
    y_sp,
    u_min,
    u_max,
    u_nom=None,
    y_min=None,
    y_max=None,
    u_tight=None,
    y_tight=None,
    H=None,
):
    A_aug = np.asarray(A_aug, float)
    B_aug = np.asarray(B_aug, float)
    C_aug = np.asarray(C_aug, float)
    xhat_aug = np.asarray(xhat_aug, float).reshape(-1)
    y_sp = np.asarray(y_sp, float).reshape(-1)
    u_min = np.asarray(u_min, float).reshape(-1)
    u_max = np.asarray(u_max, float).reshape(-1)

    if A_aug.ndim != 2 or A_aug.shape[0] != A_aug.shape[1]:
        raise ValueError("A_aug must be square.")
    if B_aug.ndim != 2 or B_aug.shape[0] != A_aug.shape[0]:
        raise ValueError("B_aug has incompatible shape.")
    if C_aug.ndim != 2 or C_aug.shape[1] != A_aug.shape[0]:
        raise ValueError("C_aug has incompatible shape.")
    if xhat_aug.size != A_aug.shape[0]:
        raise ValueError("xhat_aug has incorrect size.")

    n_aug = A_aug.shape[0]
    n_y = C_aug.shape[0]
    n_x = n_aug - n_y
    n_d = n_y
    n_u = B_aug.shape[1]
    if n_x <= 0:
        raise ValueError("Invalid augmented model: inferred physical state dimension must be positive.")
    if u_min.size != n_u or u_max.size != n_u:
        raise ValueError("u_min and u_max must both have size n_u.")

    if H is None:
        H_arr = None
        n_c = n_y
    else:
        H_arr = np.asarray(H, float)
        if H_arr.ndim != 2 or H_arr.shape[1] != n_y:
            raise ValueError("H must have shape (n_c, n_y).")
        n_c = H_arr.shape[0]

    if y_sp.size != n_c:
        raise ValueError(f"y_sp has incorrect size. Expected {n_c}, got {y_sp.size}.")

    A = A_aug[:n_x, :n_x]
    Bd = A_aug[:n_x, n_x:]
    B = B_aug[:n_x, :]
    C = C_aug[:, :n_x]
    Cd = C_aug[:, n_x:]
    d_hat = xhat_aug[n_x:].copy()
    if d_hat.size != n_d:
        raise ValueError(
            "This selector assumes the augmented state is ordered as [x; d] with len(d) == n_y."
        )

    if u_nom is None:
        u_nom = np.zeros(n_u, dtype=float)
    else:
        u_nom = np.asarray(u_nom, float).reshape(-1)
        if u_nom.size != n_u:
            raise ValueError("u_nom has incorrect size.")

    u_tight = np.maximum(vector_or_zeros(u_tight, n_u), 0.0)
    y_tight = np.maximum(vector_or_zeros(y_tight, n_y), 0.0)
    u_lo = u_min + u_tight
    u_hi = u_max - u_tight
    if np.any(u_lo > u_hi):
        raise ValueError("Input tightening is too large. Tightened bounds are infeasible.")

    y_lo = None
    y_hi = None
    if y_min is not None:
        y_min = np.asarray(y_min, float).reshape(-1)
        if y_min.size != n_y:
            raise ValueError("y_min has incorrect size.")
        y_lo = y_min + y_tight
    if y_max is not None:
        y_max = np.asarray(y_max, float).reshape(-1)
        if y_max.size != n_y:
            raise ValueError("y_max has incorrect size.")
        y_hi = y_max - y_tight
    if y_lo is not None and y_hi is not None and np.any(y_lo > y_hi):
        raise ValueError("Output tightening is too large. Tightened output bounds are infeasible.")

    return {
        "A": A,
        "B": B,
        "Bd": Bd,
        "C": C,
        "Cd": Cd,
        "d_hat": d_hat,
        "y_sp": y_sp,
        "u_nom": u_nom,
        "u_min": u_min,
        "u_max": u_max,
        "u_lo": u_lo,
        "u_hi": u_hi,
        "u_tight": u_tight,
        "y_min": y_min,
        "y_max": y_max,
        "y_lo": y_lo,
        "y_hi": y_hi,
        "y_tight": y_tight,
        "n_aug": n_aug,
        "n_x": n_x,
        "n_y": n_y,
        "n_d": n_d,
        "n_u": n_u,
        "n_c": n_c,
        "H": H_arr,
        "I_minus_A": np.eye(n_x, dtype=float) - A,
    }


def _build_output_bounds_terms(ctx, y_expr, soft_output_bounds, Wy_low_diag, Wy_high_diag):
    constraints = []
    objective = 0.0
    s_y_low = None
    s_y_high = None
    Wy_low_used = None
    Wy_high_used = None
    n_y = int(ctx["n_y"])

    if ctx["y_lo"] is not None:
        if soft_output_bounds:
            s_y_low = cp.Variable(n_y, nonneg=True)
            Wy_low, Wy_low_used = diag_psd_from_vector(Wy_low_diag, n_y, eps=1e-12, default=1e3)
            objective += cp.quad_form(s_y_low, Wy_low)
            constraints.append(y_expr + s_y_low >= ctx["y_lo"])
        else:
            constraints.append(y_expr >= ctx["y_lo"])

    if ctx["y_hi"] is not None:
        if soft_output_bounds:
            s_y_high = cp.Variable(n_y, nonneg=True)
            Wy_high, Wy_high_used = diag_psd_from_vector(Wy_high_diag, n_y, eps=1e-12, default=1e3)
            objective += cp.quad_form(s_y_high, Wy_high)
            constraints.append(y_expr - s_y_high <= ctx["y_hi"])
        else:
            constraints.append(y_expr <= ctx["y_hi"])

    return {
        "constraints": constraints,
        "objective": objective,
        "s_y_low": s_y_low,
        "s_y_high": s_y_high,
        "Wy_low_diag_used": Wy_low_used,
        "Wy_high_diag_used": Wy_high_used,
    }


def _selector_reference_for_target(y_s, yc_s, r_s, requested_y_sp):
    candidates = [r_s, yc_s, y_s]
    for cand in candidates:
        if cand is None:
            continue
        cand = np.asarray(cand, float).reshape(-1)
        if cand.size == requested_y_sp.size:
            return cand.copy()
    return None


def _assemble_target_info(
    selector_mode,
    ctx,
    dbg,
    x_s,
    u_s,
    d_s,
    y_s,
    yc_s,
    r_s,
    success,
    x_s_prev=None,
    u_s_prev=None,
):
    requested_y_sp = np.asarray(ctx["y_sp"], float).reshape(-1)
    target_reference = _selector_reference_for_target(y_s, yc_s, r_s, requested_y_sp)
    target_error = None if target_reference is None else (target_reference - requested_y_sp)
    target_error_inf = None if target_error is None else float(np.max(np.abs(target_error)))
    target_error_norm = None if target_error is None else float(np.linalg.norm(target_error))
    d_s_minus_dhat = None if d_s is None else np.asarray(d_s, float).reshape(-1) - np.asarray(ctx["d_hat"], float).reshape(-1)
    d_s_minus_dhat_inf = None if d_s_minus_dhat is None else float(np.max(np.abs(d_s_minus_dhat)))
    x_s_aug = None
    if x_s is not None and d_s is not None:
        x_s_aug = np.concatenate([np.asarray(x_s, float).reshape(-1), np.asarray(d_s, float).reshape(-1)])

    return {
        "success": bool(success),
        "selector_mode": str(selector_mode),
        "solve_stage": dbg.get("solve_stage"),
        "x_s": None if x_s is None else np.asarray(x_s, float).reshape(-1),
        "u_s": None if u_s is None else np.asarray(u_s, float).reshape(-1),
        "d_s": None if d_s is None else np.asarray(d_s, float).reshape(-1),
        "x_s_aug": None if x_s_aug is None else x_s_aug.copy(),
        "y_s": None if y_s is None else np.asarray(y_s, float).reshape(-1),
        "yc_s": None if yc_s is None else np.asarray(yc_s, float).reshape(-1),
        "r_s": None if r_s is None else np.asarray(r_s, float).reshape(-1),
        "requested_y_sp": requested_y_sp.copy(),
        "target_error": None if target_error is None else target_error.copy(),
        "target_error_inf": target_error_inf,
        "target_error_norm": target_error_norm,
        "target_slack_inf": target_error_inf,
        "dyn_residual_inf": dbg.get("dyn_residual_inf"),
        "bound_violation_inf": dbg.get("bound_violation_inf"),
        "warm_start": {
            "enabled": bool(dbg.get("warm_start_enabled", False)),
            "available": bool(dbg.get("warm_start_available", False)),
            "used": bool(dbg.get("warm_start_used", False)),
            "x_s_prev": None if x_s_prev is None else np.asarray(x_s_prev, float).reshape(-1).copy(),
            "u_s_prev": None if u_s_prev is None else np.asarray(u_s_prev, float).reshape(-1).copy(),
        },
        "Qdx_diag_used": dbg.get("Qdx_diag_used"),
        "Rdu_diag_used": dbg.get("Rdu_diag_used"),
        "Qd_diag_used": dbg.get("Qd_diag_used"),
        "Qr_diag_used": dbg.get("Qr_diag_used"),
        "d_s_frozen": dbg.get("d_s_frozen"),
        "d_s_optimized": dbg.get("d_s_optimized"),
        "d_s_minus_dhat": None if d_s_minus_dhat is None else d_s_minus_dhat.copy(),
        "d_s_minus_dhat_inf": d_s_minus_dhat_inf,
        "objective_value": dbg.get("objective_value"),
        "selector_debug": dbg,
    }


def _prepare_mode0_refined_selector(
    A_aug,
    B_aug,
    C_aug,
    xhat_aug,
    y_sp,
    u_min,
    u_max,
    u_nom=None,
    Ty_diag=None,
    Ru_diag=None,
    Qx_diag=None,
    w_x=1e-6,
    prev_target=None,
    x_s_prev=None,
    u_s_prev=None,
    Qdx_diag=None,
    Rdu_diag=None,
    y_min=None,
    y_max=None,
    u_tight=None,
    y_tight=None,
    soft_output_bounds=True,
    Wy_low_diag=None,
    Wy_high_diag=None,
    solver_pref=DEFAULT_CVXPY_SOLVERS,
    warm_start=True,
    return_debug=False,
    H=None,
):
    x_s_prev, u_s_prev, _d_s_prev, _r_s_prev = _resolve_prev_target_vectors(
        prev_target,
        x_s_prev=x_s_prev,
        u_s_prev=u_s_prev,
    )

    x_s, u_s, d_s, dbg = compute_ss_target_refined_rawlings(
        A_aug=A_aug,
        B_aug=B_aug,
        C_aug=C_aug,
        xhat_aug=xhat_aug,
        y_sp=y_sp,
        u_min=u_min,
        u_max=u_max,
        u_nom=u_nom,
        Ty_diag=Ty_diag,
        Ru_diag=Ru_diag,
        Qx_diag=Qx_diag,
        w_x=w_x,
        x_s_prev=x_s_prev,
        u_s_prev=u_s_prev,
        Qdx_diag=Qdx_diag,
        Rdu_diag=Rdu_diag,
        y_min=y_min,
        y_max=y_max,
        u_tight=u_tight,
        y_tight=y_tight,
        soft_output_bounds=soft_output_bounds,
        Wy_low_diag=Wy_low_diag,
        Wy_high_diag=Wy_high_diag,
        solver_pref=solver_pref,
        warm_start=warm_start,
        return_debug=True,
        H=H,
    )

    ctx = _selector_context(
        A_aug=A_aug,
        B_aug=B_aug,
        C_aug=C_aug,
        xhat_aug=xhat_aug,
        y_sp=y_sp,
        u_min=u_min,
        u_max=u_max,
        u_nom=u_nom,
        y_min=y_min,
        y_max=y_max,
        u_tight=u_tight,
        y_tight=y_tight,
        H=H,
    )
    y_s_dbg = None if dbg.get("y_s") is None else np.asarray(dbg["y_s"], float).reshape(-1)
    yc_s_dbg = None if y_s_dbg is None else (y_s_dbg.copy() if ctx["H"] is None else ctx["H"] @ y_s_dbg)
    dbg = dict(dbg)
    dbg.update({
        "selector_mode": "current_exact_fallback_frozen_d",
        "dhat_used": np.asarray(ctx["d_hat"], float).reshape(-1).copy(),
        "d_s_frozen": True,
        "d_s_optimized": False,
        "Qd_diag_used": None,
        "Qr_diag_used": None,
        "r_s": None if yc_s_dbg is None else np.asarray(yc_s_dbg, float).reshape(-1).copy(),
    })
    target_info = _assemble_target_info(
        selector_mode="current_exact_fallback_frozen_d",
        ctx=ctx,
        dbg=dbg,
        x_s=x_s,
        u_s=u_s,
        d_s=d_s,
        y_s=y_s_dbg,
        yc_s=yc_s_dbg,
        r_s=yc_s_dbg,
        success=bool(dbg.get("success", False) and x_s is not None and u_s is not None),
        x_s_prev=x_s_prev,
        u_s_prev=u_s_prev,
    )
    if return_debug:
        return target_info, dbg
    return target_info


def _evaluate_single_stage_solution(
    ctx,
    solve_info,
    x_var,
    u_var,
    d_value,
    y_expr,
    yc_expr,
    r_expr,
    s_y_low,
    s_y_high,
    tol,
    allow_dyn_slack=False,
    dyn_slack_value=None,
):
    result = {
        "accepted": False,
        "reject_reason": None,
        "x_s": None,
        "u_s": None,
        "d_s": None,
        "y_s": None,
        "yc_s": None,
        "r_s": None,
        "dyn_residual_inf": None,
        "bound_violation_inf": None,
        "s_y_low": np.zeros(ctx["n_y"], dtype=float),
        "s_y_high": np.zeros(ctx["n_y"], dtype=float),
        "solve_info": solve_info,
    }
    if not solve_info["accepted_by_status"]:
        result["reject_reason"] = "solver_status"
        return result
    if any(var is None or var.value is None for var in (x_var, u_var)):
        result["reject_reason"] = "missing_solution"
        return result

    x_s = np.asarray(x_var.value, float).reshape(-1)
    u_s = np.asarray(u_var.value, float).reshape(-1)
    d_s = np.asarray(d_value() if callable(d_value) else d_value, float).reshape(-1)
    y_s = np.asarray(y_expr.value if hasattr(y_expr, "value") and y_expr.value is not None else ctx["C"] @ x_s + ctx["Cd"] @ d_s, float).reshape(-1)
    yc_s = np.asarray(yc_expr.value if hasattr(yc_expr, "value") and yc_expr.value is not None else (y_s if ctx["H"] is None else ctx["H"] @ y_s), float).reshape(-1)
    r_s = np.asarray(r_expr.value if hasattr(r_expr, "value") and r_expr.value is not None else yc_s, float).reshape(-1)
    dyn_residual = ctx["I_minus_A"] @ x_s - ctx["B"] @ u_s - ctx["Bd"] @ d_s
    dyn_residual_inf = float(np.max(np.abs(dyn_residual)))
    if dyn_slack_value is not None:
        dyn_residual_inf = float(max(dyn_residual_inf, np.max(np.abs(np.asarray(dyn_slack_value, float).reshape(-1)))))

    bound_violation_inf = _bound_violation_inf(u_s, lower=ctx["u_lo"], upper=ctx["u_hi"])
    if ctx["y_lo"] is not None or ctx["y_hi"] is not None:
        bound_violation_inf = max(bound_violation_inf, _bound_violation_inf(y_s, lower=ctx["y_lo"], upper=ctx["y_hi"]))

    if not allow_dyn_slack and dyn_residual_inf > tol:
        result["reject_reason"] = "dyn_residual"
    elif bound_violation_inf > tol:
        result["reject_reason"] = "bound_violation"
    else:
        result["accepted"] = True

    result.update({
        "x_s": x_s,
        "u_s": u_s,
        "d_s": d_s,
        "y_s": y_s,
        "yc_s": yc_s,
        "r_s": r_s,
        "dyn_residual_inf": dyn_residual_inf,
        "bound_violation_inf": float(bound_violation_inf),
        "s_y_low": _output_slack_value(s_y_low, ctx["n_y"]),
        "s_y_high": _output_slack_value(s_y_high, ctx["n_y"]),
    })
    return result


def _compute_target_mode2_compromised_reference(ctx, cfg, prev_target, x_s_prev, u_s_prev, warm_start):
    if not HAS_CVXPY:
        raise ImportError("CVXPY is required for the target selector.")

    x_s_prev, u_s_prev, _d_s_prev, _r_s_prev = _resolve_prev_target_vectors(prev_target, x_s_prev=x_s_prev, u_s_prev=u_s_prev)
    Qr, Qr_used = diag_psd_from_vector(cfg.get("Qr_diag"), ctx["n_c"], eps=1e-12, default=1.0)
    Ru, Ru_used = diag_psd_from_vector(cfg.get("Ru_diag"), ctx["n_u"], eps=1e-12, default=1.0)
    Qdx = None
    Qdx_used = None
    if x_s_prev is not None and cfg.get("Qdx_diag") is not None:
        Qdx, Qdx_used = diag_psd_from_vector(cfg.get("Qdx_diag"), ctx["n_x"], eps=1e-12, default=0.0)
    Rdu = None
    Rdu_used = None
    if u_s_prev is not None and cfg.get("Rdu_diag") is not None:
        Rdu, Rdu_used = diag_psd_from_vector(cfg.get("Rdu_diag"), ctx["n_u"], eps=1e-12, default=0.0)

    x = cp.Variable(ctx["n_x"])
    u = cp.Variable(ctx["n_u"])
    d_s = ctx["d_hat"].copy()
    y_expr = ctx["C"] @ x + ctx["Cd"] @ d_s
    yc_expr = y_expr if ctx["H"] is None else ctx["H"] @ y_expr
    r_expr = yc_expr
    objective = cp.quad_form(r_expr - ctx["y_sp"], Qr) + cp.quad_form(u - ctx["u_nom"], Ru)
    if Qdx is not None:
        objective += cp.quad_form(x - np.asarray(x_s_prev, float).reshape(-1), Qdx)
    if Rdu is not None:
        objective += cp.quad_form(u - np.asarray(u_s_prev, float).reshape(-1), Rdu)
    bound_terms = _build_output_bounds_terms(ctx, y_expr, bool(cfg.get("soft_output_bounds", True)), cfg.get("Wy_low_diag"), cfg.get("Wy_high_diag"))
    objective += bound_terms["objective"]
    constraints = [
        ctx["I_minus_A"] @ x - ctx["B"] @ u - ctx["Bd"] @ d_s == 0.0,
        u >= ctx["u_lo"],
        u <= ctx["u_hi"],
    ] + bound_terms["constraints"]
    problem = cp.Problem(cp.Minimize(objective), constraints)
    warm_start_enabled = bool(warm_start)
    warm_start_available = bool(x_s_prev is not None or u_s_prev is not None)
    solve_info = _solve_problem_with_preferences(
        problem,
        [x, u, bound_terms["s_y_low"], bound_terms["s_y_high"]],
        cfg.get("solver_pref"),
        warm_start=warm_start_enabled and warm_start_available,
        initial_values=[
            None if x_s_prev is None else np.asarray(x_s_prev, float).reshape(-1).copy(),
            None if u_s_prev is None else np.asarray(u_s_prev, float).reshape(-1).copy(),
            None if bound_terms["s_y_low"] is None else np.zeros(ctx["n_y"], dtype=float),
            None if bound_terms["s_y_high"] is None else np.zeros(ctx["n_y"], dtype=float),
        ],
    )
    tol = float(cfg.get("tol_optimal", 1e-6)) if solve_info["status"] == "optimal" else float(cfg.get("tol_optimal_inaccurate", 1e-5))
    eval_info = _evaluate_single_stage_solution(
        ctx=ctx,
        solve_info=solve_info,
        x_var=x,
        u_var=u,
        d_value=d_s,
        y_expr=y_expr,
        yc_expr=yc_expr,
        r_expr=r_expr,
        s_y_low=bound_terms["s_y_low"],
        s_y_high=bound_terms["s_y_high"],
        tol=tol,
    )
    dbg = {
        "success": bool(eval_info["accepted"]),
        "selector_mode": "compromised_reference",
        "status": solve_info["status"],
        "solver": solve_info["solver"],
        "error": solve_info["error"],
        "objective_value": solve_info["objective_value"],
        "solve_stage": "compromised_reference",
        "warm_start_enabled": warm_start_enabled,
        "warm_start_available": warm_start_available,
        "warm_start_used": bool(warm_start_enabled and warm_start_available),
        "state_smoothing_active": bool(Qdx is not None),
        "input_smoothing_active": bool(Rdu is not None),
        "Qdx_diag_used": Qdx_used,
        "Rdu_diag_used": Rdu_used,
        "Qr_diag_used": Qr_used,
        "Ru_diag_used": Ru_used,
        "Qd_diag_used": None,
        "dhat_used": ctx["d_hat"].copy(),
        "d_s_frozen": True,
        "d_s_optimized": False,
        "dyn_residual_inf": eval_info["dyn_residual_inf"],
        "bound_violation_inf": eval_info["bound_violation_inf"],
        "reject_reason": eval_info["reject_reason"],
        "target_move_x_inf": None if x_s_prev is None or eval_info["x_s"] is None else float(np.max(np.abs(eval_info["x_s"] - np.asarray(x_s_prev, float).reshape(-1)))),
        "target_move_u_inf": None if u_s_prev is None or eval_info["u_s"] is None else float(np.max(np.abs(eval_info["u_s"] - np.asarray(u_s_prev, float).reshape(-1)))),
        "s_y_low": eval_info["s_y_low"].copy(),
        "s_y_high": eval_info["s_y_high"].copy(),
        "Wy_low_diag_used": bound_terms["Wy_low_diag_used"],
        "Wy_high_diag_used": bound_terms["Wy_high_diag_used"],
    }
    target_info = _assemble_target_info(
        selector_mode="compromised_reference",
        ctx=ctx,
        dbg=dbg,
        x_s=eval_info["x_s"],
        u_s=eval_info["u_s"],
        d_s=eval_info["d_s"],
        y_s=eval_info["y_s"],
        yc_s=eval_info["yc_s"],
        r_s=eval_info["r_s"],
        success=eval_info["accepted"],
        x_s_prev=x_s_prev,
        u_s_prev=u_s_prev,
    )
    return target_info, dbg


def _compute_target_mode1_free_disturbance_prior(ctx, cfg, prev_target, x_s_prev, u_s_prev, warm_start):
    if not HAS_CVXPY:
        raise ImportError("CVXPY is required for the target selector.")

    x_s_prev, u_s_prev, d_s_prev, _r_s_prev = _resolve_prev_target_vectors(prev_target, x_s_prev=x_s_prev, u_s_prev=u_s_prev)
    Qr, Qr_used = diag_psd_from_vector(cfg.get("Qr_diag"), ctx["n_c"], eps=1e-12, default=1.0)
    Ru, Ru_used = diag_psd_from_vector(cfg.get("Ru_diag"), ctx["n_u"], eps=1e-12, default=1.0)
    Qd, Qd_used = diag_psd_from_vector(cfg.get("Qd_diag"), ctx["n_d"], eps=1e-12, default=1e2)
    Qdx = None
    Qdx_used = None
    if x_s_prev is not None and cfg.get("Qdx_diag") is not None:
        Qdx, Qdx_used = diag_psd_from_vector(cfg.get("Qdx_diag"), ctx["n_x"], eps=1e-12, default=0.0)
    Rdu = None
    Rdu_used = None
    if u_s_prev is not None and cfg.get("Rdu_diag") is not None:
        Rdu, Rdu_used = diag_psd_from_vector(cfg.get("Rdu_diag"), ctx["n_u"], eps=1e-12, default=0.0)

    x = cp.Variable(ctx["n_x"])
    u = cp.Variable(ctx["n_u"])
    d = cp.Variable(ctx["n_d"])
    y_expr = ctx["C"] @ x + ctx["Cd"] @ d
    yc_expr = y_expr if ctx["H"] is None else ctx["H"] @ y_expr
    r_expr = yc_expr
    objective = cp.quad_form(r_expr - ctx["y_sp"], Qr) + cp.quad_form(u - ctx["u_nom"], Ru) + cp.quad_form(d - ctx["d_hat"], Qd)
    if Qdx is not None:
        objective += cp.quad_form(x - np.asarray(x_s_prev, float).reshape(-1), Qdx)
    if Rdu is not None:
        objective += cp.quad_form(u - np.asarray(u_s_prev, float).reshape(-1), Rdu)
    bound_terms = _build_output_bounds_terms(ctx, y_expr, bool(cfg.get("soft_output_bounds", True)), cfg.get("Wy_low_diag"), cfg.get("Wy_high_diag"))
    objective += bound_terms["objective"]
    constraints = [
        ctx["I_minus_A"] @ x - ctx["B"] @ u - ctx["Bd"] @ d == 0.0,
        u >= ctx["u_lo"],
        u <= ctx["u_hi"],
    ] + bound_terms["constraints"]
    delta_d_inf = cfg.get("delta_d_inf")
    if delta_d_inf is not None:
        constraints.extend([
            d - ctx["d_hat"] <= float(delta_d_inf),
            ctx["d_hat"] - d <= float(delta_d_inf),
        ])
    problem = cp.Problem(cp.Minimize(objective), constraints)
    warm_start_enabled = bool(warm_start)
    warm_start_available = bool(x_s_prev is not None or u_s_prev is not None or d_s_prev is not None)
    solve_info = _solve_problem_with_preferences(
        problem,
        [x, u, d, bound_terms["s_y_low"], bound_terms["s_y_high"]],
        cfg.get("solver_pref"),
        warm_start=warm_start_enabled and warm_start_available,
        initial_values=[
            None if x_s_prev is None else np.asarray(x_s_prev, float).reshape(-1).copy(),
            None if u_s_prev is None else np.asarray(u_s_prev, float).reshape(-1).copy(),
            None if d_s_prev is None else np.asarray(d_s_prev, float).reshape(-1).copy(),
            None if bound_terms["s_y_low"] is None else np.zeros(ctx["n_y"], dtype=float),
            None if bound_terms["s_y_high"] is None else np.zeros(ctx["n_y"], dtype=float),
        ],
    )
    tol = float(cfg.get("tol_optimal", 1e-6)) if solve_info["status"] == "optimal" else float(cfg.get("tol_optimal_inaccurate", 1e-5))
    eval_info = _evaluate_single_stage_solution(
        ctx=ctx,
        solve_info=solve_info,
        x_var=x,
        u_var=u,
        d_value=lambda: d.value,
        y_expr=y_expr,
        yc_expr=yc_expr,
        r_expr=r_expr,
        s_y_low=bound_terms["s_y_low"],
        s_y_high=bound_terms["s_y_high"],
        tol=tol,
    )
    dbg = {
        "success": bool(eval_info["accepted"]),
        "selector_mode": "free_disturbance_prior",
        "status": solve_info["status"],
        "solver": solve_info["solver"],
        "error": solve_info["error"],
        "objective_value": solve_info["objective_value"],
        "solve_stage": "free_disturbance_prior",
        "warm_start_enabled": warm_start_enabled,
        "warm_start_available": warm_start_available,
        "warm_start_used": bool(warm_start_enabled and warm_start_available),
        "state_smoothing_active": bool(Qdx is not None),
        "input_smoothing_active": bool(Rdu is not None),
        "Qdx_diag_used": Qdx_used,
        "Rdu_diag_used": Rdu_used,
        "Qr_diag_used": Qr_used,
        "Ru_diag_used": Ru_used,
        "Qd_diag_used": Qd_used,
        "dhat_used": ctx["d_hat"].copy(),
        "d_s_frozen": False,
        "d_s_optimized": True,
        "delta_d_inf": delta_d_inf,
        "dyn_residual_inf": eval_info["dyn_residual_inf"],
        "bound_violation_inf": eval_info["bound_violation_inf"],
        "reject_reason": eval_info["reject_reason"],
        "target_move_x_inf": None if x_s_prev is None or eval_info["x_s"] is None else float(np.max(np.abs(eval_info["x_s"] - np.asarray(x_s_prev, float).reshape(-1)))),
        "target_move_u_inf": None if u_s_prev is None or eval_info["u_s"] is None else float(np.max(np.abs(eval_info["u_s"] - np.asarray(u_s_prev, float).reshape(-1)))),
        "s_y_low": eval_info["s_y_low"].copy(),
        "s_y_high": eval_info["s_y_high"].copy(),
        "Wy_low_diag_used": bound_terms["Wy_low_diag_used"],
        "Wy_high_diag_used": bound_terms["Wy_high_diag_used"],
    }
    target_info = _assemble_target_info(
        selector_mode="free_disturbance_prior",
        ctx=ctx,
        dbg=dbg,
        x_s=eval_info["x_s"],
        u_s=eval_info["u_s"],
        d_s=eval_info["d_s"],
        y_s=eval_info["y_s"],
        yc_s=eval_info["yc_s"],
        r_s=eval_info["r_s"],
        success=eval_info["accepted"],
        x_s_prev=x_s_prev,
        u_s_prev=u_s_prev,
    )
    return target_info, dbg


def _compute_target_mode3_single_stage_robust_sstp(ctx, cfg, prev_target, x_s_prev, u_s_prev, warm_start):
    if not HAS_CVXPY:
        raise ImportError("CVXPY is required for the target selector.")

    x_s_prev, u_s_prev, _d_s_prev, _r_s_prev = _resolve_prev_target_vectors(prev_target, x_s_prev=x_s_prev, u_s_prev=u_s_prev)
    Qr, Qr_used = diag_psd_from_vector(cfg.get("Qr_diag"), ctx["n_c"], eps=1e-12, default=1.0)
    Ru, Ru_used = diag_psd_from_vector(cfg.get("Ru_diag"), ctx["n_u"], eps=1e-12, default=1.0)
    Qdx = None
    Qdx_used = None
    if x_s_prev is not None and cfg.get("Qdx_diag") is not None:
        Qdx, Qdx_used = diag_psd_from_vector(cfg.get("Qdx_diag"), ctx["n_x"], eps=1e-12, default=0.0)
    Rdu = None
    Rdu_used = None
    if u_s_prev is not None and cfg.get("Rdu_diag") is not None:
        Rdu, Rdu_used = diag_psd_from_vector(cfg.get("Rdu_diag"), ctx["n_u"], eps=1e-12, default=0.0)

    x = cp.Variable(ctx["n_x"])
    u = cp.Variable(ctx["n_u"])
    eps_x = cp.Variable(ctx["n_x"])
    eps_y = cp.Variable(ctx["n_c"])
    d_s = ctx["d_hat"].copy()
    y_expr = ctx["C"] @ x + ctx["Cd"] @ d_s
    yc_expr = y_expr if ctx["H"] is None else ctx["H"] @ y_expr
    r_expr = yc_expr + eps_y
    objective = cp.quad_form(r_expr - ctx["y_sp"], Qr) + cp.quad_form(u - ctx["u_nom"], Ru)
    if Qdx is not None:
        objective += cp.quad_form(x - np.asarray(x_s_prev, float).reshape(-1), Qdx)
    if Rdu is not None:
        objective += cp.quad_form(u - np.asarray(u_s_prev, float).reshape(-1), Rdu)
    objective += float(cfg.get("rho_x", 1e5)) * cp.norm1(eps_x)
    objective += float(cfg.get("rho_y", 1e5)) * cp.norm1(eps_y)
    bound_terms = _build_output_bounds_terms(ctx, y_expr, bool(cfg.get("soft_output_bounds", True)), cfg.get("Wy_low_diag"), cfg.get("Wy_high_diag"))
    objective += bound_terms["objective"]
    constraints = [
        ctx["I_minus_A"] @ x - ctx["B"] @ u - ctx["Bd"] @ d_s == eps_x,
        u >= ctx["u_lo"],
        u <= ctx["u_hi"],
    ] + bound_terms["constraints"]
    problem = cp.Problem(cp.Minimize(objective), constraints)
    warm_start_enabled = bool(warm_start)
    warm_start_available = bool(x_s_prev is not None or u_s_prev is not None)
    solve_info = _solve_problem_with_preferences(
        problem,
        [x, u, eps_x, eps_y, bound_terms["s_y_low"], bound_terms["s_y_high"]],
        cfg.get("solver_pref"),
        warm_start=warm_start_enabled and warm_start_available,
        initial_values=[
            None if x_s_prev is None else np.asarray(x_s_prev, float).reshape(-1).copy(),
            None if u_s_prev is None else np.asarray(u_s_prev, float).reshape(-1).copy(),
            np.zeros(ctx["n_x"], dtype=float),
            np.zeros(ctx["n_c"], dtype=float),
            None if bound_terms["s_y_low"] is None else np.zeros(ctx["n_y"], dtype=float),
            None if bound_terms["s_y_high"] is None else np.zeros(ctx["n_y"], dtype=float),
        ],
    )
    tol = float(cfg.get("tol_optimal", 1e-6)) if solve_info["status"] == "optimal" else float(cfg.get("tol_optimal_inaccurate", 1e-5))
    eval_info = _evaluate_single_stage_solution(
        ctx=ctx,
        solve_info=solve_info,
        x_var=x,
        u_var=u,
        d_value=d_s,
        y_expr=y_expr,
        yc_expr=yc_expr,
        r_expr=r_expr,
        s_y_low=bound_terms["s_y_low"],
        s_y_high=bound_terms["s_y_high"],
        tol=tol,
        allow_dyn_slack=True,
        dyn_slack_value=None if eps_x.value is None else np.asarray(eps_x.value, float).reshape(-1),
    )
    dbg = {
        "success": bool(eval_info["accepted"]),
        "selector_mode": "single_stage_robust_sstp",
        "status": solve_info["status"],
        "solver": solve_info["solver"],
        "error": solve_info["error"],
        "objective_value": solve_info["objective_value"],
        "solve_stage": "single_stage_robust_sstp",
        "warm_start_enabled": warm_start_enabled,
        "warm_start_available": warm_start_available,
        "warm_start_used": bool(warm_start_enabled and warm_start_available),
        "state_smoothing_active": bool(Qdx is not None),
        "input_smoothing_active": bool(Rdu is not None),
        "Qdx_diag_used": Qdx_used,
        "Rdu_diag_used": Rdu_used,
        "Qr_diag_used": Qr_used,
        "Ru_diag_used": Ru_used,
        "Qd_diag_used": None,
        "dhat_used": ctx["d_hat"].copy(),
        "d_s_frozen": True,
        "d_s_optimized": False,
        "rho_x": float(cfg.get("rho_x", 1e5)),
        "rho_y": float(cfg.get("rho_y", 1e5)),
        "eps_x_inf": None if eps_x.value is None else float(np.max(np.abs(np.asarray(eps_x.value, float).reshape(-1)))),
        "eps_y_inf": None if eps_y.value is None else float(np.max(np.abs(np.asarray(eps_y.value, float).reshape(-1)))),
        "dyn_residual_inf": eval_info["dyn_residual_inf"],
        "bound_violation_inf": eval_info["bound_violation_inf"],
        "reject_reason": eval_info["reject_reason"],
        "target_move_x_inf": None if x_s_prev is None or eval_info["x_s"] is None else float(np.max(np.abs(eval_info["x_s"] - np.asarray(x_s_prev, float).reshape(-1)))),
        "target_move_u_inf": None if u_s_prev is None or eval_info["u_s"] is None else float(np.max(np.abs(eval_info["u_s"] - np.asarray(u_s_prev, float).reshape(-1)))),
        "s_y_low": eval_info["s_y_low"].copy(),
        "s_y_high": eval_info["s_y_high"].copy(),
        "Wy_low_diag_used": bound_terms["Wy_low_diag_used"],
        "Wy_high_diag_used": bound_terms["Wy_high_diag_used"],
    }
    target_info = _assemble_target_info(
        selector_mode="single_stage_robust_sstp",
        ctx=ctx,
        dbg=dbg,
        x_s=eval_info["x_s"],
        u_s=eval_info["u_s"],
        d_s=eval_info["d_s"],
        y_s=eval_info["y_s"],
        yc_s=eval_info["yc_s"],
        r_s=eval_info["r_s"],
        success=eval_info["accepted"],
        x_s_prev=x_s_prev,
        u_s_prev=u_s_prev,
    )
    return target_info, dbg


def prepare_filter_target(
    selector_mode="current_exact_fallback_frozen_d",
    A_aug=None,
    B_aug=None,
    C_aug=None,
    xhat_aug=None,
    y_sp=None,
    u_min=None,
    u_max=None,
    config=None,
    prev_target=None,
    H=None,
    return_debug=False,
    warm_start=True,
    x_s_prev=None,
    u_s_prev=None,
    y_min=None,
    y_max=None,
):
    config_dict = _as_config_dict(config)
    ctx = _selector_context(
        A_aug=A_aug,
        B_aug=B_aug,
        C_aug=C_aug,
        xhat_aug=xhat_aug,
        y_sp=y_sp,
        u_min=u_min,
        u_max=u_max,
        u_nom=config_dict.get("u_nom"),
        y_min=y_min,
        y_max=y_max,
        u_tight=config_dict.get("u_tight"),
        y_tight=config_dict.get("y_tight"),
        H=H,
    )
    mode = _coerce_mode(config_dict.get("selector_mode", selector_mode))
    cfg = build_target_selector_config(
        selector_mode=mode,
        user_overrides=config_dict,
        n_x=ctx["n_x"],
        n_u=ctx["n_u"],
        n_y=ctx["n_y"],
        n_d=ctx["n_d"],
        Q_out=np.ones(ctx["n_c"], dtype=float),
        Rmove_diag=np.ones(ctx["n_u"], dtype=float),
    )
    cfg_dict = _as_config_dict(cfg)
    ctx = _selector_context(
        A_aug=A_aug,
        B_aug=B_aug,
        C_aug=C_aug,
        xhat_aug=xhat_aug,
        y_sp=y_sp,
        u_min=u_min,
        u_max=u_max,
        u_nom=cfg_dict.get("u_nom"),
        y_min=y_min,
        y_max=y_max,
        u_tight=cfg_dict.get("u_tight"),
        y_tight=cfg_dict.get("y_tight"),
        H=H,
    )

    if mode == "current_exact_fallback_frozen_d":
        target_info, dbg = _prepare_mode0_refined_selector(
            A_aug=A_aug,
            B_aug=B_aug,
            C_aug=C_aug,
            xhat_aug=xhat_aug,
            y_sp=y_sp,
            u_min=u_min,
            u_max=u_max,
            u_nom=cfg_dict.get("u_nom"),
            Ty_diag=cfg_dict.get("Ty_diag"),
            Ru_diag=cfg_dict.get("Ru_diag"),
            Qx_diag=cfg_dict.get("Qx_diag"),
            w_x=cfg_dict.get("w_x", 1e-6),
            prev_target=prev_target,
            x_s_prev=x_s_prev,
            u_s_prev=u_s_prev,
            Qdx_diag=cfg_dict.get("Qdx_diag"),
            Rdu_diag=cfg_dict.get("Rdu_diag"),
            y_min=y_min,
            y_max=y_max,
            u_tight=cfg_dict.get("u_tight"),
            y_tight=cfg_dict.get("y_tight"),
            soft_output_bounds=cfg_dict.get("soft_output_bounds", True),
            Wy_low_diag=cfg_dict.get("Wy_low_diag"),
            Wy_high_diag=cfg_dict.get("Wy_high_diag"),
            solver_pref=cfg_dict.get("solver_pref"),
            warm_start=warm_start,
            return_debug=True,
            H=H,
        )
    elif mode == "compromised_reference":
        target_info, dbg = _compute_target_mode2_compromised_reference(ctx, cfg_dict, prev_target, x_s_prev, u_s_prev, warm_start)
    elif mode == "free_disturbance_prior":
        target_info, dbg = _compute_target_mode1_free_disturbance_prior(ctx, cfg_dict, prev_target, x_s_prev, u_s_prev, warm_start)
    elif mode == "single_stage_robust_sstp":
        target_info, dbg = _compute_target_mode3_single_stage_robust_sstp(ctx, cfg_dict, prev_target, x_s_prev, u_s_prev, warm_start)
    else:
        raise ValueError(f"Unsupported selector_mode '{mode}'.")

    if return_debug:
        return target_info, dbg
    return target_info
