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


def _solve_problem_with_preferences(problem, variables, solver_pref):
    last_status = None
    last_solver = None
    last_err = None

    for solver_name in _solver_sequence(solver_pref):
        try:
            _reset_variable_values(variables)
            problem.solve(solver=solver_name, warm_start=False, verbose=False)
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

    exact_stage = build_stage_problem("exact")
    exact_solve = _solve_problem_with_preferences(
        exact_stage["problem"],
        [exact_stage["x"], exact_stage["u"], exact_stage["s_y_low"], exact_stage["s_y_high"]],
        solver_pref,
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
    return_debug=False,
    H=None,
):
    if prev_target is not None:
        if x_s_prev is None and prev_target.get("x_s") is not None:
            x_s_prev = prev_target["x_s"]
        if u_s_prev is None and prev_target.get("u_s") is not None:
            u_s_prev = prev_target["u_s"]

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
        return_debug=True,
        H=H,
    )

    requested_y_sp = np.asarray(y_sp, float).reshape(-1)
    y_s_dbg = None if dbg.get("y_s") is None else np.asarray(dbg["y_s"], float).reshape(-1)
    if H is None or y_s_dbg is None:
        yc_s_dbg = None if y_s_dbg is None else y_s_dbg.copy()
    else:
        yc_s_dbg = np.asarray(H, float) @ y_s_dbg

    target_info = {
        "success": bool(dbg.get("success", False) and x_s is not None and u_s is not None),
        "x_s": None if x_s is None else np.asarray(x_s, float).reshape(-1),
        "u_s": None if u_s is None else np.asarray(u_s, float).reshape(-1),
        "d_s": None if d_s is None else np.asarray(d_s, float).reshape(-1),
        "x_s_aug": None if dbg.get("x_s_aug") is None else np.asarray(dbg["x_s_aug"], float).reshape(-1),
        "y_s": y_s_dbg,
        "yc_s": yc_s_dbg,
        "requested_y_sp": requested_y_sp.copy(),
        "solve_stage": dbg.get("solve_stage"),
        "target_error": None if dbg.get("target_error") is None else np.asarray(dbg["target_error"], float).reshape(-1),
        "target_error_inf": dbg.get("target_error_inf"),
        "target_error_norm": dbg.get("target_error_norm"),
        "target_slack_inf": dbg.get("target_slack_inf"),
        "dyn_residual_inf": dbg.get("dyn_residual_inf"),
        "bound_violation_inf": dbg.get("bound_violation_inf"),
        "warm_start": {
            "x_s_prev": None if x_s is None else np.asarray(x_s, float).reshape(-1).copy(),
            "u_s_prev": None if u_s is None else np.asarray(u_s, float).reshape(-1).copy(),
        },
        "selector_debug": dbg,
    }

    if return_debug:
        return target_info, dbg
    return target_info
