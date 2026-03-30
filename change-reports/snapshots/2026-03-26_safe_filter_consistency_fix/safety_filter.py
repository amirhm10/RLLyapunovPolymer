import numpy as np

try:
    import cvxpy as cp

    HAS_CVXPY = True
except Exception:
    HAS_CVXPY = False

from Lyapunov.lyapunov_core import evaluate_candidate_action, lyapunov_bound
from Lyapunov.upstream_controllers import solve_offset_free_mpc_candidate
from utils.lyapunov_utils import diag_psd_from_vector, safety_filter_solver_sequence


_OPTIMAL_STATUSES = {"optimal", "optimal_inaccurate"}


def _as_1d(name, value, expected_size=None):
    arr = np.asarray(value, float).reshape(-1)
    if expected_size is not None and arr.size != expected_size:
        raise ValueError(f"{name} has size {arr.size}, expected {expected_size}.")
    return arr


def _weight_diag(diag_vals, size, default):
    if diag_vals is not None:
        diag_vals = np.asarray(diag_vals, float).reshape(-1)
        if diag_vals.size == 1:
            diag_vals = np.full(size, float(diag_vals.item()), dtype=float)
    mat, diag_used = diag_psd_from_vector(diag_vals, size, eps=1e-12, default=default)
    return mat, diag_used


def _maybe_vector(values, size):
    if values is None:
        return None
    values = np.asarray(values, float).reshape(-1)
    if values.size == 1:
        return np.full(size, float(values.item()), dtype=float)
    return _as_1d("vector", values, expected_size=size)


def _qcqp_output_target(target_info, lyap_config, n_y):
    y_target = lyap_config.get("tracking_output_target")
    if y_target is not None:
        return _as_1d("tracking_output_target", y_target, expected_size=n_y)
    if target_info is not None and target_info.get("y_s") is not None:
        return np.asarray(target_info["y_s"], float).reshape(-1)
    return None


def _postcheck_action(u_try, xhat_aug, target_info, model_info, lyap_config, bounds_info, u_prev):
    tol = float(lyap_config.get("tol", 1e-9))
    return evaluate_candidate_action(
        u_cand=u_try,
        xhat_aug=xhat_aug,
        target_info=target_info,
        ingredients=model_info,
        rho=float(lyap_config.get("rho", 0.99)),
        eps_lyap=float(lyap_config.get("eps_lyap", 1e-9)),
        u_min=bounds_info.get("u_min"),
        u_max=bounds_info.get("u_max"),
        u_prev=u_prev,
        du_min=bounds_info.get("du_min"),
        du_max=bounds_info.get("du_max"),
        tol=tol,
    )


def _final_lyap_target_info(target_info, lyap_config):
    if target_info is not None and target_info.get("success", False):
        return target_info, "current_target"

    diagnostic_target = lyap_config.get("final_lyap_target_info")
    if isinstance(diagnostic_target, dict) and diagnostic_target.get("success", False):
        return diagnostic_target, str(lyap_config.get("final_lyap_target_source", "last_valid_target"))

    return None, None


def _attach_final_lyap(
    base_debug,
    u_try,
    xhat_aug,
    target_info,
    model_info,
    lyap_config,
    bounds_info,
    u_prev,
):
    base_debug.update({
        "final_lyap_value": None,
        "final_lyap_margin": None,
        "final_lyap_ok": None,
        "final_lyap_bound": None,
        "final_y_next_pred": None,
        "final_lyap_target_source": None,
    })

    if u_try is None:
        return base_debug

    lyap_target, lyap_target_source = _final_lyap_target_info(target_info, lyap_config)
    if lyap_target is None:
        return base_debug

    post = _postcheck_action(
        u_try=u_try,
        xhat_aug=xhat_aug,
        target_info=lyap_target,
        model_info=model_info,
        lyap_config=lyap_config,
        bounds_info=bounds_info,
        u_prev=u_prev,
    )

    V_next = post.get("V_next_cand")
    V_bound = post.get("V_bound")
    base_debug.update({
        "final_lyap_value": V_next,
        "final_lyap_bound": V_bound,
        "final_lyap_margin": None if V_next is None or V_bound is None else float(V_bound) - float(V_next),
        "final_lyap_ok": post.get("candidate_lyap_ok"),
        "final_y_next_pred": None if post.get("y_next_pred") is None else np.asarray(post["y_next_pred"], float).reshape(-1).copy(),
        "final_lyap_target_source": lyap_target_source,
    })
    return base_debug


def _fallback_candidates(target_info, bounds_info, u_prev):
    u_min = bounds_info.get("u_min")
    u_max = bounds_info.get("u_max")
    out = []

    fallback_safe_input = bounds_info.get("fallback_safe_input")
    if fallback_safe_input is not None:
        u_prev_safe = np.asarray(fallback_safe_input, float).reshape(-1)
        if u_min is not None and u_max is not None:
            u_prev_safe = np.clip(u_prev_safe, u_min, u_max)
        out.append(("fallback_previous_secondary", u_prev_safe))
    elif u_prev is not None:
        u_prev = np.asarray(u_prev, float).reshape(-1)
        if u_min is not None and u_max is not None:
            u_prev = np.clip(u_prev, u_min, u_max)
        out.append(("fallback_previous_secondary", u_prev))

    if target_info is not None and target_info.get("u_s") is not None:
        u_s = np.asarray(target_info["u_s"], float).reshape(-1)
        if u_min is not None and u_max is not None:
            u_s = np.clip(u_s, u_min, u_max)
        out.append(("fallback_steady_secondary", u_s))

    return out


def _attempt_secondary_fallbacks(base_debug, xhat_aug, target_info, model_info, lyap_config, bounds_info, u_prev):
    for mode, fallback in _fallback_candidates(target_info, bounds_info, u_prev):
        if target_info is None or not target_info.get("success", False):
            base_debug.update({
                "u_safe": fallback.copy(),
                "correction_mode": mode,
                "fallback_mode": mode,
                "fallback_verified": False,
                "verified": False,
            })
            _attach_final_lyap(
                base_debug=base_debug,
                u_try=fallback,
                xhat_aug=xhat_aug,
                target_info=target_info,
                model_info=model_info,
                lyap_config=lyap_config,
                bounds_info=bounds_info,
                u_prev=u_prev,
            )
            return fallback.copy(), base_debug

        post = _postcheck_action(
            u_try=fallback,
            xhat_aug=xhat_aug,
            target_info=target_info,
            model_info=model_info,
            lyap_config=lyap_config,
            bounds_info=bounds_info,
            u_prev=u_prev,
        )
        if post.get("accepted", False):
            base_debug.update({
                "accepted": True,
                "accept_reason": mode,
                "u_safe": fallback.copy(),
                "correction_mode": mode,
                "fallback_mode": mode,
                "fallback_verified": True,
                "fallback_bounds_ok": bool(post.get("candidate_bounds_ok", False)),
                "fallback_move_ok": bool(post.get("candidate_move_ok", False)),
                "fallback_lyap_ok": bool(post.get("candidate_lyap_ok", False)),
                "verified": True,
            })
            _attach_final_lyap(
                base_debug=base_debug,
                u_try=fallback,
                xhat_aug=xhat_aug,
                target_info=target_info,
                model_info=model_info,
                lyap_config=lyap_config,
                bounds_info=bounds_info,
                u_prev=u_prev,
            )
            return fallback.copy(), base_debug

    return None, base_debug


def _attempt_mpc_fallback(
    base_debug,
    xhat_aug,
    target_info,
    model_info,
    lyap_config,
    bounds_info,
    u_prev,
    fallback_config,
):
    if fallback_config is None:
        return None, base_debug

    fallback_config = dict(fallback_config)
    mode = fallback_config.get("mode", "offset_free_mpc")
    if mode != "offset_free_mpc":
        return None, base_debug

    y_sp = fallback_config.get("y_sp")
    MPC_obj = fallback_config.get("MPC_obj")
    if y_sp is None or MPC_obj is None:
        return None, base_debug

    u_mpc, mpc_info = solve_offset_free_mpc_candidate(
        MPC_obj=MPC_obj,
        y_sp=y_sp,
        u_prev_dev=u_prev if u_prev is not None else fallback_config.get("u_prev_dev", np.zeros(model_info["n_u"], dtype=float)),
        x0_model=fallback_config.get("x0_model", xhat_aug),
        IC_opt=fallback_config.get("IC_opt"),
        bnds=fallback_config.get("bnds"),
        cons=fallback_config.get("cons"),
        return_debug=True,
    )

    base_debug.update({
        "u_fallback_mpc": None if u_mpc is None else np.asarray(u_mpc, float).reshape(-1).copy(),
        "fallback_mode": "offset_free_mpc",
        "fallback_solver_status": mpc_info.get("status"),
        "fallback_solver_message": mpc_info.get("message"),
        "fallback_objective_value": mpc_info.get("objective_value"),
        "fallback_ic_next": None if mpc_info.get("IC_opt_next") is None else np.asarray(mpc_info["IC_opt_next"], float).reshape(-1).copy(),
        "fallback_upstream_info": mpc_info,
        "fallback_tracking_target_source": fallback_config.get("tracking_target_source"),
        "fallback_target_mismatch_inf": fallback_config.get("target_mismatch_inf"),
    })

    if u_mpc is None:
        return None, base_debug

    if target_info is None or not target_info.get("success", False):
        if bool(fallback_config.get("allow_unverified", True)):
            u_mpc = np.asarray(u_mpc, float).reshape(-1)
            if bounds_info.get("u_min") is not None and bounds_info.get("u_max") is not None:
                u_mpc = np.clip(u_mpc, bounds_info["u_min"], bounds_info["u_max"])
            base_debug.update({
                "accepted": False,
                "accept_reason": None,
                "reject_reason": "target_unavailable",
                "u_safe": u_mpc.copy(),
                "correction_mode": "fallback_mpc_unverified",
                "fallback_verified": False,
                "verified": False,
            })
            _attach_final_lyap(
                base_debug=base_debug,
                u_try=u_mpc,
                xhat_aug=xhat_aug,
                target_info=target_info,
                model_info=model_info,
                lyap_config=lyap_config,
                bounds_info=bounds_info,
                u_prev=u_prev,
            )
            return u_mpc.copy(), base_debug
        return None, base_debug

    post = _postcheck_action(
        u_try=u_mpc,
        xhat_aug=xhat_aug,
        target_info=target_info,
        model_info=model_info,
        lyap_config=lyap_config,
        bounds_info=bounds_info,
        u_prev=u_prev,
    )
    base_debug.update({
        "fallback_bounds_ok": bool(post.get("candidate_bounds_ok", False)),
        "fallback_move_ok": bool(post.get("candidate_move_ok", False)),
        "fallback_lyap_ok": bool(post.get("candidate_lyap_ok", False)),
    })

    if post.get("accepted", False):
        base_debug.update({
            "accepted": True,
            "accept_reason": "fallback_mpc_verified",
            "u_safe": np.asarray(u_mpc, float).reshape(-1).copy(),
            "correction_mode": "fallback_mpc_verified",
            "fallback_verified": True,
            "verified": True,
        })
        _attach_final_lyap(
            base_debug=base_debug,
            u_try=u_mpc,
            xhat_aug=xhat_aug,
            target_info=target_info,
            model_info=model_info,
            lyap_config=lyap_config,
            bounds_info=bounds_info,
            u_prev=u_prev,
        )
        return np.asarray(u_mpc, float).reshape(-1).copy(), base_debug

    if bool(fallback_config.get("allow_unverified", True)):
        u_mpc = np.asarray(u_mpc, float).reshape(-1)
        if bounds_info.get("u_min") is not None and bounds_info.get("u_max") is not None:
            u_mpc = np.clip(u_mpc, bounds_info["u_min"], bounds_info["u_max"])
        base_debug.update({
            "accepted": False,
            "accept_reason": None,
            "u_safe": u_mpc.copy(),
            "correction_mode": "fallback_mpc_unverified",
            "fallback_verified": False,
            "verified": False,
        })
        _attach_final_lyap(
            base_debug=base_debug,
            u_try=u_mpc,
            xhat_aug=xhat_aug,
            target_info=target_info,
            model_info=model_info,
            lyap_config=lyap_config,
            bounds_info=bounds_info,
            u_prev=u_prev,
        )
        return u_mpc.copy(), base_debug

    return None, base_debug


def apply_lyapunov_safety_filter(
    u_cand,
    xhat_aug,
    target_info,
    model_info,
    lyap_config,
    u_prev=None,
    bounds_info=None,
    fallback_config=None,
    return_debug=False,
):
    if bounds_info is None:
        bounds_info = {}
    else:
        bounds_info = dict(bounds_info)
    lyap_config = {} if lyap_config is None else dict(lyap_config)

    n_u = int(model_info["n_u"])
    n_y = int(model_info["n_y"])

    u_cand = _as_1d("u_cand", u_cand, expected_size=n_u)
    xhat_aug = _as_1d("xhat_aug", xhat_aug)
    u_prev = None if u_prev is None else _as_1d("u_prev", u_prev, expected_size=n_u)

    bounds_info["u_min"] = _maybe_vector(bounds_info.get("u_min"), n_u)
    bounds_info["u_max"] = _maybe_vector(bounds_info.get("u_max"), n_u)
    bounds_info["du_min"] = _maybe_vector(bounds_info.get("du_min"), n_u)
    bounds_info["du_max"] = _maybe_vector(bounds_info.get("du_max"), n_u)

    candidate_eval = _postcheck_action(
        u_try=u_cand,
        xhat_aug=xhat_aug,
        target_info=target_info,
        model_info=model_info,
        lyap_config=lyap_config,
        bounds_info=bounds_info,
        u_prev=u_prev,
    )

    base_debug = {
        "source": str(lyap_config.get("source", "unknown")),
        "accepted": bool(candidate_eval.get("accepted", False)),
        "accept_reason": candidate_eval.get("accept_reason"),
        "reject_reason": candidate_eval.get("reject_reason"),
        "candidate_bounds_ok": candidate_eval.get("candidate_bounds_ok"),
        "candidate_move_ok": candidate_eval.get("candidate_move_ok"),
        "candidate_lyap_ok": candidate_eval.get("candidate_lyap_ok"),
        "u_cand": u_cand.copy(),
        "u_prev": None if u_prev is None else u_prev.copy(),
        "u_safe": None,
        "u_s": None if target_info is None or target_info.get("u_s") is None else np.asarray(target_info["u_s"], float).reshape(-1).copy(),
        "x_s": None if target_info is None or target_info.get("x_s") is None else np.asarray(target_info["x_s"], float).reshape(-1).copy(),
        "d_s": None if target_info is None or target_info.get("d_s") is None else np.asarray(target_info["d_s"], float).reshape(-1).copy(),
        "e_x": None if candidate_eval.get("e_x") is None else np.asarray(candidate_eval["e_x"], float).reshape(-1).copy(),
        "V_k": candidate_eval.get("V_k"),
        "V_next_cand": candidate_eval.get("V_next_cand"),
        "V_bound": candidate_eval.get("V_bound"),
        "rho": float(lyap_config.get("rho", 0.99)),
        "eps_lyap": float(lyap_config.get("eps_lyap", 1e-9)),
        "solver_status": None,
        "solver_name": None,
        "solver_residuals": {},
        "trust_region_violation": None,
        "slack_v": 0.0,
        "slack_u": 0.0,
        "correction_mode": None,
        "verified": False,
        "target_success": bool(target_info is not None and target_info.get("success", False)),
        "target_info": target_info,
        "u_fallback_mpc": None,
        "fallback_mode": None,
        "fallback_verified": False,
        "fallback_solver_status": None,
        "fallback_solver_message": None,
        "fallback_objective_value": None,
        "fallback_bounds_ok": None,
        "fallback_move_ok": None,
        "fallback_lyap_ok": None,
        "fallback_ic_next": None,
        "fallback_upstream_info": None,
        "fallback_tracking_target_source": None,
        "fallback_target_mismatch_inf": None,
        "qcqp_tracking_target": None,
        "qcqp_tracking_target_source": lyap_config.get("tracking_output_target_source"),
        "final_lyap_value": None,
        "final_lyap_margin": None,
        "final_lyap_ok": None,
        "final_lyap_bound": None,
        "final_y_next_pred": None,
        "final_lyap_target_source": None,
    }

    if candidate_eval.get("accepted", False):
        base_debug.update({
            "u_safe": u_cand.copy(),
            "correction_mode": "accepted_candidate",
            "verified": True,
        })
        _attach_final_lyap(
            base_debug=base_debug,
            u_try=u_cand,
            xhat_aug=xhat_aug,
            target_info=target_info,
            model_info=model_info,
            lyap_config=lyap_config,
            bounds_info=bounds_info,
            u_prev=u_prev,
        )
        return u_cand.copy(), base_debug

    if target_info is None or not target_info.get("success", False):
        u_safe, base_debug = _attempt_mpc_fallback(
            base_debug=base_debug,
            xhat_aug=xhat_aug,
            target_info=target_info,
            model_info=model_info,
            lyap_config=lyap_config,
            bounds_info=bounds_info,
            u_prev=u_prev,
            fallback_config=fallback_config,
        )
        if u_safe is not None:
            return u_safe, base_debug

        u_safe, base_debug = _attempt_secondary_fallbacks(
            base_debug=base_debug,
            xhat_aug=xhat_aug,
            target_info=target_info,
            model_info=model_info,
            lyap_config=lyap_config,
            bounds_info=bounds_info,
            u_prev=u_prev,
        )
        if u_safe is not None:
            return u_safe, base_debug

        base_debug.update({
            "u_safe": u_cand.copy(),
            "correction_mode": "target_unavailable_unverified",
            "verified": False,
        })
        _attach_final_lyap(
            base_debug=base_debug,
            u_try=u_cand,
            xhat_aug=xhat_aug,
            target_info=target_info,
            model_info=model_info,
            lyap_config=lyap_config,
            bounds_info=bounds_info,
            u_prev=u_prev,
        )
        return u_cand.copy(), base_debug

    if not HAS_CVXPY:
        u_safe, base_debug = _attempt_mpc_fallback(
            base_debug=base_debug,
            xhat_aug=xhat_aug,
            target_info=target_info,
            model_info=model_info,
            lyap_config=lyap_config,
            bounds_info=bounds_info,
            u_prev=u_prev,
            fallback_config=fallback_config,
        )
        if u_safe is not None:
            return u_safe, base_debug

        u_safe, base_debug = _attempt_secondary_fallbacks(
            base_debug=base_debug,
            xhat_aug=xhat_aug,
            target_info=target_info,
            model_info=model_info,
            lyap_config=lyap_config,
            bounds_info=bounds_info,
            u_prev=u_prev,
        )
        if u_safe is not None:
            return u_safe, base_debug

        fallback = np.asarray(target_info["u_s"], float).reshape(-1).copy()
        if bounds_info["u_min"] is not None and bounds_info["u_max"] is not None:
            fallback = np.clip(fallback, bounds_info["u_min"], bounds_info["u_max"])
        base_debug.update({
            "u_safe": fallback.copy(),
            "correction_mode": "no_cvxpy_unverified",
            "solver_status": "no_cvxpy",
            "verified": False,
        })
        _attach_final_lyap(
            base_debug=base_debug,
            u_try=fallback,
            xhat_aug=xhat_aug,
            target_info=target_info,
            model_info=model_info,
            lyap_config=lyap_config,
            bounds_info=bounds_info,
            u_prev=u_prev,
        )
        return fallback.copy(), base_debug

    rho = float(lyap_config.get("rho", 0.99))
    eps_lyap = float(lyap_config.get("eps_lyap", 1e-9))

    W_cand, _ = _weight_diag(lyap_config.get("candidate_weight_diag"), n_u, default=1.0)
    W_move, _ = _weight_diag(lyap_config.get("move_weight_diag"), n_u, default=1.0)
    W_steady, _ = _weight_diag(lyap_config.get("steady_weight_diag"), n_u, default=1.0)
    W_output, _ = _weight_diag(lyap_config.get("output_weight_diag"), n_y, default=1.0)

    use_output_tracking = bool(lyap_config.get("use_output_tracking_term", True))
    allow_lyap_slack = bool(lyap_config.get("allow_lyap_slack", False))
    lyap_slack_weight = float(lyap_config.get("lyap_slack_weight", 1e6))
    trust_region_delta = lyap_config.get("trust_region_delta")
    trust_region_weight = float(lyap_config.get("trust_region_weight", 1e4))

    e_x = np.asarray(candidate_eval["e_x"], float).reshape(-1)
    u_s = np.asarray(target_info["u_s"], float).reshape(-1)
    d_s = np.asarray(target_info["d_s"], float).reshape(-1)
    y_s = np.asarray(target_info["y_s"], float).reshape(-1)
    y_track = _qcqp_output_target(target_info, lyap_config, n_y=n_y)
    A = np.asarray(model_info["A_phys"], float)
    B = np.asarray(model_info["B_phys"], float)
    C = np.asarray(model_info["C_phys"], float)
    Cd = np.asarray(model_info["Cd_phys"], float)
    P_x = np.asarray(model_info["P_x"], float)
    V_k = float(candidate_eval["V_k"])
    V_bound = float(lyapunov_bound(V_k, rho=rho, eps_lyap=eps_lyap))
    base_debug["qcqp_tracking_target"] = None if y_track is None else y_track.copy()
    if base_debug["qcqp_tracking_target_source"] is None:
        base_debug["qcqp_tracking_target_source"] = "selector_target"

    u_var = cp.Variable(n_u)
    slack_v = cp.Variable(nonneg=True) if allow_lyap_slack else None
    slack_u = cp.Variable(nonneg=True) if trust_region_delta is not None else None

    e_next_expr = A @ e_x + B @ (u_var - u_s)
    y_next_expr = C @ (np.asarray(target_info["x_s"], float).reshape(-1) + e_next_expr) + Cd @ d_s
    V_next_expr = cp.quad_form(e_next_expr, P_x)

    objective = cp.quad_form(u_var - u_cand, W_cand)
    if u_prev is not None:
        objective += cp.quad_form(u_var - u_prev, W_move)
    objective += cp.quad_form(u_var - u_s, W_steady)
    if use_output_tracking and y_track is not None:
        objective += cp.quad_form(y_next_expr - y_track, W_output)
    if slack_v is not None:
        objective += lyap_slack_weight * cp.square(slack_v)
    if slack_u is not None:
        objective += trust_region_weight * cp.square(slack_u)

    constraints = []
    if bounds_info["u_min"] is not None:
        constraints.append(u_var >= bounds_info["u_min"])
    if bounds_info["u_max"] is not None:
        constraints.append(u_var <= bounds_info["u_max"])
    if u_prev is not None and bounds_info["du_min"] is not None:
        constraints.append(u_var - u_prev >= bounds_info["du_min"])
    if u_prev is not None and bounds_info["du_max"] is not None:
        constraints.append(u_var - u_prev <= bounds_info["du_max"])

    lyap_rhs = V_bound if slack_v is None else V_bound + slack_v
    constraints.append(V_next_expr <= lyap_rhs)

    trust_region_violation = None
    if trust_region_delta is not None:
        trust_region_delta = np.maximum(_maybe_vector(trust_region_delta, n_u), 0.0)
        trust_rhs = trust_region_delta if slack_u is None else trust_region_delta + slack_u
        constraints.append(u_var - u_cand <= trust_rhs)
        constraints.append(u_cand - u_var <= trust_rhs)

    problem = cp.Problem(cp.Minimize(objective), constraints)
    solver_sequence = safety_filter_solver_sequence(
        quadratic_constraint_active=True,
        solver_pref=lyap_config.get("solver_pref"),
    )

    best_action = None
    best_debug = None
    last_error = None

    for solver_name in solver_sequence:
        try:
            u_var.value = None
            if slack_v is not None:
                slack_v.value = None
            if slack_u is not None:
                slack_u.value = None
            problem.solve(solver=solver_name, warm_start=True, verbose=False)
        except Exception as exc:
            last_error = repr(exc)
            continue

        if u_var.value is None or problem.status not in _OPTIMAL_STATUSES:
            continue

        u_try = np.asarray(u_var.value, float).reshape(-1)
        post = _postcheck_action(
            u_try=u_try,
            xhat_aug=xhat_aug,
            target_info=target_info,
            model_info=model_info,
            lyap_config=lyap_config,
            bounds_info=bounds_info,
            u_prev=u_prev,
        )
        trust_region_violation = 0.0
        if trust_region_delta is not None:
            trust_region_violation = float(
                np.max(np.maximum(np.abs(u_try - u_cand) - trust_region_delta, 0.0))
            )

        trial_debug = {
            "solver_status": problem.status,
            "solver_name": solver_name,
            "solver_residuals": {
                "lyap_margin_post": float(post.get("lyap_margin", np.nan)),
                "input_bounds_violation": float(post.get("candidate_bounds_violation", np.nan)),
                "move_bounds_violation": float(post.get("candidate_move_violation", np.nan)),
            },
            "trust_region_violation": trust_region_violation,
            "slack_v": 0.0 if slack_v is None or slack_v.value is None else float(np.asarray(slack_v.value).item()),
            "slack_u": 0.0 if slack_u is None or slack_u.value is None else float(np.asarray(slack_u.value).item()),
            "objective_value": None if problem.value is None else float(problem.value),
            "V_next_post": post.get("V_next_cand"),
            "y_next_post": None if post.get("y_next_pred") is None else np.asarray(post["y_next_pred"], float).reshape(-1).copy(),
        }

        best_action = u_try
        best_debug = trial_debug
        if post.get("accepted", False):
            base_debug.update({
                "accepted": True,
                "accept_reason": "optimized_correction",
                "reject_reason": candidate_eval.get("reject_reason"),
                "u_safe": u_try.copy(),
                "correction_mode": "optimized_correction",
                "solver_status": trial_debug["solver_status"],
                "solver_name": trial_debug["solver_name"],
                "solver_residuals": trial_debug["solver_residuals"],
                "trust_region_violation": trial_debug["trust_region_violation"],
                "slack_v": trial_debug["slack_v"],
                "slack_u": trial_debug["slack_u"],
                "verified": True,
            })
            _attach_final_lyap(
                base_debug=base_debug,
                u_try=u_try,
                xhat_aug=xhat_aug,
                target_info=target_info,
                model_info=model_info,
                lyap_config=lyap_config,
                bounds_info=bounds_info,
                u_prev=u_prev,
            )
            return u_try.copy(), base_debug

    if best_debug is not None:
        base_debug.update({
            "solver_status": best_debug["solver_status"],
            "solver_name": best_debug["solver_name"],
            "solver_residuals": best_debug["solver_residuals"],
            "trust_region_violation": best_debug["trust_region_violation"],
            "slack_v": best_debug["slack_v"],
            "slack_u": best_debug["slack_u"],
        })

    u_safe, base_debug = _attempt_mpc_fallback(
        base_debug=base_debug,
        xhat_aug=xhat_aug,
        target_info=target_info,
        model_info=model_info,
        lyap_config=lyap_config,
        bounds_info=bounds_info,
        u_prev=u_prev,
        fallback_config=fallback_config,
    )
    if u_safe is not None:
        return u_safe, base_debug

    u_safe, base_debug = _attempt_secondary_fallbacks(
        base_debug=base_debug,
        xhat_aug=xhat_aug,
        target_info=target_info,
        model_info=model_info,
        lyap_config=lyap_config,
        bounds_info=bounds_info,
        u_prev=u_prev,
    )
    if u_safe is not None:
        return u_safe, base_debug

    if best_action is None:
        best_action = np.asarray(target_info["u_s"], float).reshape(-1).copy()
    if bounds_info["u_min"] is not None and bounds_info["u_max"] is not None:
        best_action = np.clip(best_action, bounds_info["u_min"], bounds_info["u_max"])

    base_debug.update({
        "accepted": False,
        "accept_reason": None,
        "reject_reason": candidate_eval.get("reject_reason"),
        "u_safe": best_action.copy(),
        "correction_mode": "unverified_fallback",
        "solver_residuals": {
            **base_debug["solver_residuals"],
            "solver_error": last_error,
        },
        "verified": False,
    })
    _attach_final_lyap(
        base_debug=base_debug,
        u_try=best_action,
        xhat_aug=xhat_aug,
        target_info=target_info,
        model_info=model_info,
        lyap_config=lyap_config,
        bounds_info=bounds_info,
        u_prev=u_prev,
    )
    return best_action.copy(), base_debug
