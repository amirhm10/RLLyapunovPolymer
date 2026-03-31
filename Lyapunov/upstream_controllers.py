from types import SimpleNamespace

import numpy as np
import scipy.optimize as spo


def build_repeated_input_bounds(u_min, u_max, horizon_control):
    u_min = np.asarray(u_min, float).reshape(-1)
    u_max = np.asarray(u_max, float).reshape(-1)
    if u_min.size != u_max.size:
        raise ValueError("u_min and u_max must have the same size.")
    if np.any(u_min > u_max):
        raise ValueError("u_min must be <= u_max elementwise.")

    bounds = []
    for _ in range(int(horizon_control)):
        for idx in range(u_min.size):
            bounds.append((float(u_min[idx]), float(u_max[idx])))
    return tuple(bounds)


def default_mpc_initial_guess(n_inputs, horizon_control, fill_value=0.0):
    return np.full(int(n_inputs) * int(horizon_control), float(fill_value), dtype=float)


def solve_offset_free_mpc_candidate(
    MPC_obj,
    y_sp,
    u_prev_dev,
    x0_model,
    IC_opt=None,
    bnds=None,
    cons=None,
    return_debug=False,
):
    n_u = int(MPC_obj.B.shape[1])
    horizon_control = int(getattr(MPC_obj, "NC", 1))

    y_sp = np.asarray(y_sp, float).reshape(-1)
    u_prev_dev = np.asarray(u_prev_dev, float).reshape(-1)
    x0_model = np.asarray(x0_model, float).reshape(-1)

    if IC_opt is None:
        ic_used = default_mpc_initial_guess(n_u, horizon_control)
    else:
        ic_used = np.asarray(IC_opt, float).reshape(-1)

    if ic_used.size != n_u * horizon_control:
        raise ValueError(
            f"IC_opt has size {ic_used.size}, expected {n_u * horizon_control}."
        )

    constraints = () if cons is None else tuple(cons)

    sol = spo.minimize(
        lambda x: MPC_obj.mpc_opt_fun(x, y_sp, u_prev_dev, x0_model),
        ic_used,
        bounds=bnds,
        constraints=constraints,
    )

    x_opt = None if getattr(sol, "x", None) is None else np.asarray(sol.x, float).reshape(-1)
    if x_opt is None or x_opt.size < n_u:
        u_candidate = None
        ic_next = ic_used.copy()
    else:
        u_candidate = x_opt[:n_u].copy()
        ic_next = x_opt.copy()

    info = {
        "success": bool(getattr(sol, "success", False)),
        "status": getattr(sol, "status", None),
        "message": None if getattr(sol, "message", None) is None else str(sol.message),
        "objective_value": None if getattr(sol, "fun", None) is None else float(sol.fun),
        "nit": getattr(sol, "nit", None),
        "candidate_available": bool(u_candidate is not None),
        "u_candidate": None if u_candidate is None else u_candidate.copy(),
        "x_opt": None if x_opt is None else x_opt.copy(),
        "IC_opt_used": ic_used.copy(),
        "IC_opt_next": ic_next.copy(),
        "bounds_used": bnds,
        "num_constraints": len(constraints),
    }

    if return_debug:
        return u_candidate, info
    return u_candidate


def _predict_augmented_state_path(MPC_obj, decision, x0_model):
    n_u = int(MPC_obj.B.shape[1])
    horizon_control = int(getattr(MPC_obj, "NC", 1))
    horizon_prediction = int(getattr(MPC_obj, "NP", 1))

    decision = np.asarray(decision, float).reshape(-1)
    x0_model = np.asarray(x0_model, float).reshape(-1)
    U = decision[: n_u * horizon_control].reshape(horizon_control, n_u)

    x_pred = np.zeros((int(MPC_obj.A.shape[0]), horizon_prediction + 1), dtype=float)
    x_pred[:, 0] = x0_model
    for step_idx in range(horizon_prediction):
        ctrl_idx = step_idx if step_idx < horizon_control else horizon_control - 1
        x_pred[:, step_idx + 1] = MPC_obj.A @ x_pred[:, step_idx] + MPC_obj.B @ U[ctrl_idx, :]
    return x_pred


def solve_offset_free_mpc_candidate_with_first_step_contraction(
    MPC_obj,
    y_sp,
    u_prev_dev,
    x0_model,
    x_s,
    P_x,
    rho_lyap,
    eps_lyap,
    lyap_tol=1e-9,
    IC_opt=None,
    bnds=None,
    cons=None,
    return_debug=False,
):
    from Lyapunov.lyapunov_core import lyapunov_bound, lyapunov_value

    n_u = int(MPC_obj.B.shape[1])
    horizon_control = int(getattr(MPC_obj, "NC", 1))

    y_sp = np.asarray(y_sp, float).reshape(-1)
    u_prev_dev = np.asarray(u_prev_dev, float).reshape(-1)
    x0_model = np.asarray(x0_model, float).reshape(-1)
    x_s = np.asarray(x_s, float).reshape(-1)
    P_x = np.asarray(P_x, float)
    n_x = int(x_s.size)

    if x0_model.size < n_x:
        raise ValueError("x0_model must contain at least the physical-state coordinates.")
    if P_x.shape != (n_x, n_x):
        raise ValueError(f"P_x must have shape {(n_x, n_x)}, got {P_x.shape}.")

    if IC_opt is None:
        ic_used = default_mpc_initial_guess(n_u, horizon_control)
    else:
        ic_used = np.asarray(IC_opt, float).reshape(-1)

    if ic_used.size != n_u * horizon_control:
        raise ValueError(
            f"IC_opt has size {ic_used.size}, expected {n_u * horizon_control}."
        )

    V_k = lyapunov_value(x0_model[:n_x] - x_s, P_x)
    V_bound = float(lyapunov_bound(V_k, rho=rho_lyap, eps_lyap=eps_lyap))

    constraints = [] if cons is None else list(cons)

    def _first_step_contraction_fun(decision):
        x_pred = _predict_augmented_state_path(MPC_obj, decision, x0_model)
        e_x_next_first = np.asarray(x_pred[:n_x, 1], float).reshape(-1) - x_s
        V_next_first = lyapunov_value(e_x_next_first, P_x)
        return float(V_bound - V_next_first)

    constraints.append({"type": "ineq", "fun": _first_step_contraction_fun})

    sol = spo.minimize(
        lambda x: MPC_obj.mpc_opt_fun(x, y_sp, u_prev_dev, x0_model),
        ic_used,
        bounds=bnds,
        constraints=tuple(constraints),
        method="SLSQP",
    )

    x_opt = None if getattr(sol, "x", None) is None else np.asarray(sol.x, float).reshape(-1)
    if x_opt is None or x_opt.size < n_u:
        u_candidate = None
        ic_next = ic_used.copy()
        x_pred = None
        V_next_first = None
        contraction_margin = None
        first_step_contraction_satisfied = None
    else:
        u_candidate = x_opt[:n_u].copy()
        ic_next = x_opt.copy()
        x_pred = _predict_augmented_state_path(MPC_obj, x_opt, x0_model)
        e_x_next_first = np.asarray(x_pred[:n_x, 1], float).reshape(-1) - x_s
        V_next_first = lyapunov_value(e_x_next_first, P_x)
        contraction_margin = float(V_next_first - V_bound)
        first_step_contraction_satisfied = bool(contraction_margin <= float(lyap_tol))

    info = {
        "success": bool(getattr(sol, "success", False)),
        "status": getattr(sol, "status", None),
        "message": None if getattr(sol, "message", None) is None else str(sol.message),
        "objective_value": None if getattr(sol, "fun", None) is None else float(sol.fun),
        "nit": getattr(sol, "nit", None),
        "candidate_available": bool(u_candidate is not None),
        "u_candidate": None if u_candidate is None else u_candidate.copy(),
        "x_opt": None if x_opt is None else x_opt.copy(),
        "IC_opt_used": ic_used.copy(),
        "IC_opt_next": ic_next.copy(),
        "bounds_used": bnds,
        "num_constraints": len(constraints),
        "x_pred_path": None if x_pred is None else x_pred.copy(),
        "V_k": float(V_k),
        "V_bound": float(V_bound),
        "V_next_first": None if V_next_first is None else float(V_next_first),
        "contraction_margin": None if contraction_margin is None else float(contraction_margin),
        "first_step_contraction_satisfied": first_step_contraction_satisfied,
        "solver_name": "SLSQP",
    }

    if return_debug:
        return u_candidate, info
    return u_candidate
