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
