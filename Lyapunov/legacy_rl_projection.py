import numpy as np
from scipy.linalg import solve_discrete_are

try:
    import cvxpy as cp

    HAS_CVXPY = True
except Exception:
    HAS_CVXPY = False

from utils.lyapunov_utils import diag_psd_from_vector


def compute_ss_target_slack_bounded(
    A,
    B,
    C,
    y_sp,
    d_hat,
    u_min,
    u_max,
    u_nom=None,
    Qs_diag=None,
    Ru_diag=None,
    w_x=1e-6,
    solver_pref=("OSQP", "CLARABEL", "SCS"),
    return_debug=False,
):
    if not HAS_CVXPY:
        raise ImportError("CVXPY is required for the legacy RL steady-state target solver.")

    A = np.asarray(A, float)
    B = np.asarray(B, float)
    C = np.asarray(C, float)
    y_sp = np.asarray(y_sp, float).reshape(-1)
    d_hat = np.asarray(d_hat, float).reshape(-1)
    u_min = np.asarray(u_min, float).reshape(-1)
    u_max = np.asarray(u_max, float).reshape(-1)

    n_x = A.shape[0]
    n_u = B.shape[1]
    n_y = C.shape[0]

    if A.shape != (n_x, n_x):
        raise ValueError("A must be square.")
    if B.shape[0] != n_x:
        raise ValueError("B row dimension must match A.")
    if C.shape[1] != n_x:
        raise ValueError("C column dimension must match A.")
    if y_sp.size != n_y:
        raise ValueError("y_sp size mismatch.")
    if d_hat.size != n_y:
        raise ValueError("d_hat size mismatch.")
    if u_min.size != n_u or u_max.size != n_u:
        raise ValueError("u_min/u_max size mismatch.")
    if np.any(u_min > u_max):
        raise ValueError("u_min must be <= u_max elementwise.")

    if u_nom is None:
        u_nom = np.zeros(n_u, dtype=float)
    else:
        u_nom = np.asarray(u_nom, float).reshape(-1)
        if u_nom.size != n_u:
            raise ValueError("u_nom size mismatch.")

    Qs, qs_used = diag_psd_from_vector(Qs_diag, n_y, eps=1e-12, default=1.0)
    Ru, ru_used = diag_psd_from_vector(Ru_diag, n_u, eps=1e-12, default=1.0)

    y_t = y_sp - d_hat

    x = cp.Variable(n_x)
    u = cp.Variable(n_u)
    s = cp.Variable(n_y)

    objective = (
        cp.quad_form(u - u_nom, Ru)
        + cp.quad_form(s, Qs)
        + float(w_x) * cp.sum_squares(x)
    )

    constraints = [
        (np.eye(n_x) - A) @ x - B @ u == 0.0,
        C @ x - y_t == s,
        u >= u_min,
        u <= u_max,
    ]

    problem = cp.Problem(cp.Minimize(objective), constraints)

    solved = False
    last_status = None
    last_solver = None
    last_err = None

    tol_dyn = 1e-8
    tol_slack = 1e-6

    for solver_name in solver_pref:
        try:
            x.value = None
            u.value = None
            s.value = None

            problem.solve(
                solver=solver_name,
                warm_start=False,
                verbose=False,
            )

            last_status = problem.status
            last_solver = solver_name

            if x.value is None or u.value is None or s.value is None:
                continue

            x_try = np.asarray(x.value).reshape(-1)
            u_try = np.asarray(u.value).reshape(-1)
            s_try = np.asarray(s.value).reshape(-1)

            r_dyn = (np.eye(n_x) - A) @ x_try - B @ u_try
            r_slack = C @ x_try - y_t - s_try

            r_dyn_inf = float(np.max(np.abs(r_dyn)))
            r_slack_inf = float(np.max(np.abs(r_slack)))

            if problem.status == "optimal":
                solved = True
                break

            if problem.status == "optimal_inaccurate":
                if r_dyn_inf <= tol_dyn and r_slack_inf <= tol_slack:
                    solved = True
                    break

        except Exception as ex:
            last_err = str(ex)
            last_status = None
            last_solver = solver_name

    if not solved:
        dbg = {
            "success": False,
            "solver": last_solver,
            "status": last_status,
            "err": last_err,
            "bounded": True,
            "y_target": y_sp.copy(),
            "y_shifted_target": y_t.copy(),
            "d_s": d_hat.copy(),
            "u_nom": u_nom.copy(),
            "Qs_diag": qs_used.copy(),
            "Ru_diag": ru_used.copy(),
            "objective_value": None,
            "r_dyn_inf": None,
            "r_y_inf": None,
            "slack_y_inf": None,
            "slack_y_norm": None,
            "y_s_pred": None,
        }
        if return_debug:
            return None, None, d_hat.copy(), dbg
        return None, None, d_hat.copy()

    x_s = np.asarray(x.value).reshape(-1)
    u_s = np.asarray(u.value).reshape(-1)
    s_val = np.asarray(s.value).reshape(-1)
    d_s = d_hat.copy()

    r_dyn = (np.eye(n_x) - A) @ x_s - B @ u_s
    r_y = C @ x_s - y_t
    y_s_pred = C @ x_s + d_s
    target_error = y_s_pred - y_sp

    dbg = {
        "success": True,
        "solver": last_solver,
        "status": problem.status,
        "bounded": True,
        "objective_value": float(problem.value) if problem.value is not None else None,
        "r_dyn_inf": float(np.max(np.abs(r_dyn))),
        "r_y_inf": float(np.max(np.abs(r_y))),
        "dyn_residual_inf": float(np.max(np.abs(r_dyn))),
        "target_error_inf": float(np.max(np.abs(target_error))),
        "target_error_norm": float(np.linalg.norm(target_error, ord=2)),
        "slack_y_inf": float(np.max(np.abs(s_val))),
        "slack_y_norm": float(np.linalg.norm(s_val, ord=2)),
        "y_s_pred": y_s_pred.copy(),
        "y_target": y_sp.copy(),
        "y_shifted_target": y_t.copy(),
        "d_s": d_s.copy(),
        "u_nom": u_nom.copy(),
        "Qs_diag": qs_used.copy(),
        "Ru_diag": ru_used.copy(),
        "x_s_norm": float(np.linalg.norm(x_s, ord=2)),
        "u_s_norm": float(np.linalg.norm(u_s, ord=2)),
    }

    if return_debug:
        return x_s, u_s, d_s, dbg
    return x_s, u_s, d_s


def compute_ss_target_slack_from_augmented(
    A_aug,
    B_aug,
    C_aug,
    xhat_aug,
    y_sp,
    u_min,
    u_max,
    u_nom=None,
    Qs_diag=None,
    Ru_diag=None,
    w_x=1e-6,
    solver_pref=("OSQP", "CLARABEL", "SCS"),
    return_debug=False,
):
    A_aug = np.asarray(A_aug, float)
    B_aug = np.asarray(B_aug, float)
    C_aug = np.asarray(C_aug, float)
    xhat_aug = np.asarray(xhat_aug, float).reshape(-1)

    n_aug = A_aug.shape[0]
    n_y = C_aug.shape[0]
    n_x = n_aug - n_y

    if A_aug.shape[1] != n_aug:
        raise ValueError("A_aug must be square.")
    if B_aug.shape[0] != n_aug:
        raise ValueError("B_aug row dimension must match A_aug.")
    if C_aug.shape[1] != n_aug:
        raise ValueError("C_aug column dimension must match A_aug.")
    if xhat_aug.size != n_aug:
        raise ValueError("xhat_aug size mismatch.")

    A = A_aug[:n_x, :n_x]
    B = B_aug[:n_x, :]
    C = C_aug[:, :n_x]
    d_hat = xhat_aug[n_x:]

    return compute_ss_target_slack_bounded(
        A=A,
        B=B,
        C=C,
        y_sp=y_sp,
        d_hat=d_hat,
        u_min=u_min,
        u_max=u_max,
        u_nom=u_nom,
        Qs_diag=Qs_diag,
        Ru_diag=Ru_diag,
        w_x=w_x,
        solver_pref=solver_pref,
        return_debug=return_debug,
    )


def design_riccati_P_aug_physical(
    A_aug,
    B_aug,
    C_aug,
    Qy_diag,
    Ru_diag=None,
    u_min=None,
    u_max=None,
    u_nom=None,
    lambda_u=1.0,
    pd_eps=0.0,
    eps_r=1e-9,
    return_debug=False,
):
    A_aug = np.asarray(A_aug, float)
    B_aug = np.asarray(B_aug, float)
    C_aug = np.asarray(C_aug, float)
    Qy_diag = np.asarray(Qy_diag, float).reshape(-1)

    n_aug = A_aug.shape[0]
    n_u = B_aug.shape[1]
    n_y = C_aug.shape[0]
    n_x = n_aug - n_y

    if A_aug.shape[1] != n_aug:
        raise ValueError("A_aug must be square.")
    if B_aug.shape[0] != n_aug:
        raise ValueError("B_aug row dimension must match A_aug.")
    if C_aug.shape[1] != n_aug:
        raise ValueError("C_aug column dimension must match A_aug.")
    if Qy_diag.size != n_y:
        raise ValueError(f"Qy_diag size mismatch. Expected {n_y}, got {Qy_diag.size}.")

    A = A_aug[:n_x, :n_x]
    B = B_aug[:n_x, :]
    C = C_aug[:, :n_x]

    if u_nom is None:
        u_nom = np.zeros(n_u, dtype=float)
    else:
        u_nom = np.asarray(u_nom, float).reshape(-1)
        if u_nom.size != n_u:
            raise ValueError(f"u_nom size mismatch. Expected {n_u}, got {u_nom.size}.")

    Qy = np.diag(np.maximum(Qy_diag, 0.0))
    Qx = C.T @ Qy @ C
    Qx = 0.5 * (Qx + Qx.T)

    if Ru_diag is None:
        if u_min is None or u_max is None:
            raise ValueError("u_min and u_max must be provided when Ru_diag is None.")

        u_min = np.asarray(u_min, float).reshape(-1)
        u_max = np.asarray(u_max, float).reshape(-1)

        if u_min.size != n_u or u_max.size != n_u:
            raise ValueError("u_min/u_max size mismatch.")
        if np.any(u_min > u_max):
            raise ValueError("u_min must be <= u_max elementwise.")
        if np.any(u_nom < u_min) or np.any(u_nom > u_max):
            raise ValueError("u_nom must lie inside [u_min, u_max] for Bryson-style Ru.")

        u_allow = np.minimum(u_nom - u_min, u_max - u_nom)
        if np.any(u_allow <= 0.0):
            raise ValueError("Each input must have positive room around u_nom.")

        Ru_diag = float(lambda_u) / np.maximum(u_allow ** 2, eps_r)

    Ru_diag = np.asarray(Ru_diag, float).reshape(-1)
    if Ru_diag.size != n_u:
        raise ValueError(f"Ru_diag size mismatch. Expected {n_u}, got {Ru_diag.size}.")

    Ru_diag = np.maximum(Ru_diag, eps_r)
    Ru = np.diag(Ru_diag)

    Px = solve_discrete_are(A, B, Qx, Ru)
    Px = 0.5 * (Px + Px.T)

    Pd = float(pd_eps) * np.eye(n_y, dtype=float)
    P_aug = np.block([
        [Px, np.zeros((n_x, n_y), dtype=float)],
        [np.zeros((n_y, n_x), dtype=float), Pd],
    ])
    P_aug = 0.5 * (P_aug + P_aug.T)

    if return_debug:
        dbg = {
            "n_x": n_x,
            "n_y": n_y,
            "n_u": n_u,
            "u_nom": u_nom.copy(),
            "Qy_diag": Qy_diag.copy(),
            "Qx": Qx.copy(),
            "Ru_diag": Ru_diag.copy(),
            "pd_eps": float(pd_eps),
        }
        return P_aug, dbg

    return P_aug


def lyap_V(x, P):
    x = np.asarray(x, float).reshape(-1, 1)
    P = np.asarray(P, float)
    return float((x.T @ P @ x).item())


def factor_psd_left(P, neg_tol=1e-10):
    P = np.asarray(P, float)
    P = 0.5 * (P + P.T)

    w, Q = np.linalg.eigh(P)

    if np.min(w) < -neg_tol:
        raise ValueError(f"P is not positive semidefinite. Min eigenvalue = {np.min(w):.3e}")

    w = np.where(w > 0.0, w, 0.0)
    return np.diag(np.sqrt(w)) @ Q.T


def _legacy_target_info(xhat_aug, y_sp, u_prev_dev, x_s, u_s, d_s, dbg_tgt):
    y_sp = np.asarray(y_sp, float).reshape(-1)
    xhat_aug = np.asarray(xhat_aug, float).reshape(-1)
    u_prev_dev = np.asarray(u_prev_dev, float).reshape(-1)
    d_s = None if d_s is None else np.asarray(d_s, float).reshape(-1)
    x_s = None if x_s is None else np.asarray(x_s, float).reshape(-1)
    u_s = None if u_s is None else np.asarray(u_s, float).reshape(-1)
    y_s = dbg_tgt.get("y_s_pred")
    y_s = None if y_s is None else np.asarray(y_s, float).reshape(-1)

    target_error = None if y_s is None else (y_s - y_sp)
    n_x = 0 if x_s is None else x_s.size
    xhat_x = xhat_aug[:n_x] if n_x > 0 else None

    target_info = {
        "success": bool(dbg_tgt.get("success", False) and x_s is not None and u_s is not None),
        "selector_mode": "legacy_augstate_rl",
        "selector_name": "legacy_augmented_slack_target",
        "solve_stage": None if not dbg_tgt.get("success", False) else "legacy_slack_target",
        "x_s": None if x_s is None else x_s.copy(),
        "u_s": None if u_s is None else u_s.copy(),
        "d_s": None if d_s is None else d_s.copy(),
        "x_s_aug": None if x_s is None or d_s is None else np.concatenate([x_s, d_s], axis=0),
        "y_s": None if y_s is None else y_s.copy(),
        "yc_s": None if y_s is None else y_s.copy(),
        "r_s": y_sp.copy(),
        "requested_y_sp": y_sp.copy(),
        "target_error": None if target_error is None else target_error.copy(),
        "target_error_inf": None if target_error is None else float(np.max(np.abs(target_error))),
        "target_error_norm": None if target_error is None else float(np.linalg.norm(target_error, ord=2)),
        "target_slack_inf": dbg_tgt.get("slack_y_inf"),
        "dyn_residual_inf": dbg_tgt.get("r_dyn_inf"),
        "bound_violation_inf": 0.0 if dbg_tgt.get("success", False) else None,
        "d_s_minus_dhat_inf": 0.0 if dbg_tgt.get("success", False) else None,
        "d_s_frozen": True,
        "d_s_optimized": False,
        "objective_terms": {},
        "objective_value": dbg_tgt.get("objective_value"),
        "selector_debug": {
            "status": dbg_tgt.get("status"),
            "solver": dbg_tgt.get("solver"),
            "prev_input_term_active": False,
            "prev_state_term_active": False,
            "Qr_diag_used": dbg_tgt.get("Qs_diag"),
            "R_u_ref_diag_used": dbg_tgt.get("Ru_diag"),
            "R_delta_u_sel_diag_used": None,
            "Q_delta_x_diag_used": None,
            "Q_x_ref_diag_used": None,
            "Qx_base_diag_used": None,
            "Rdu_diag_used": None,
        },
        "x_s_minus_xhat": None if x_s is None or xhat_x is None else (x_s - xhat_x),
        "x_s_minus_x_prev": None,
        "u_s_minus_u_applied": None if u_s is None else (u_s - u_prev_dev),
        "u_s_minus_u_prev": None if u_s is None else (u_s - u_prev_dev),
    }
    return target_info


def lyapunov_project_layer_augstate(
    xhat_aug,
    y_sp,
    u_rl_dev,
    u_prev_dev,
    u_min,
    u_max,
    A_aug,
    B_aug,
    C_aug,
    P_lyap,
    S_lyap,
    rho=0.99,
    eps_v=1e-9,
    w_rl=1.0,
    w_track=1.0,
    w_move=1.0,
    w_ss=1.0,
    Qy_track_diag=None,
    Rmove_diag=None,
    Qs_tgt_diag=None,
    Ru_tgt_diag=None,
    u_nom_tgt=None,
    w_x_tgt=1e-6,
    solver_pref_target=("OSQP", "CLARABEL", "SCS"),
    solver_pref_qp=("CLARABEL", "SCS", "ECOS"),
    tol=1e-10,
    box_tol=1e-9,
):
    if not HAS_CVXPY:
        u_rl = np.asarray(u_rl_dev, float).reshape(-1)
        u_min = np.asarray(u_min, float).reshape(-1)
        u_max = np.asarray(u_max, float).reshape(-1)
        u_rl = np.clip(u_rl, u_min, u_max)
        return u_rl, {
            "used": False,
            "filtered": False,
            "method": "no_cvxpy",
            "success": False,
            "target_info": {
                "success": False,
                "selector_mode": "legacy_augstate_rl",
                "selector_name": "legacy_augmented_slack_target",
                "solve_stage": None,
                "requested_y_sp": np.asarray(y_sp, float).reshape(-1),
                "selector_debug": {},
            },
        }

    xhat_aug = np.asarray(xhat_aug, float).reshape(-1)
    y_sp = np.asarray(y_sp, float).reshape(-1)
    u_min = np.asarray(u_min, float).reshape(-1)
    u_max = np.asarray(u_max, float).reshape(-1)
    u_rl = np.asarray(u_rl_dev, float).reshape(-1)
    u_prev_dev = np.asarray(u_prev_dev, float).reshape(-1)
    A_aug = np.asarray(A_aug, float)
    B_aug = np.asarray(B_aug, float)
    C_aug = np.asarray(C_aug, float)
    P_lyap = np.asarray(P_lyap, float)
    S_lyap = np.asarray(S_lyap, float)

    if not (0.0 < float(rho) < 1.0):
        raise ValueError("rho must satisfy 0 < rho < 1.")
    if float(eps_v) < 0.0:
        raise ValueError("eps_v must be >= 0.")

    n_a = A_aug.shape[0]
    n_u = B_aug.shape[1]
    n_y = C_aug.shape[0]

    if A_aug.shape != (n_a, n_a):
        raise ValueError("A_aug must be square.")
    if B_aug.shape[0] != n_a:
        raise ValueError("B_aug row dimension must match A_aug.")
    if C_aug.shape[1] != n_a:
        raise ValueError("C_aug column dimension must match A_aug.")
    if P_lyap.shape != (n_a, n_a):
        raise ValueError("P_lyap shape mismatch.")
    if S_lyap.shape != (n_a, n_a):
        raise ValueError("S_lyap shape mismatch.")
    if xhat_aug.size != n_a:
        raise ValueError("xhat_aug size mismatch.")
    if y_sp.size != n_y:
        raise ValueError("y_sp size mismatch.")
    if u_min.size != n_u or u_max.size != n_u:
        raise ValueError("u_min/u_max size mismatch.")
    if u_rl.size != n_u:
        raise ValueError("u_rl_dev size mismatch.")
    if u_prev_dev.size != n_u:
        raise ValueError("u_prev_dev size mismatch.")
    if np.any(u_min > u_max):
        raise ValueError("u_min must be <= u_max elementwise.")

    u_rl = np.clip(u_rl, u_min, u_max)

    def _diag_sqrt(diag_vals, size, default=1.0, eps=1e-12):
        if diag_vals is None:
            v = default * np.ones(size, dtype=float)
        else:
            v = np.asarray(diag_vals, float).reshape(-1)
            if v.size != size:
                raise ValueError(f"Diagonal size mismatch. Expected {size}, got {v.size}.")
            v = np.maximum(v, eps)
        return np.sqrt(v), v

    qy_sqrt, qy_track_used = _diag_sqrt(Qy_track_diag, n_y, default=1.0)
    rmove_sqrt, rmove_used = _diag_sqrt(Rmove_diag, n_u, default=1.0)

    x_s, u_s, d_s, dbg_tgt = compute_ss_target_slack_from_augmented(
        A_aug=A_aug,
        B_aug=B_aug,
        C_aug=C_aug,
        xhat_aug=xhat_aug,
        y_sp=y_sp,
        u_min=u_min,
        u_max=u_max,
        u_nom=u_nom_tgt,
        Qs_diag=Qs_tgt_diag,
        Ru_diag=Ru_tgt_diag,
        w_x=w_x_tgt,
        solver_pref=solver_pref_target,
        return_debug=True,
    )

    target_info = _legacy_target_info(
        xhat_aug=xhat_aug,
        y_sp=y_sp,
        u_prev_dev=u_prev_dev,
        x_s=x_s,
        u_s=u_s,
        d_s=d_s,
        dbg_tgt=dbg_tgt,
    )

    if not dbg_tgt.get("success", False) or x_s is None or u_s is None:
        return u_rl, {
            "used": False,
            "filtered": False,
            "method": "target_fail",
            "success": False,
            "target_dbg": dbg_tgt,
            "target_info": target_info,
            "u_rl": u_rl.copy(),
            "u_prev_dev": u_prev_dev.copy(),
        }

    x_s_aug = np.concatenate([x_s, d_s], axis=0)
    x_tilde = xhat_aug - x_s_aug
    V_k = lyap_V(x_tilde, P_lyap)
    c = float(rho) * V_k + float(eps_v)

    b_k = A_aug @ x_tilde - B_aug @ u_s
    y0_k = C_aug @ (A_aug @ xhat_aug)
    M = C_aug @ B_aug

    x_tilde_next_rl = b_k + B_aug @ u_rl
    V_next_rl = float(np.sum((S_lyap @ x_tilde_next_rl) ** 2))
    y_pred_next_rl = y0_k + M @ u_rl
    e_pred_next_rl = y_pred_next_rl - y_sp
    track_inf_rl = float(np.max(np.abs(e_pred_next_rl)))

    base_info = {
        "used": True,
        "target_dbg": dbg_tgt,
        "target_info": target_info,
        "rho": float(rho),
        "eps_v": float(eps_v),
        "V_k": V_k,
        "c": c,
        "u_rl": u_rl.copy(),
        "u_prev_dev": u_prev_dev.copy(),
        "u_s": u_s.copy(),
        "x_s": x_s.copy(),
        "d_s": d_s.copy(),
        "y_s": None if target_info.get("y_s") is None else target_info["y_s"].copy(),
        "r_s": target_info["r_s"].copy(),
        "Qy_track_diag": qy_track_used.copy(),
        "Rmove_diag": rmove_used.copy(),
        "margin_rl": float(V_next_rl - c),
    }

    if V_next_rl <= c + tol:
        return u_rl, {
            **base_info,
            "success": True,
            "filtered": False,
            "method": "accept",
            "V_next": V_next_rl,
            "track_inf": track_inf_rl,
            "du_inf": 0.0,
        }

    u = cp.Variable(n_u)
    x_tilde_next = b_k + B_aug @ u
    y_pred_next = y0_k + M @ u
    e_pred_next = y_pred_next - y_sp

    objective = 0.0
    objective += float(w_rl) * cp.sum_squares(u - u_rl)
    objective += float(w_move) * cp.sum_squares(cp.multiply(rmove_sqrt, u - u_prev_dev))
    objective += float(w_track) * cp.sum_squares(cp.multiply(qy_sqrt, e_pred_next))
    objective += float(w_ss) * cp.sum_squares(u - u_s)

    constraints = [
        u >= u_min,
        u <= u_max,
        cp.sum_squares(S_lyap @ x_tilde_next) <= c,
    ]

    problem = cp.Problem(cp.Minimize(objective), constraints)

    solved = False
    last_status = None
    last_solver = None
    last_err = None
    last_box_violation = None
    last_margin_try = None

    for solver_name in solver_pref_qp:
        try:
            u.value = None
            problem.solve(solver=solver_name, warm_start=True, verbose=False)

            last_status = problem.status
            last_solver = solver_name

            if u.value is None or problem.status is None:
                continue

            u_try = np.asarray(u.value).reshape(-1)
            x_tilde_next_try = b_k + B_aug @ u_try
            V_next_try = float(np.sum((S_lyap @ x_tilde_next_try) ** 2))
            margin_try = float(V_next_try - c)

            lower_violation = float(np.max(u_min - u_try))
            upper_violation = float(np.max(u_try - u_max))
            box_violation = max(lower_violation, upper_violation, 0.0)

            last_margin_try = margin_try
            last_box_violation = box_violation

            if problem.status != "optimal":
                continue
            if margin_try > 10.0 * tol:
                continue
            if box_violation > box_tol:
                continue

            solved = True
            break

        except Exception as ex:
            last_err = str(ex)
            last_status = None
            last_solver = solver_name

    if not solved:
        return u_rl, {
            **base_info,
            "success": False,
            "filtered": False,
            "method": "qp_fail",
            "status": last_status,
            "solver": last_solver,
            "err": last_err,
            "V_next": V_next_rl,
            "track_inf": track_inf_rl,
            "qp_margin_try": last_margin_try,
            "qp_box_violation": last_box_violation,
        }

    u_star = np.asarray(u.value).reshape(-1)
    x_tilde_next_star = b_k + B_aug @ u_star
    V_next_star = float(np.sum((S_lyap @ x_tilde_next_star) ** 2))
    margin_star = float(V_next_star - c)

    lower_violation_star = float(np.max(u_min - u_star))
    upper_violation_star = float(np.max(u_star - u_max))
    box_violation_star = max(lower_violation_star, upper_violation_star, 0.0)

    if margin_star > 10.0 * tol or box_violation_star > box_tol:
        return u_rl, {
            **base_info,
            "success": False,
            "filtered": False,
            "method": "qp_fail",
            "status": problem.status,
            "solver": last_solver,
            "err": "accepted_qp_solution_failed_postcheck",
            "V_next": V_next_rl,
            "track_inf": track_inf_rl,
            "qp_margin_try": margin_star,
            "qp_box_violation": box_violation_star,
        }

    y_pred_next_star = y0_k + M @ u_star
    e_pred_next_star = y_pred_next_star - y_sp
    track_inf_star = float(np.max(np.abs(e_pred_next_star)))
    du_inf = float(np.max(np.abs(u_star - u_rl)))
    filtered = bool(du_inf > 1e-10)

    return u_star, {
        **base_info,
        "success": True,
        "filtered": filtered,
        "method": "qp",
        "status": problem.status,
        "solver": last_solver,
        "objective_value": float(problem.value) if problem.value is not None else None,
        "V_next": V_next_star,
        "margin_star": margin_star,
        "box_violation_star": box_violation_star,
        "track_inf": track_inf_star,
        "du_inf": du_inf,
    }
