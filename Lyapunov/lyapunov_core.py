from types import SimpleNamespace

import numpy as np
from scipy.linalg import solve_discrete_are

try:
    import cvxpy as cp

    HAS_CVXPY = True
except Exception:
    HAS_CVXPY = False

from utils.lyapunov_utils import (
    compute_du_sequence,
    diag_psd_from_vector,
    reshape_u_sequence,
    safety_filter_solver_sequence,
    tracking_solver_sequence,
    vector_or_zeros,
)


_OPTIMAL_STATUSES = {"optimal", "optimal_inaccurate"}
_TRACKING_TOL_BY_STATUS = {
    "optimal": 1e-7,
    "optimal_inaccurate": 1e-5,
}


def design_standard_tracking_terminal_ingredients(
    A_aug,
    B_aug,
    C_aug,
    Qy_diag,
    Su_diag=None,
    u_min=None,
    u_max=None,
    lambda_u=1.0,
    qx_eps=1e-10,
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

    A = A_aug[:n_x, :n_x]
    B = B_aug[:n_x, :]
    C = C_aug[:, :n_x]

    Qy = np.diag(np.maximum(Qy_diag, 0.0))
    Qx = C.T @ Qy @ C + float(qx_eps) * np.eye(n_x)

    if Su_diag is None:
        if u_min is None or u_max is None:
            su = lambda_u * np.ones(n_u, dtype=float)
        else:
            u_min = np.asarray(u_min, float).reshape(-1)
            u_max = np.asarray(u_max, float).reshape(-1)
            span = np.maximum(0.5 * (u_max - u_min), eps_r)
            su = lambda_u / np.maximum(span ** 2, eps_r)
    else:
        su = np.maximum(np.asarray(Su_diag, float).reshape(-1), eps_r)

    Su = np.diag(su)
    P_x = solve_discrete_are(A, B, Qx, Su)
    K_x = -np.linalg.solve(Su + B.T @ P_x @ B, B.T @ P_x @ A)

    if return_debug:
        dbg = {
            "A_phys": A.copy(),
            "B_phys": B.copy(),
            "C_phys": C.copy(),
            "Qx": Qx.copy(),
            "Su": Su.copy(),
            "K_x": K_x.copy(),
            "eig_cl": np.linalg.eigvals(A + B @ K_x),
        }
        return P_x, K_x, dbg
    return P_x, K_x


def compute_terminal_alpha_input_only(P_x, K_x, u_s, u_min, u_max, alpha_scale=1.0):
    P_x = np.asarray(P_x, float)
    K_x = np.asarray(K_x, float)
    u_s = np.asarray(u_s, float).reshape(-1)
    u_min = np.asarray(u_min, float).reshape(-1)
    u_max = np.asarray(u_max, float).reshape(-1)

    P_inv = np.linalg.inv(P_x)

    alphas = []
    for row_idx in range(K_x.shape[0]):
        k_row = K_x[row_idx:row_idx + 1, :]
        gamma_sq = float((k_row @ P_inv @ k_row.T).item())
        gamma_sq = max(gamma_sq, 0.0)
        gamma = float(np.sqrt(gamma_sq))

        headroom = float(min(u_max[row_idx] - u_s[row_idx], u_s[row_idx] - u_min[row_idx]))
        if headroom < 0.0:
            return 0.0

        if gamma <= 1e-14:
            alphas.append(np.inf)
        else:
            alphas.append((headroom / gamma) ** 2)

    alpha = float(min(alphas)) if alphas else 0.0
    return max(float(alpha_scale) * alpha, 0.0)


def _bound_violation_inf(value, lower=None, upper=None):
    violation = 0.0
    if lower is not None:
        violation = max(violation, float(np.max(np.maximum(lower - value, 0.0))))
    if upper is not None:
        violation = max(violation, float(np.max(np.maximum(value - upper, 0.0))))
    return violation


def _bounds_to_horizon_matrices(bnds, n_u, horizon_control):
    if bnds is None:
        return None, None

    if len(bnds) == n_u:
        bnds = list(bnds) * int(horizon_control)
    elif len(bnds) != n_u * horizon_control:
        raise ValueError(
            f"bnds must have length {n_u} or {n_u * horizon_control}, got {len(bnds)}."
        )

    lower = np.full((horizon_control, n_u), -np.inf, dtype=float)
    upper = np.full((horizon_control, n_u), np.inf, dtype=float)

    for idx, bound in enumerate(bnds):
        row = idx // n_u
        col = idx % n_u
        lo, hi = bound
        if lo is not None:
            lower[row, col] = float(lo)
        if hi is not None:
            upper[row, col] = float(hi)

    return lower, upper


def _extract_num_iters(problem):
    solver_stats = getattr(problem, "solver_stats", None)
    if solver_stats is None:
        return None
    return getattr(solver_stats, "num_iters", None)


def split_augmented_model(A_aug, B_aug, C_aug):
    A_aug = np.asarray(A_aug, float)
    B_aug = np.asarray(B_aug, float)
    C_aug = np.asarray(C_aug, float)

    n_aug = A_aug.shape[0]
    if A_aug.shape != (n_aug, n_aug):
        raise ValueError("A_aug must be square.")
    if B_aug.ndim != 2 or B_aug.shape[0] != n_aug:
        raise ValueError("B_aug row dimension must match A_aug.")
    if C_aug.ndim != 2 or C_aug.shape[1] != n_aug:
        raise ValueError("C_aug column dimension must match A_aug.")

    n_y = C_aug.shape[0]
    n_x = n_aug - n_y
    if n_x <= 0:
        raise ValueError("Invalid augmented model: inferred physical state dimension must be positive.")

    return {
        "A_phys": A_aug[:n_x, :n_x].copy(),
        "B_phys": B_aug[:n_x, :].copy(),
        "Bd_phys": A_aug[:n_x, n_x:].copy(),
        "C_phys": C_aug[:, :n_x].copy(),
        "Cd_phys": C_aug[:, n_x:].copy(),
        "n_x": int(n_x),
        "n_y": int(n_y),
        "n_u": int(B_aug.shape[1]),
        "n_aug": int(n_aug),
    }


def factor_psd_left(P, neg_tol=1e-10):
    P = np.asarray(P, float)
    P = 0.5 * (P + P.T)
    eigvals, eigvecs = np.linalg.eigh(P)
    min_eig = float(np.min(eigvals))
    if min_eig < -float(neg_tol):
        raise ValueError(f"P is not positive semidefinite. Min eigenvalue = {min_eig:.3e}.")

    eigvals = np.where(eigvals > 0.0, eigvals, 0.0)
    return np.diag(np.sqrt(eigvals)) @ eigvecs.T


def design_lyapunov_filter_ingredients(
    A_aug,
    B_aug,
    C_aug,
    Qy_diag,
    Ru_diag=None,
    u_min=None,
    u_max=None,
    u_nom=None,
    lambda_u=1.0,
    qx_eps=1e-10,
    eps_r=1e-9,
    return_debug=False,
):
    model = split_augmented_model(A_aug, B_aug, C_aug)
    A = model["A_phys"]
    B = model["B_phys"]
    C = model["C_phys"]
    n_u = model["n_u"]
    n_y = model["n_y"]

    Qy_diag = np.asarray(Qy_diag, float).reshape(-1)
    if Qy_diag.size != n_y:
        raise ValueError(f"Qy_diag size mismatch. Expected {n_y}, got {Qy_diag.size}.")

    if u_nom is None:
        u_nom = np.zeros(n_u, dtype=float)
    else:
        u_nom = np.asarray(u_nom, float).reshape(-1)
        if u_nom.size != n_u:
            raise ValueError(f"u_nom size mismatch. Expected {n_u}, got {u_nom.size}.")

    Qy = np.diag(np.maximum(Qy_diag, 0.0))
    Qx = C.T @ Qy @ C + float(qx_eps) * np.eye(model["n_x"], dtype=float)
    Qx = 0.5 * (Qx + Qx.T)

    if Ru_diag is None:
        if u_min is None or u_max is None:
            ru_diag = float(lambda_u) * np.ones(n_u, dtype=float)
        else:
            u_min = np.asarray(u_min, float).reshape(-1)
            u_max = np.asarray(u_max, float).reshape(-1)
            if u_min.size != n_u or u_max.size != n_u:
                raise ValueError("u_min/u_max size mismatch.")
            span = np.maximum(np.minimum(u_nom - u_min, u_max - u_nom), eps_r)
            ru_diag = float(lambda_u) / np.maximum(span ** 2, eps_r)
    else:
        ru_diag = np.asarray(Ru_diag, float).reshape(-1)
        if ru_diag.size != n_u:
            raise ValueError(f"Ru_diag size mismatch. Expected {n_u}, got {ru_diag.size}.")
        ru_diag = np.maximum(ru_diag, eps_r)

    R_lyap = np.diag(np.maximum(ru_diag, eps_r))
    P_x = solve_discrete_are(A, B, Qx, R_lyap)
    P_x = 0.5 * (P_x + P_x.T)
    K_x = -np.linalg.solve(R_lyap + B.T @ P_x @ B, B.T @ P_x @ A)
    S_x = factor_psd_left(P_x)

    ingredients = {
        **model,
        "Qy_diag": Qy_diag.copy(),
        "Qx": Qx.copy(),
        "R_lyap_diag": ru_diag.copy(),
        "R_lyap": R_lyap.copy(),
        "u_nom": u_nom.copy(),
        "P_x": P_x.copy(),
        "K_x": K_x.copy(),
        "S_x": S_x.copy(),
        "eig_cl": np.linalg.eigvals(A + B @ K_x),
    }

    if return_debug:
        dbg = {
            "A_phys": A.copy(),
            "B_phys": B.copy(),
            "C_phys": C.copy(),
            "Qx": Qx.copy(),
            "R_lyap": R_lyap.copy(),
            "P_x": P_x.copy(),
            "K_x": K_x.copy(),
            "eig_cl": ingredients["eig_cl"].copy(),
        }
        return ingredients, dbg
    return ingredients


def physical_error_from_target(xhat_aug, target_info):
    xhat_aug = np.asarray(xhat_aug, float).reshape(-1)
    if target_info is None or not target_info.get("success", False):
        raise ValueError("target_info must contain a successful target package.")

    x_s = np.asarray(target_info["x_s"], float).reshape(-1)
    n_x = x_s.size
    if xhat_aug.size < n_x:
        raise ValueError("xhat_aug is shorter than the physical target state.")
    return xhat_aug[:n_x] - x_s


def predict_next_physical_error(ingredients, e_x, u, u_s):
    A = np.asarray(ingredients["A_phys"], float)
    B = np.asarray(ingredients["B_phys"], float)
    e_x = np.asarray(e_x, float).reshape(-1)
    u = np.asarray(u, float).reshape(-1)
    u_s = np.asarray(u_s, float).reshape(-1)
    return A @ e_x + B @ (u - u_s)


def lyapunov_value(e_x, P_x):
    e_x = np.asarray(e_x, float).reshape(-1, 1)
    P_x = np.asarray(P_x, float)
    return float((e_x.T @ P_x @ e_x).item())


def lyapunov_bound(V_k, rho, eps_lyap):
    rho = float(rho)
    eps_lyap = float(eps_lyap)
    if not (0.0 < rho < 1.0):
        raise ValueError("rho must satisfy 0 < rho < 1.")
    if eps_lyap < 0.0:
        raise ValueError("eps_lyap must be nonnegative.")
    return rho * float(V_k) + eps_lyap


def first_step_contraction_metrics(x0_aug, x_pred, x_s, P_x, rho, eps_lyap, tol=1e-9):
    x0_aug = np.asarray(x0_aug, float).reshape(-1)
    x_pred = np.asarray(x_pred, float)
    x_s = np.asarray(x_s, float).reshape(-1)
    P_x = np.asarray(P_x, float)
    n_x = x_s.size

    if x0_aug.size < n_x:
        raise ValueError("x0_aug is shorter than x_s.")
    if x_pred.ndim != 2 or x_pred.shape[0] < n_x or x_pred.shape[1] < 2:
        raise ValueError("x_pred must contain at least the initial and first predicted states.")

    e_x = x0_aug[:n_x] - x_s
    e_x_next_first = x_pred[:n_x, 1] - x_s
    V_k = lyapunov_value(e_x, P_x)
    V_next_first = lyapunov_value(e_x_next_first, P_x)
    V_bound = lyapunov_bound(V_k, rho=rho, eps_lyap=eps_lyap)
    contraction_margin = float(V_next_first - V_bound)
    first_step_contraction_satisfied = bool(contraction_margin <= float(tol))
    contraction_constraint_violation = max(contraction_margin, 0.0)

    return {
        "e_x": e_x.copy(),
        "e_x_next_first": e_x_next_first.copy(),
        "V_k": float(V_k),
        "V_next_first": float(V_next_first),
        "V_bound": float(V_bound),
        "contraction_margin": float(contraction_margin),
        "first_step_contraction_satisfied": bool(first_step_contraction_satisfied),
        "contraction_constraint_violation": float(contraction_constraint_violation),
        "rho_lyap": float(rho),
        "eps_lyap": float(eps_lyap),
    }


def _bound_ok(value, lower=None, upper=None, tol=1e-9):
    violation = _bound_violation_inf(value, lower=lower, upper=upper)
    return bool(violation <= float(tol)), float(violation)


def evaluate_candidate_action(
    u_cand,
    xhat_aug,
    target_info,
    ingredients,
    rho=0.99,
    eps_lyap=1e-9,
    u_min=None,
    u_max=None,
    u_prev=None,
    du_min=None,
    du_max=None,
    tol=1e-9,
):
    if target_info is None or not target_info.get("success", False):
        return {
            "accepted": False,
            "accept_reason": None,
            "reject_reason": "target_unavailable",
            "candidate_bounds_ok": False,
            "candidate_move_ok": False,
            "candidate_lyap_ok": False,
        }

    u_cand = np.asarray(u_cand, float).reshape(-1)
    u_s = np.asarray(target_info["u_s"], float).reshape(-1)
    d_s = np.asarray(target_info["d_s"], float).reshape(-1)
    x_s = np.asarray(target_info["x_s"], float).reshape(-1)
    e_x = physical_error_from_target(xhat_aug, target_info)
    e_u = u_cand - u_s

    e_x_next = predict_next_physical_error(ingredients, e_x=e_x, u=u_cand, u_s=u_s)
    V_k = lyapunov_value(e_x, ingredients["P_x"])
    V_next_cand = lyapunov_value(e_x_next, ingredients["P_x"])
    V_bound = lyapunov_bound(V_k, rho=rho, eps_lyap=eps_lyap)
    lyap_margin = float(V_next_cand - V_bound)
    candidate_lyap_ok = bool(lyap_margin <= float(tol))

    bounds_ok, bounds_violation = _bound_ok(u_cand, lower=u_min, upper=u_max, tol=tol)

    if u_prev is None or (du_min is None and du_max is None):
        candidate_move_ok = True
        move_violation = 0.0
    else:
        delta_u = u_cand - np.asarray(u_prev, float).reshape(-1)
        candidate_move_ok, move_violation = _bound_ok(delta_u, lower=du_min, upper=du_max, tol=tol)

    C = np.asarray(ingredients["C_phys"], float)
    Cd = np.asarray(ingredients["Cd_phys"], float)
    x_next_pred = x_s + e_x_next
    y_next_pred = C @ x_next_pred + Cd @ d_s
    y_s = np.asarray(target_info["y_s"], float).reshape(-1)

    accepted = bool(bounds_ok and candidate_move_ok and candidate_lyap_ok)
    accept_reason = "candidate_ok" if accepted else None
    if accepted:
        reject_reason = None
    elif not bounds_ok:
        reject_reason = "input_bounds"
    elif not candidate_move_ok:
        reject_reason = "move_bounds"
    else:
        reject_reason = "lyapunov"

    return {
        "accepted": accepted,
        "accept_reason": accept_reason,
        "reject_reason": reject_reason,
        "candidate_bounds_ok": bool(bounds_ok),
        "candidate_move_ok": bool(candidate_move_ok),
        "candidate_lyap_ok": bool(candidate_lyap_ok),
        "u_cand": u_cand.copy(),
        "u_s": u_s.copy(),
        "x_s": x_s.copy(),
        "d_s": d_s.copy(),
        "e_x": e_x.copy(),
        "e_u": e_u.copy(),
        "e_x_next_pred": e_x_next.copy(),
        "y_next_pred": y_next_pred.copy(),
        "y_s": y_s.copy(),
        "V_k": float(V_k),
        "V_next_cand": float(V_next_cand),
        "V_bound": float(V_bound),
        "lyap_margin": float(lyap_margin),
        "candidate_bounds_violation": float(bounds_violation),
        "candidate_move_violation": float(move_violation),
        "rho": float(rho),
        "eps_lyap": float(eps_lyap),
    }


class StandardTrackingLyapunovMpcRawlingsTargetSolver:
    def __init__(
        self,
        A_aug,
        B_aug,
        C_aug,
        Qy_diag,
        Su_diag,
        NP,
        NC,
        P_x,
        K_x,
        Rdu_diag=None,
        terminal_set_on=True,
        terminal_alpha_scale=1.0,
        terminal_cost_scale=1.0,
        D=None,
        solver_pref_qp=None,
        solver_pref_conic=None,
    ):
        self.A = np.asarray(A_aug, float)
        self.B = np.asarray(B_aug, float)
        self.C = np.asarray(C_aug, float)
        self.D = None if D is None else np.asarray(D, float)

        self.Qy = np.asarray(Qy_diag, float).reshape(-1)
        self.Su = np.asarray(Su_diag, float).reshape(-1)
        self.Rdu = None if Rdu_diag is None else np.asarray(Rdu_diag, float).reshape(-1)

        self.NP = int(NP)
        self.NC = int(NC)

        self.P_x = np.asarray(P_x, float)
        self.K_x = np.asarray(K_x, float)
        self.terminal_set_on = bool(terminal_set_on)
        self.terminal_alpha_scale = float(terminal_alpha_scale)
        self.terminal_cost_scale = float(terminal_cost_scale)

        self.n_aug = self.A.shape[0]
        self.n_u = self.B.shape[1]
        self.n_y = self.C.shape[0]
        self.n_x = self.n_aug - self.n_y

        self.A_phys = self.A[:self.n_x, :self.n_x]
        self.B_phys = self.B[:self.n_x, :]
        self.C_phys = self.C[:, :self.n_x]

        self.Qy_mat = np.diag(np.maximum(self.Qy, 0.0))
        self.Su_mat = np.diag(np.maximum(self.Su, 0.0))
        self.Rdu_mat = None if self.Rdu is None else np.diag(np.maximum(self.Rdu, 0.0))

        self.solver_pref_qp = solver_pref_qp
        self.solver_pref_conic = solver_pref_conic

        if self.P_x.shape != (self.n_x, self.n_x):
            raise ValueError("P_x shape mismatch.")
        if self.K_x.shape != (self.n_u, self.n_x):
            raise ValueError("K_x shape mismatch.")
        if self.Qy.size != self.n_y:
            raise ValueError("Qy_diag size mismatch.")
        if self.Su.size != self.n_u:
            raise ValueError("Su_diag size mismatch.")
        if self.Rdu is not None and self.Rdu.size != self.n_u:
            raise ValueError("Rdu_diag size mismatch.")

    def _predict_from_sequence(self, u_sequence, x0_aug):
        x0_aug = np.asarray(x0_aug, float).reshape(-1)
        x_pred = np.zeros((self.n_aug, self.NP + 1), dtype=float)
        x_pred[:, 0] = x0_aug
        for step_idx in range(self.NP):
            ctrl_idx = step_idx if step_idx < self.NC else self.NC - 1
            x_pred[:, step_idx + 1] = self.A @ x_pred[:, step_idx] + self.B @ u_sequence[ctrl_idx, :]

        y_pred = self.C @ x_pred
        if self.D is not None:
            for step_idx in range(self.NP):
                ctrl_idx = step_idx if step_idx < self.NC else self.NC - 1
                y_pred[:, step_idx + 1] = y_pred[:, step_idx + 1] + self.D @ u_sequence[ctrl_idx, :]
        return x_pred, y_pred

    def _tracking_cost(self, u_sequence, y_target, u_prev_dev, x0_aug, x_s, u_s):
        y_target = np.asarray(y_target, float).reshape(self.n_y)
        u_prev_dev = np.asarray(u_prev_dev, float).reshape(self.n_u)
        x_s = np.asarray(x_s, float).reshape(self.n_x)
        u_s = np.asarray(u_s, float).reshape(self.n_u)

        x_pred, y_pred = self._predict_from_sequence(u_sequence, x0_aug)
        y_err = y_pred[:, 1:] - y_target[:, None]
        du = compute_du_sequence(u_sequence, u_prev_dev)
        u_err = u_sequence - u_s.reshape(1, -1)

        obj = 0.0
        for output_idx in range(self.n_y):
            obj += float(self.Qy[output_idx]) * float(np.sum(y_err[output_idx, :] ** 2))
        for input_idx in range(self.n_u):
            obj += float(self.Su[input_idx]) * float(np.sum(u_err[:, input_idx] ** 2))
        if self.Rdu is not None:
            for input_idx in range(self.n_u):
                obj += float(self.Rdu[input_idx]) * float(np.sum(du[:, input_idx] ** 2))

        e_terminal = x_pred[:self.n_x, -1] - x_s
        obj += self.terminal_cost_scale * float(e_terminal.T @ self.P_x @ e_terminal)
        return float(obj)

    def _terminal_value_from_sequence(self, u_sequence, x0_aug, x_s):
        x_pred, _ = self._predict_from_sequence(u_sequence, x0_aug)
        e_terminal = x_pred[:self.n_x, -1] - np.asarray(x_s, float).reshape(self.n_x)
        return float(e_terminal.T @ self.P_x @ e_terminal)

    def _terminal_value(self, x_opt, x0_aug, x_s):
        u_sequence = reshape_u_sequence(x_opt, self.n_u, self.NC)
        return self._terminal_value_from_sequence(u_sequence, x0_aug=x0_aug, x_s=x_s)

    def _evaluate_tracking_solution(
        self,
        u_sequence,
        x_pred,
        x0_aug,
        x_s,
        lower,
        upper,
        alpha_terminal,
        terminal_constraint_active,
        status,
    ):
        tol = _TRACKING_TOL_BY_STATUS.get(status)
        result = {
            "accepted": False,
            "reject_reason": None,
            "dyn_residual_inf": None,
            "initial_residual_inf": None,
            "bound_violation_inf": None,
            "terminal_value": None,
            "terminal_constraint_violation": None,
        }

        if tol is None:
            result["reject_reason"] = "solver_status"
            return result

        x0_aug = np.asarray(x0_aug, float).reshape(-1)
        x_s = np.asarray(x_s, float).reshape(-1)
        u_sequence = np.asarray(u_sequence, float)
        x_pred = np.asarray(x_pred, float)

        initial_residual_inf = float(np.max(np.abs(x_pred[:, 0] - x0_aug)))
        dyn_residual_inf = initial_residual_inf
        for step_idx in range(self.NP):
            ctrl_idx = step_idx if step_idx < self.NC else self.NC - 1
            residual = x_pred[:, step_idx + 1] - (self.A @ x_pred[:, step_idx] + self.B @ u_sequence[ctrl_idx, :])
            dyn_residual_inf = max(dyn_residual_inf, float(np.max(np.abs(residual))))

        bound_violation_inf = _bound_violation_inf(u_sequence, lower=lower, upper=upper)
        terminal_value = self._terminal_value_from_sequence(u_sequence, x0_aug=x0_aug, x_s=x_s)
        terminal_constraint_violation = 0.0
        if terminal_constraint_active:
            terminal_constraint_violation = max(float(terminal_value) - float(alpha_terminal), 0.0)

        if dyn_residual_inf > tol:
            result["reject_reason"] = "dyn_residual"
        elif bound_violation_inf > tol:
            result["reject_reason"] = "bound_violation"
        elif terminal_constraint_active and terminal_constraint_violation > tol:
            result["reject_reason"] = "terminal_constraint"
        else:
            result["accepted"] = True

        result.update({
            "dyn_residual_inf": dyn_residual_inf,
            "initial_residual_inf": initial_residual_inf,
            "bound_violation_inf": float(bound_violation_inf),
            "terminal_value": float(terminal_value),
            "terminal_constraint_violation": float(terminal_constraint_violation),
        })
        return result

    def standard_tracking_report(self, x_opt, x0_aug, x_s, u_s, y_target, u_prev_dev, alpha_terminal):
        u_sequence = reshape_u_sequence(x_opt, self.n_u, self.NC)
        du = compute_du_sequence(u_sequence, u_prev_dev)
        x_pred, y_pred = self._predict_from_sequence(u_sequence, x0_aug)

        x_s = np.asarray(x_s, float).reshape(self.n_x)
        u_s = np.asarray(u_s, float).reshape(self.n_u)
        y_target = np.asarray(y_target, float).reshape(self.n_y)

        e_x_path = x_pred[:self.n_x, :] - x_s[:, None]
        e_y_path = y_pred - y_target[:, None]
        u_err_path = u_sequence - u_s.reshape(1, -1)
        terminal_value = float(e_x_path[:, -1].T @ self.P_x @ e_x_path[:, -1])

        if alpha_terminal is None:
            terminal_margin = None
            terminal_set_violated = False
            alpha_terminal_used = None
        else:
            alpha_terminal_used = float(alpha_terminal)
            terminal_margin = float(alpha_terminal_used - terminal_value)
            terminal_set_violated = bool(terminal_value > alpha_terminal_used + 1e-8)

        return {
            "x_pred_path": x_pred.copy(),
            "y_pred_path": y_pred.copy(),
            "e_x_path": e_x_path.copy(),
            "e_y_path": e_y_path.copy(),
            "u_seq_opt": u_sequence.copy(),
            "du_seq_opt": du.copy(),
            "u_err_path": u_err_path.copy(),
            "terminal_value": terminal_value,
            "alpha_terminal_used": alpha_terminal_used,
            "terminal_margin": terminal_margin,
            "terminal_set_violated": terminal_set_violated,
        }

    def solve_tracking_mpc_step(
        self,
        IC_opt,
        bnds,
        y_target,
        u_prev_dev,
        x0_aug,
        x_s,
        u_s,
        alpha_terminal,
        options=None,
    ):
        if not HAS_CVXPY:
            raise ImportError("CVXPY is required for the standard Lyapunov tracking MPC solver.")

        options = {} if options is None else dict(options)
        solver_pref_override = options.pop("solver_pref", None)
        warm_start = bool(options.pop("warm_start", True))
        verbose = bool(options.pop("verbose", False))
        solve_kwargs = dict(options.pop("solve_kwargs", {}))

        x0_aug = np.asarray(x0_aug, float).reshape(-1)
        x_s = np.asarray(x_s, float).reshape(self.n_x)
        u_s = np.asarray(u_s, float).reshape(self.n_u)
        y_target = np.asarray(y_target, float).reshape(self.n_y)
        u_prev_dev = np.asarray(u_prev_dev, float).reshape(self.n_u)
        lower, upper = _bounds_to_horizon_matrices(bnds, self.n_u, self.NC)

        active_terminal_constraint = (
            self.terminal_set_on
            and alpha_terminal is not None
            and np.isfinite(float(alpha_terminal))
        )

        u_var = cp.Variable((self.NC, self.n_u))
        x_var = cp.Variable((self.n_aug, self.NP + 1))

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
        for step_idx in range(self.NP):
            ctrl_idx = step_idx if step_idx < self.NC else self.NC - 1
            constraints.append(
                x_var[:, step_idx + 1] == self.A @ x_var[:, step_idx] + self.B @ u_var[ctrl_idx, :]
            )

            y_expr = self.C @ x_var[:, step_idx + 1]
            if self.D is not None:
                y_expr = y_expr + self.D @ u_var[ctrl_idx, :]
            objective += cp.quad_form(y_expr - y_target, self.Qy_mat)

        for ctrl_idx in range(self.NC):
            objective += cp.quad_form(u_var[ctrl_idx, :] - u_s, self.Su_mat)

        if self.Rdu_mat is not None:
            objective += cp.quad_form(u_var[0, :] - u_prev_dev, self.Rdu_mat)
            for ctrl_idx in range(1, self.NC):
                objective += cp.quad_form(u_var[ctrl_idx, :] - u_var[ctrl_idx - 1, :], self.Rdu_mat)

        terminal_error = x_var[:self.n_x, self.NP] - x_s
        terminal_value_expr = cp.quad_form(terminal_error, self.P_x)
        objective += self.terminal_cost_scale * terminal_value_expr
        if active_terminal_constraint:
            constraints.append(terminal_value_expr <= float(alpha_terminal))

        problem = cp.Problem(cp.Minimize(objective), constraints)

        ic_flat = np.asarray(IC_opt, float).reshape(-1)
        if ic_flat.size == self.n_u * self.NC:
            try:
                u_guess = reshape_u_sequence(ic_flat, self.n_u, self.NC)
                u_var.value = u_guess
                x_guess, _ = self._predict_from_sequence(u_guess, x0_aug)
                x_var.value = x_guess
            except Exception:
                pass

        if solver_pref_override is None:
            solver_pref = self.solver_pref_conic if active_terminal_constraint else self.solver_pref_qp
        else:
            solver_pref = solver_pref_override
        solver_sequence = tracking_solver_sequence(
            active_terminal_constraint,
            solver_pref=solver_pref,
        )

        last_status = None
        last_solver = None
        last_error = None
        last_objective = None
        last_nit = None
        last_eval = None

        for solver_name in solver_sequence:
            try:
                problem.solve(
                    solver=solver_name,
                    warm_start=warm_start,
                    verbose=verbose,
                    **solve_kwargs,
                )
                last_status = problem.status
                last_solver = solver_name
                last_nit = _extract_num_iters(problem)
                if problem.value is not None:
                    last_objective = float(problem.value)

                if u_var.value is None or x_var.value is None:
                    continue

                u_value = np.asarray(u_var.value, float)
                x_value = np.asarray(x_var.value, float)
                last_eval = self._evaluate_tracking_solution(
                    u_sequence=u_value,
                    x_pred=x_value,
                    x0_aug=x0_aug,
                    x_s=x_s,
                    lower=lower,
                    upper=upper,
                    alpha_terminal=alpha_terminal,
                    terminal_constraint_active=active_terminal_constraint,
                    status=problem.status,
                )
                if problem.status in _OPTIMAL_STATUSES and last_eval["accepted"]:
                    return SimpleNamespace(
                        success=True,
                        x=u_value.reshape(-1),
                        status=problem.status,
                        message="optimal",
                        fun=last_objective,
                        nit=last_nit,
                        solver=solver_name,
                        error=None,
                        objective_value=last_objective,
                        dyn_residual_inf=last_eval["dyn_residual_inf"],
                        bound_violation_inf=last_eval["bound_violation_inf"],
                        terminal_value=last_eval["terminal_value"],
                        terminal_constraint_violation=last_eval["terminal_constraint_violation"],
                    )
            except Exception as exc:
                last_error = repr(exc)

        reject_reason = None if last_eval is None else last_eval.get("reject_reason")
        if reject_reason is None and last_error is not None:
            reject_reason = "solver_error"
        if reject_reason is None:
            reject_reason = "solver_status"

        return SimpleNamespace(
            success=False,
            x=None,
            status=last_status,
            message=reject_reason,
            fun=last_objective,
            nit=last_nit,
            solver=last_solver,
            error=last_error,
            objective_value=last_objective,
            dyn_residual_inf=None if last_eval is None else last_eval["dyn_residual_inf"],
            bound_violation_inf=None if last_eval is None else last_eval["bound_violation_inf"],
            terminal_value=None if last_eval is None else last_eval["terminal_value"],
            terminal_constraint_violation=None
            if last_eval is None
            else last_eval["terminal_constraint_violation"],
        )


class FirstStepContractionTrackingLyapunovMpcSolver(StandardTrackingLyapunovMpcRawlingsTargetSolver):
    def _evaluate_tracking_solution_with_contraction(
        self,
        u_sequence,
        x_pred,
        x0_aug,
        x_s,
        lower,
        upper,
        alpha_terminal,
        terminal_constraint_active,
        first_step_contraction_on,
        rho_lyap,
        eps_lyap,
        status,
    ):
        result = super()._evaluate_tracking_solution(
            u_sequence=u_sequence,
            x_pred=x_pred,
            x0_aug=x0_aug,
            x_s=x_s,
            lower=lower,
            upper=upper,
            alpha_terminal=alpha_terminal,
            terminal_constraint_active=terminal_constraint_active,
            status=status,
        )
        result.update({
            "V_k": None,
            "V_next_first": None,
            "V_bound": None,
            "contraction_margin": None,
            "first_step_contraction_satisfied": None,
            "contraction_constraint_violation": None,
            "rho_lyap": None if rho_lyap is None else float(rho_lyap),
            "eps_lyap": None if eps_lyap is None else float(eps_lyap),
        })

        if not first_step_contraction_on:
            return result

        contraction = first_step_contraction_metrics(
            x0_aug=x0_aug,
            x_pred=x_pred,
            x_s=x_s,
            P_x=self.P_x,
            rho=rho_lyap,
            eps_lyap=eps_lyap,
            tol=_TRACKING_TOL_BY_STATUS.get(status, 1e-5),
        )
        result.update(contraction)
        if result["accepted"] and not contraction["first_step_contraction_satisfied"]:
            result["accepted"] = False
            result["reject_reason"] = "first_step_contraction"
        return result

    def standard_tracking_report(
        self,
        x_opt,
        x0_aug,
        x_s,
        u_s,
        y_target,
        u_prev_dev,
        alpha_terminal,
        rho_lyap=None,
        eps_lyap=None,
        first_step_contraction_on=False,
    ):
        report = super().standard_tracking_report(
            x_opt=x_opt,
            x0_aug=x0_aug,
            x_s=x_s,
            u_s=u_s,
            y_target=y_target,
            u_prev_dev=u_prev_dev,
            alpha_terminal=alpha_terminal,
        )
        report.update({
            "V_k": None,
            "V_next_first": None,
            "V_bound": None,
            "contraction_margin": None,
            "first_step_contraction_satisfied": None,
            "contraction_constraint_violation": None,
            "rho_lyap": None if rho_lyap is None else float(rho_lyap),
            "eps_lyap": None if eps_lyap is None else float(eps_lyap),
        })

        if first_step_contraction_on:
            contraction = first_step_contraction_metrics(
                x0_aug=x0_aug,
                x_pred=report["x_pred_path"],
                x_s=x_s,
                P_x=self.P_x,
                rho=rho_lyap,
                eps_lyap=eps_lyap,
            )
            report.update(contraction)
        return report

    def solve_tracking_mpc_step(
        self,
        IC_opt,
        bnds,
        y_target,
        u_prev_dev,
        x0_aug,
        x_s,
        u_s,
        alpha_terminal,
        rho_lyap=0.99,
        eps_lyap=1e-9,
        first_step_contraction_on=True,
        options=None,
    ):
        if not HAS_CVXPY:
            raise ImportError("CVXPY is required for the standard Lyapunov tracking MPC solver.")

        options = {} if options is None else dict(options)
        solver_pref_override = options.pop("solver_pref", None)
        warm_start = bool(options.pop("warm_start", True))
        verbose = bool(options.pop("verbose", False))
        solve_kwargs = dict(options.pop("solve_kwargs", {}))

        x0_aug = np.asarray(x0_aug, float).reshape(-1)
        x_s = np.asarray(x_s, float).reshape(self.n_x)
        u_s = np.asarray(u_s, float).reshape(self.n_u)
        y_target = np.asarray(y_target, float).reshape(self.n_y)
        u_prev_dev = np.asarray(u_prev_dev, float).reshape(self.n_u)
        lower, upper = _bounds_to_horizon_matrices(bnds, self.n_u, self.NC)

        active_terminal_constraint = (
            self.terminal_set_on
            and alpha_terminal is not None
            and np.isfinite(float(alpha_terminal))
        )
        active_first_step_contraction = bool(first_step_contraction_on)

        u_var = cp.Variable((self.NC, self.n_u))
        x_var = cp.Variable((self.n_aug, self.NP + 1))

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
        for step_idx in range(self.NP):
            ctrl_idx = step_idx if step_idx < self.NC else self.NC - 1
            constraints.append(
                x_var[:, step_idx + 1] == self.A @ x_var[:, step_idx] + self.B @ u_var[ctrl_idx, :]
            )

            y_expr = self.C @ x_var[:, step_idx + 1]
            if self.D is not None:
                y_expr = y_expr + self.D @ u_var[ctrl_idx, :]
            objective += cp.quad_form(y_expr - y_target, self.Qy_mat)

        for ctrl_idx in range(self.NC):
            objective += cp.quad_form(u_var[ctrl_idx, :] - u_s, self.Su_mat)

        if self.Rdu_mat is not None:
            objective += cp.quad_form(u_var[0, :] - u_prev_dev, self.Rdu_mat)
            for ctrl_idx in range(1, self.NC):
                objective += cp.quad_form(u_var[ctrl_idx, :] - u_var[ctrl_idx - 1, :], self.Rdu_mat)

        terminal_error = x_var[:self.n_x, self.NP] - x_s
        terminal_value_expr = cp.quad_form(terminal_error, self.P_x)
        objective += self.terminal_cost_scale * terminal_value_expr
        if active_terminal_constraint:
            constraints.append(terminal_value_expr <= float(alpha_terminal))

        if active_first_step_contraction:
            V_k = lyapunov_value(x0_aug[:self.n_x] - x_s, self.P_x)
            V_bound = float(lyapunov_bound(V_k, rho=rho_lyap, eps_lyap=eps_lyap))
            first_step_error = x_var[:self.n_x, 1] - x_s
            first_step_value_expr = cp.quad_form(first_step_error, self.P_x)
            constraints.append(first_step_value_expr <= V_bound)

        problem = cp.Problem(cp.Minimize(objective), constraints)

        ic_flat = np.asarray(IC_opt, float).reshape(-1)
        if ic_flat.size == self.n_u * self.NC:
            try:
                u_guess = reshape_u_sequence(ic_flat, self.n_u, self.NC)
                u_var.value = u_guess
                x_guess, _ = self._predict_from_sequence(u_guess, x0_aug)
                x_var.value = x_guess
            except Exception:
                pass

        if solver_pref_override is None:
            needs_conic = bool(active_terminal_constraint or active_first_step_contraction)
            solver_pref = self.solver_pref_conic if needs_conic else self.solver_pref_qp
        else:
            solver_pref = solver_pref_override
        solver_sequence = tracking_solver_sequence(
            bool(active_terminal_constraint or active_first_step_contraction),
            solver_pref=solver_pref,
        )

        last_status = None
        last_solver = None
        last_error = None
        last_objective = None
        last_nit = None
        last_eval = None

        for solver_name in solver_sequence:
            try:
                problem.solve(
                    solver=solver_name,
                    warm_start=warm_start,
                    verbose=verbose,
                    **solve_kwargs,
                )
                last_status = problem.status
                last_solver = solver_name
                last_nit = _extract_num_iters(problem)
                if problem.value is not None:
                    last_objective = float(problem.value)

                if u_var.value is None or x_var.value is None:
                    continue

                u_value = np.asarray(u_var.value, float)
                x_value = np.asarray(x_var.value, float)
                last_eval = self._evaluate_tracking_solution_with_contraction(
                    u_sequence=u_value,
                    x_pred=x_value,
                    x0_aug=x0_aug,
                    x_s=x_s,
                    lower=lower,
                    upper=upper,
                    alpha_terminal=alpha_terminal,
                    terminal_constraint_active=active_terminal_constraint,
                    first_step_contraction_on=active_first_step_contraction,
                    rho_lyap=rho_lyap,
                    eps_lyap=eps_lyap,
                    status=problem.status,
                )
                if problem.status in _OPTIMAL_STATUSES and last_eval["accepted"]:
                    return SimpleNamespace(
                        success=True,
                        x=u_value.reshape(-1),
                        status=problem.status,
                        message="optimal",
                        fun=last_objective,
                        nit=last_nit,
                        solver=solver_name,
                        error=None,
                        objective_value=last_objective,
                        dyn_residual_inf=last_eval["dyn_residual_inf"],
                        bound_violation_inf=last_eval["bound_violation_inf"],
                        terminal_value=last_eval["terminal_value"],
                        terminal_constraint_violation=last_eval["terminal_constraint_violation"],
                        V_k=last_eval["V_k"],
                        V_next_first=last_eval["V_next_first"],
                        V_bound=last_eval["V_bound"],
                        contraction_margin=last_eval["contraction_margin"],
                        first_step_contraction_satisfied=last_eval["first_step_contraction_satisfied"],
                        contraction_constraint_violation=last_eval["contraction_constraint_violation"],
                    )
            except Exception as exc:
                last_error = repr(exc)

        reject_reason = None if last_eval is None else last_eval.get("reject_reason")
        if reject_reason is None and last_error is not None:
            reject_reason = "solver_error"
        if reject_reason is None:
            reject_reason = "solver_status"

        return SimpleNamespace(
            success=False,
            x=None,
            status=last_status,
            message=reject_reason,
            fun=last_objective,
            nit=last_nit,
            solver=last_solver,
            error=last_error,
            objective_value=last_objective,
            dyn_residual_inf=None if last_eval is None else last_eval["dyn_residual_inf"],
            bound_violation_inf=None if last_eval is None else last_eval["bound_violation_inf"],
            terminal_value=None if last_eval is None else last_eval["terminal_value"],
            terminal_constraint_violation=None
            if last_eval is None
            else last_eval["terminal_constraint_violation"],
            V_k=None if last_eval is None else last_eval["V_k"],
            V_next_first=None if last_eval is None else last_eval["V_next_first"],
            V_bound=None if last_eval is None else last_eval["V_bound"],
            contraction_margin=None if last_eval is None else last_eval["contraction_margin"],
            first_step_contraction_satisfied=None
            if last_eval is None
            else last_eval["first_step_contraction_satisfied"],
            contraction_constraint_violation=None
            if last_eval is None
            else last_eval["contraction_constraint_violation"],
        )
