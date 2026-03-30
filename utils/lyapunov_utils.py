import numpy as np


DEFAULT_CVXPY_SOLVERS = ("OSQP", "CLARABEL", "SCS")
DEFAULT_TRACKING_QP_SOLVERS = ("CLARABEL", "SCS", "OSQP")
DEFAULT_TRACKING_CONIC_SOLVERS = ("CLARABEL", "SCS")
DEFAULT_SAFETY_FILTER_QP_SOLVERS = ("OSQP", "CLARABEL", "SCS")
DEFAULT_SAFETY_FILTER_QCQP_SOLVERS = ("CLARABEL", "SCS")


def diag_psd_from_vector(diag_vals, size, eps=1e-12, default=1.0):
    if diag_vals is None:
        vals = default * np.ones(size, dtype=float)
    else:
        vals = np.asarray(diag_vals, float).reshape(-1)
        if vals.size == 1 and size > 1:
            vals = np.full(size, float(vals.item()), dtype=float)
        if vals.size != size:
            raise ValueError(f"Diagonal size mismatch. Expected {size}, got {vals.size}.")
        vals = np.maximum(vals, eps)
    return np.diag(vals), vals


def vector_or_zeros(values, size):
    if values is None:
        return np.zeros(size, dtype=float)
    out = np.asarray(values, float).reshape(-1)
    if out.size != size:
        raise ValueError(f"Size mismatch. Expected {size}, got {out.size}.")
    return out


def tracking_solver_sequence(terminal_constraint_active, solver_pref=None):
    if solver_pref is None:
        seq = (
            DEFAULT_TRACKING_CONIC_SOLVERS
            if terminal_constraint_active
            else DEFAULT_TRACKING_QP_SOLVERS
        )
    elif isinstance(solver_pref, str):
        seq = (solver_pref,)
    else:
        seq = tuple(solver_pref)

    seq = tuple(str(name).upper() for name in seq if name is not None)
    if terminal_constraint_active:
        seq = tuple(name for name in seq if name != "OSQP")
        if not seq:
            seq = DEFAULT_TRACKING_CONIC_SOLVERS
    return seq


def safety_filter_solver_sequence(quadratic_constraint_active=True, solver_pref=None):
    if solver_pref is None:
        seq = (
            DEFAULT_SAFETY_FILTER_QCQP_SOLVERS
            if quadratic_constraint_active
            else DEFAULT_SAFETY_FILTER_QP_SOLVERS
        )
    elif isinstance(solver_pref, str):
        seq = (solver_pref,)
    else:
        seq = tuple(solver_pref)

    seq = tuple(str(name).upper() for name in seq if name is not None)
    if quadratic_constraint_active:
        seq = tuple(name for name in seq if name != "OSQP")
        if not seq:
            seq = DEFAULT_SAFETY_FILTER_QCQP_SOLVERS
    elif not seq:
        seq = DEFAULT_SAFETY_FILTER_QP_SOLVERS
    return seq


def get_y_sp_step(y_sp, step_idx, n_outputs):
    y_sp = np.asarray(y_sp, float)
    if y_sp.ndim == 1:
        if y_sp.size != n_outputs:
            raise ValueError(f"y_sp vector has size {y_sp.size}, expected {n_outputs}.")
        return y_sp.reshape(-1)

    if y_sp.ndim != 2:
        raise ValueError(f"y_sp must be 1D or 2D, got shape {y_sp.shape}.")

    if y_sp.shape[0] == n_outputs:
        return y_sp[:, step_idx].reshape(-1)

    if y_sp.shape[1] == n_outputs:
        return y_sp[step_idx, :].reshape(-1)

    raise ValueError(f"Cannot infer y_sp orientation from shape {y_sp.shape} with n_outputs = {n_outputs}.")


def reshape_u_sequence(x_opt, n_inputs, horizon_control):
    return np.asarray(x_opt[:n_inputs * horizon_control], float).reshape(horizon_control, n_inputs)


def compute_du_sequence(u_sequence, u_prev_dev):
    u_prev_dev = np.asarray(u_prev_dev, float).reshape(1, -1)
    u_prev_path = np.vstack([u_prev_dev, u_sequence[:-1, :]])
    return u_sequence - u_prev_path


def shift_input_guess(x_opt, n_inputs, horizon_control):
    u_sequence = reshape_u_sequence(x_opt, n_inputs, horizon_control)
    shifted = np.vstack([u_sequence[1:, :], u_sequence[-1:, :]])
    return shifted.reshape(-1)
