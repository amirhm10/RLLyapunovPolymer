import numpy as np


def apply_min_max(data, min_val, max_val, eps=1e-12):
    """Min-max scale data to [0, 1].

    If max_val == min_val (or extremely close), scaling is undefined.
    In that case, this returns 0 for that element to avoid NaNs/Infs.
    """
    x = np.asarray(data, dtype=float)
    mn = np.asarray(min_val, dtype=float)
    mx = np.asarray(max_val, dtype=float)

    diff = mx - mn
    mask = np.abs(diff) < eps
    denom = np.where(mask, 1.0, diff)
    out = (x - mn) / denom
    return np.where(mask, 0.0, out)


def reverse_min_max(scaled_data, min_val, max_val):
    """Inverse of apply_min_max (maps [0, 1] back to the original scale)."""
    z = np.asarray(scaled_data, dtype=float)
    mn = np.asarray(min_val, dtype=float)
    mx = np.asarray(max_val, dtype=float)
    return z * (mx - mn) + mn

def apply_min_max_pm1(data, min_val, max_val, eps=1e-12):
    """Min-max scale to [-1, 1] using apply_min_max([0, 1]) then affine map."""
    return 2.0 * apply_min_max(data, min_val, max_val, eps=eps) - 1.0

def apply_rl_scaled(min_max_dict, x_d_states, y_sp, u):
    x_min, x_max = min_max_dict["x_min"], min_max_dict["x_max"]
    y_sp_min, y_sp_max = min_max_dict["y_sp_min"], min_max_dict["y_sp_max"]
    u_min, u_max = min_max_dict["u_min"], min_max_dict["u_max"]

    x_d_states_scaled = apply_min_max_pm1(x_d_states, x_min, x_max)
    y_sp_scaled = apply_min_max_pm1(y_sp, y_sp_min, y_sp_max)
    u_scaled = apply_min_max_pm1(u, u_min, u_max)

    return np.hstack((x_d_states_scaled, y_sp_scaled, u_scaled))