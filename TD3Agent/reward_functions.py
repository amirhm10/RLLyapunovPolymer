import numpy as np


def make_reward_fn_relative_QR(
    data_min, data_max, n_inputs,
    k_rel, band_floor_phys,
    Q_diag, R_diag,
    tau_frac=0.7,
    gamma_out=0.5, gamma_in=0.5,
    beta=5.0, gate="geom", lam_in=1.0,
    bonus_kind="exp", bonus_k=12.0, bonus_p=0.6, bonus_c=20.0,
):
    """
    Reward with relative tracking bands.

    data_min, data_max : arrays for [u_min..., y_min...], [u_max..., y_max...]
    n_inputs           : number of inputs (so outputs start at index n_inputs)
    k_rel              : per-output relative tolerance factors (len = n_outputs)
    band_floor_phys    : per-output minimum band in physical units (len = n_outputs)
    Q_diag, R_diag     : quadratic weights (len = n_outputs, len = n_inputs)
    """

    data_min = np.asarray(data_min, float)
    data_max = np.asarray(data_max, float)

    dy = np.maximum(data_max[n_inputs:] - data_min[n_inputs:], 1e-12)

    k_rel = np.asarray(k_rel, float)
    band_floor_phys = np.asarray(band_floor_phys, float)
    Q_diag = np.asarray(Q_diag, float)
    R_diag = np.asarray(R_diag, float)

    band_floor_scaled = band_floor_phys / np.maximum(dy, 1e-12)

    def _sigmoid(x):
        x = np.clip(x, -60.0, 60.0)
        return 1.0 / (1.0 + np.exp(-x))

    def _phi(z):
        z = np.clip(z, 0.0, 1.0)
        if bonus_kind == "linear":
            return 1.0 - z
        if bonus_kind == "quadratic":
            return (1.0 - z) ** 2
        if bonus_kind == "exp":
            return (np.exp(-bonus_k * z) - np.exp(-bonus_k)) / (1.0 - np.exp(-bonus_k))
        if bonus_kind == "power":
            return 1.0 - np.power(z, bonus_p)
        if bonus_kind == "log":
            return np.log1p(bonus_c * (1.0 - z)) / np.log1p(bonus_c)
        raise ValueError("unknown bonus_kind")

    def reward_fn(e_scaled, du_scaled, y_sp_phys=None):
        e_scaled = np.asarray(e_scaled, float)
        du_scaled = np.asarray(du_scaled, float)

        if y_sp_phys is None:
            band_scaled = band_floor_scaled
        else:
            y_sp_phys = np.asarray(y_sp_phys, float)
            band_phys = np.maximum(k_rel * np.abs(y_sp_phys), band_floor_phys)
            band_scaled = band_phys / np.maximum(dy, 1e-12)

        tau_scaled = tau_frac * band_scaled

        abs_e = np.abs(e_scaled)
        s_i = _sigmoid((band_scaled - abs_e) / np.maximum(tau_scaled, 1e-12))

        if gate == "prod":
            w_in = float(np.prod(s_i, dtype=np.float64))
        elif gate == "mean":
            w_in = float(np.mean(s_i))
        elif gate == "geom":
            w_in = float(np.prod(s_i, dtype=np.float64) ** (1.0 / max(1, len(s_i))))
        else:
            raise ValueError("gate must be 'prod'|'mean'|'geom'")

        err_quad = np.sum(Q_diag * (e_scaled ** 2))
        err_eff = (1.0 - w_in) * err_quad + w_in * (lam_in * err_quad)

        move = np.sum(R_diag * (du_scaled ** 2))

        slope_at_edge = 2.0 * Q_diag * band_scaled

        overflow = np.maximum(abs_e - band_scaled, 0.0)
        lin_out = (1.0 - w_in) * np.sum(gamma_out * slope_at_edge * overflow)

        inside_mag = np.minimum(abs_e, band_scaled)
        lin_in = w_in * np.sum(gamma_in * slope_at_edge * inside_mag)

        qb2 = Q_diag * (band_scaled ** 2)
        z = abs_e / np.maximum(band_scaled, 1e-12)
        bonus = w_in * beta * np.sum(qb2 * _phi(z))

        return float(-(err_eff + move + lin_out + lin_in) + bonus)

    params = dict(
        k_rel=k_rel,
        band_floor_phys=band_floor_phys,
        band_floor_scaled=band_floor_scaled,
        Q_diag=Q_diag,
        R_diag=R_diag,
        tau_frac=tau_frac,
        gamma_out=gamma_out,
        gamma_in=gamma_in,
        beta=beta,
        gate=gate,
        lam_in=lam_in,
        bonus_kind=bonus_kind,
        bonus_k=bonus_k,
        bonus_p=bonus_p,
        bonus_c=bonus_c,
    )
    return params, reward_fn


def make_reward_fn_mpc_quadratic(Q_diag, R_diag):
    """
    One-step MPC-style quadratic stage cost in scaled deviation coordinates:
      r = - (sum_i Q_i e_i^2 + sum_j R_j du_j^2)

    Matches the signature of the relative reward:
      reward_fn(e_scaled, du_scaled, y_sp_phys=None)
    """

    Q_diag = np.asarray(Q_diag, float)
    R_diag = np.asarray(R_diag, float)

    def reward_fn(e_scaled, du_scaled, y_sp_phys=None):
        e = np.asarray(e_scaled, float)
        du = np.asarray(du_scaled, float)
        err = np.sum(Q_diag * (e ** 2))
        move = np.sum(R_diag * (du ** 2))
        return float(-(err + move))

    params = dict(Q_diag=Q_diag, R_diag=R_diag)
    return params, reward_fn
