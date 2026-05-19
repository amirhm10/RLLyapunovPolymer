import numpy as np


def make_reward_fn_relative_QR(
    data_min, data_max, n_inputs,
    k_rel, band_floor_phys,
    Q_diag, R_diag,
    tau_frac=0.7,
    gamma_out=0.5, gamma_in=0.5,
    beta=5.0, gate="geom", lam_in=1.0,
    bonus_kind="exp", bonus_k=12.0, bonus_p=0.6, bonus_c=20.0,
    gamma_fallback=0.0, fallback_event_penalty=0.0, R_fallback_diag=None,
    maintenance_band_scale=1.0,
    maintenance_move_weight=0.0,
    jitter_weight=0.0,
    dwell_bonus=0.0,
):
    """
    Reward with relative tracking bands.

    data_min, data_max : arrays for [u_min..., y_min...], [u_max..., y_max...]
    n_inputs           : number of inputs (so outputs start at index n_inputs)
    k_rel              : per-output relative tolerance factors (len = n_outputs)
    band_floor_phys    : per-output minimum band in physical units (len = n_outputs)
    Q_diag, R_diag     : quadratic weights (len = n_outputs, len = n_inputs)
    gamma_fallback     : scalar weight on safety-gate correction mismatch
    fallback_event_penalty
                       : fixed cost charged for each active safety-gate fallback.
    R_fallback_diag    : optional input-channel weights for correction mismatch.
                         Defaults to R_diag.
    maintenance_*      : optional near-setpoint terms. Defaults preserve the
                         historical reward exactly.
    """

    data_min = np.asarray(data_min, float)
    data_max = np.asarray(data_max, float)

    dy = np.maximum(data_max[n_inputs:] - data_min[n_inputs:], 1e-12)

    k_rel = np.asarray(k_rel, float)
    band_floor_phys = np.asarray(band_floor_phys, float)
    Q_diag = np.asarray(Q_diag, float)
    R_diag = np.asarray(R_diag, float)
    R_fallback_diag = R_diag.copy() if R_fallback_diag is None else np.asarray(R_fallback_diag, float)
    gamma_fallback = float(gamma_fallback)
    fallback_event_penalty = float(fallback_event_penalty)
    maintenance_band_scale = float(maintenance_band_scale)
    maintenance_move_weight = float(maintenance_move_weight)
    jitter_weight = float(jitter_weight)
    dwell_bonus = float(dwell_bonus)

    band_floor_scaled = band_floor_phys / np.maximum(dy, 1e-12)
    prev_e_scaled = None
    dwell_count = 0

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

    def reward_fn(
        e_scaled,
        du_scaled,
        y_sp_phys=None,
        *,
        fallback_gap=None,
        fallback_active=False,
        return_components=False,
    ):
        nonlocal prev_e_scaled, dwell_count
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

        reward_base = float(-(err_eff + move + lin_out + lin_in) + bonus)

        maintenance_band = np.maximum(float(maintenance_band_scale), 0.0) * band_scaled
        inside_maintenance_band = bool(np.all(abs_e <= maintenance_band + 1.0e-12))
        maintenance_move_penalty = 0.0
        if inside_maintenance_band and maintenance_move_weight != 0.0:
            maintenance_move_penalty = float(maintenance_move_weight * np.sum(R_diag * (du_scaled ** 2)))

        output_jitter_penalty = 0.0
        if prev_e_scaled is not None and jitter_weight != 0.0:
            de = e_scaled - prev_e_scaled
            output_jitter_penalty = float(jitter_weight * np.sum(Q_diag * (de ** 2)))

        if inside_maintenance_band:
            dwell_count += 1
        else:
            dwell_count = 0
        dwell_reward = float(dwell_bonus * dwell_count) if dwell_bonus != 0.0 else 0.0

        weighted_correction_gap = 0.0
        fallback_correction_penalty = 0.0
        fallback_event_penalty_applied = 0.0
        fallback_penalty = 0.0
        if bool(fallback_active):
            if fallback_gap is not None:
                gap = np.asarray(fallback_gap, float).reshape(-1)
                weighted_correction_gap = float(np.sum(R_fallback_diag * (gap ** 2)))
                fallback_correction_penalty = float(gamma_fallback * weighted_correction_gap)
            fallback_event_penalty_applied = fallback_event_penalty
            fallback_penalty = float(fallback_correction_penalty + fallback_event_penalty_applied)

        reward_augmented = float(
            reward_base
            - fallback_penalty
            - maintenance_move_penalty
            - output_jitter_penalty
            + dwell_reward
        )
        prev_e_scaled = e_scaled.copy()
        if return_components:
            return {
                "reward": reward_augmented,
                "reward_base": reward_base,
                "fallback_penalty": fallback_penalty,
                "fallback_correction_penalty": fallback_correction_penalty,
                "fallback_event_penalty": fallback_event_penalty_applied,
                "fallback_event_penalty_config": fallback_event_penalty,
                "weighted_correction_gap": weighted_correction_gap,
                "fallback_active": bool(fallback_active),
                "maintenance_move_penalty": maintenance_move_penalty,
                "output_jitter_penalty": output_jitter_penalty,
                "dwell_reward": dwell_reward,
                "dwell_count": int(dwell_count),
                "inside_maintenance_band": bool(inside_maintenance_band),
                "tracking_cost": float(err_eff + lin_out + lin_in),
                "move_cost": float(move),
                "bonus": float(bonus),
                "w_in": float(w_in),
            }
        return reward_augmented

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
        gamma_fallback=gamma_fallback,
        fallback_event_penalty=fallback_event_penalty,
        R_fallback_diag=R_fallback_diag,
        maintenance_band_scale=maintenance_band_scale,
        maintenance_move_weight=maintenance_move_weight,
        jitter_weight=jitter_weight,
        dwell_bonus=dwell_bonus,
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
