from __future__ import annotations

import json

import numpy as np
import pytest

from Lyapunov.gart_target import (
    CertifiedDisturbanceConfig,
    GARTTargetConfig,
    GARTTargetState,
    HAS_CVXPY,
    select_gart_target,
    update_certified_disturbance,
)


pytestmark = pytest.mark.skipif(not HAS_CVXPY, reason="CVXPY is required for GART target tests.")


def _augmented_model(A=0.5, B=1.0, C=1.0):
    A_aug = np.array([[A, 0.0], [0.0, 1.0]], dtype=float)
    B_aug = np.array([[B], [0.0]], dtype=float)
    C_aug = np.array([[C, 1.0]], dtype=float)
    return A_aug, B_aug, C_aug


def _config(**overrides):
    disturbance = CertifiedDisturbanceConfig(
        alpha_d=1.0,
        alpha_d_slow=0.1,
        d_rate_max=np.array([0.5], dtype=float),
        d_min=np.array([-1.0], dtype=float),
        d_max=np.array([1.0], dtype=float),
    )
    cfg = GARTTargetConfig(
        disturbance=disturbance,
        input_headroom_frac=0.0,
        alpha_terminal_min=0.0,
        dy_s_max=None,
        du_s_max=None,
        dx_s_max=None,
        Wy_diag=np.array([1.0]),
        W_u_mid_diag=np.array([0.0]),
        require_contraction_probe=True,
        rho=0.99,
        eps=1.0e-3,
        solver_pref=("CLARABEL", "OSQP", "SCS"),
    )
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


def test_exact_reachable_target():
    A_aug, B_aug, C_aug = _augmented_model()
    y_sp = np.array([0.8])
    xhat_aug = np.array([0.8, 0.0])
    result, state = select_gart_target(
        A_aug,
        B_aug,
        C_aug,
        xhat_aug,
        y_sp,
        np.array([-1.0]),
        np.array([1.0]),
        state=None,
        config=_config(),
        P_x=np.array([[1.0]]),
        K_x=np.array([[0.0]]),
    )
    assert result.success is True
    assert result.governor_alpha == pytest.approx(1.0)
    assert result.hold_previous is False
    assert np.linalg.norm(result.y_s - y_sp) < 1.0e-8
    assert state.valid is True


def test_unreachable_target_due_to_input_bounds_is_closest_reachable():
    A_aug, B_aug, C_aug = _augmented_model()
    y_sp = np.array([3.0])
    result, _ = select_gart_target(
        A_aug,
        B_aug,
        C_aug,
        np.array([2.0, 0.0]),
        y_sp,
        np.array([-1.0]),
        np.array([1.0]),
        state=None,
        config=_config(),
        P_x=np.array([[1.0]]),
        K_x=np.array([[0.0]]),
    )
    assert result.success is True
    assert result.target_error_inf > 0.0
    assert result.input_headroom_min >= -1.0e-7
    assert result.y_s[0] == pytest.approx(2.0, abs=1.0e-6)


def test_target_rate_bound_activates_governor():
    A_aug, B_aug, C_aug = _augmented_model()
    prev = GARTTargetState(
        d_cert=np.array([0.0]),
        x_s=np.array([0.0]),
        u_s=np.array([0.0]),
        y_s=np.array([0.0]),
        r_cmd=np.array([0.0]),
        valid=True,
    )
    cfg = _config(dy_s_max=np.array([0.2]), du_s_max=np.array([0.2]), dx_s_max=np.array([0.2]))
    result, _ = select_gart_target(
        A_aug,
        B_aug,
        C_aug,
        np.array([0.0, 0.0]),
        np.array([1.0]),
        np.array([-1.0]),
        np.array([1.0]),
        state=prev,
        config=cfg,
        P_x=np.array([[1.0]]),
        K_x=np.array([[0.0]]),
    )
    assert result.success is True
    assert result.governor_alpha < 1.0
    assert result.target_rate_y_inf <= 0.2 + 1.0e-6


def test_certified_disturbance_rate_limit():
    cfg = CertifiedDisturbanceConfig(
        alpha_d=1.0,
        alpha_d_slow=0.1,
        d_rate_max=np.array([0.05, 0.1]),
        d_min=np.array([-1.0, -1.0]),
        d_max=np.array([1.0, 1.0]),
    )
    d_cert, info = update_certified_disturbance(
        np.array([0.0, 0.0]),
        np.array([1.0, -1.0]),
        config=cfg,
    )
    assert np.max(np.abs(d_cert - np.array([0.0, 0.0]))) <= 0.1 + 1.0e-12
    assert np.all(np.abs(info["d_cert_delta"]) <= cfg.d_rate_max + 1.0e-12)


def test_contraction_probe_failure_is_reported():
    A_aug, B_aug, C_aug = _augmented_model(A=1.2, B=1.0)
    cfg = _config(rho=0.1, eps=0.0)
    result, _ = select_gart_target(
        A_aug,
        B_aug,
        C_aug,
        np.array([1.0, 0.0]),
        np.array([0.0]),
        np.array([-0.1]),
        np.array([0.1]),
        state=None,
        config=cfg,
        P_x=np.array([[1.0]]),
        K_x=np.array([[0.0]]),
    )
    assert result.contraction_probe_success is False


def test_result_diagnostics_are_json_serializable():
    A_aug, B_aug, C_aug = _augmented_model()
    result, _ = select_gart_target(
        A_aug,
        B_aug,
        C_aug,
        np.array([0.8, 0.0]),
        np.array([0.8]),
        np.array([-1.0]),
        np.array([1.0]),
        state=None,
        config=_config(),
        P_x=np.array([[1.0]]),
        K_x=np.array([[0.0]]),
    )
    json.dumps(result.to_dict())
