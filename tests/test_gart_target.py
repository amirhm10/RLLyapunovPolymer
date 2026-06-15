from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from Lyapunov.gart_lmpc import GARTMPCConfig, solve_gart_lmpc_step
from Lyapunov.gart_target import (
    CertifiedDisturbanceConfig,
    GARTTargetConfig,
    GARTTargetState,
    HAS_CVXPY,
    jsonable,
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
    assert result.solve_success is True
    assert result.accepted is True
    assert result.usable_for_lmpc is True
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
    assert result.solve_success is True
    assert result.accepted is True
    assert result.usable_for_lmpc is True
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


def test_alpha_zero_governor_recertifies_current_disturbance_target():
    A_aug, B_aug, C_aug = _augmented_model()
    prev = GARTTargetState(
        d_cert=np.array([0.0]),
        x_s=np.array([0.0]),
        u_s=np.array([0.0]),
        y_s=np.array([0.0]),
        r_cmd=np.array([0.0]),
        valid=True,
    )
    cfg = _config(
        dy_s_max=np.array([0.05]),
        du_s_max=np.array([1.0]),
        dx_s_max=np.array([1.0]),
        governor_grid=(1.0, 0.0),
        governor_bisect_iters=0,
    )
    result, state = select_gart_target(
        A_aug,
        B_aug,
        C_aug,
        np.array([-0.2, 0.2]),
        np.array([1.0]),
        np.array([-1.0]),
        np.array([1.0]),
        state=prev,
        config=cfg,
        P_x=np.array([[1.0]]),
        K_x=np.array([[0.0]]),
    )
    assert result.accepted is True
    assert result.usable_for_lmpc is True
    assert result.governor_alpha == pytest.approx(0.0)
    assert result.hold_previous is True
    assert result.status == "accepted_held_command_reference"
    assert result.d_cert[0] == pytest.approx(0.2)
    assert result.x_s[0] == pytest.approx(-0.2, abs=1.0e-6)
    assert result.u_s[0] == pytest.approx(-0.1, abs=1.0e-6)
    assert result.y_s[0] == pytest.approx(0.0, abs=1.0e-6)
    assert result.y_s[0] == pytest.approx(C_aug[0, 0] * result.x_s[0] + result.d_cert[0], abs=1.0e-6)
    assert state.valid is True
    assert state.x_s[0] == pytest.approx(result.x_s[0])
    assert state.d_cert[0] == pytest.approx(result.d_cert[0])


def test_stale_hold_previous_fallback_is_not_usable_when_recertification_fails():
    A_aug, B_aug, C_aug = _augmented_model()
    prev = GARTTargetState(
        d_cert=np.array([0.0]),
        x_s=np.array([0.0]),
        u_s=np.array([0.0]),
        y_s=np.array([0.0]),
        r_cmd=np.array([0.0]),
        valid=True,
    )
    cfg = _config(
        dy_s_max=np.array([0.05]),
        du_s_max=np.array([1.0]),
        dx_s_max=np.array([0.05]),
        governor_grid=(1.0, 0.0),
        governor_bisect_iters=0,
    )
    result, state = select_gart_target(
        A_aug,
        B_aug,
        C_aug,
        np.array([0.0, 0.2]),
        np.array([1.0]),
        np.array([-1.0]),
        np.array([1.0]),
        state=prev,
        config=cfg,
        P_x=np.array([[1.0]]),
        K_x=np.array([[0.0]]),
    )
    assert result.solve_success is False
    assert result.accepted is False
    assert result.usable_for_lmpc is False
    assert result.success is False
    assert result.status == "hold_previous_not_recertified"
    assert result.rejection_reason == "held_previous_target_not_recertified"
    assert result.diagnostics["held_previous_target_not_recertified"] is True
    assert result.diagnostics["stale_target_equation_residual"][0] == pytest.approx(-0.2)
    assert state.valid is True


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
    assert info["adaptive_rate_enabled"] is False
    assert np.allclose(info["d_rate_max_effective"], cfg.d_rate_max)


def test_adaptive_certified_disturbance_shrinks_large_raw_gap():
    cfg = CertifiedDisturbanceConfig(
        alpha_d=1.0,
        alpha_d_slow=0.1,
        d_rate_max=np.array([0.2, 0.2]),
        d_min=np.array([-10.0, -10.0]),
        d_max=np.array([10.0, 10.0]),
        adaptive_rate_enabled=True,
        adaptive_rate_trust_radius=0.25,
        adaptive_rate_min_scale=0.1,
    )
    d_cert, info = update_certified_disturbance(
        np.array([0.0, 0.0]),
        np.array([5.0, -0.1]),
        config=cfg,
    )
    assert info["adaptive_rate_scale"][0] == pytest.approx(0.1)
    assert info["adaptive_rate_scale"][1] == pytest.approx(1.0)
    assert info["d_rate_max_effective"][0] == pytest.approx(0.02)
    assert d_cert[0] == pytest.approx(0.02)
    assert d_cert[1] == pytest.approx(-0.1)
    assert info["d_raw_gap_inf"] == pytest.approx(5.0)


def test_adaptive_certified_disturbance_keeps_full_rate_for_small_gap():
    cfg = CertifiedDisturbanceConfig(
        alpha_d=1.0,
        alpha_d_slow=0.1,
        d_rate_max=np.array([0.2]),
        d_min=np.array([-10.0]),
        d_max=np.array([10.0]),
        adaptive_rate_enabled=True,
        adaptive_rate_trust_radius=0.25,
        adaptive_rate_min_scale=0.1,
    )
    d_cert, info = update_certified_disturbance(
        np.array([0.0]),
        np.array([0.1]),
        config=cfg,
    )
    assert info["adaptive_rate_scale"][0] == pytest.approx(1.0)
    assert info["d_rate_max_effective"][0] == pytest.approx(0.2)
    assert d_cert[0] == pytest.approx(0.1)


def test_adaptive_certified_disturbance_respects_bounds_and_serializes():
    cfg = CertifiedDisturbanceConfig(
        alpha_d=1.0,
        alpha_d_slow=0.1,
        d_rate_max=np.array([10.0]),
        d_min=np.array([-0.1]),
        d_max=np.array([0.1]),
        adaptive_rate_enabled=True,
        adaptive_rate_trust_radius=1.0,
        adaptive_rate_min_scale=0.1,
    )
    d_cert, info = update_certified_disturbance(
        np.array([0.09]),
        np.array([5.0]),
        config=cfg,
    )
    assert d_cert[0] <= 0.1 + 1.0e-12
    assert info["adaptive_rate_enabled"] is True
    json.dumps(jsonable(info))


def test_contraction_probe_failure_is_reported():
    A_aug, B_aug, C_aug = _augmented_model(A=1.2, B=1.0)
    cfg = _config(rho=0.1, eps=0.0)
    result, state = select_gart_target(
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
    assert result.solve_success is True
    assert result.accepted is False
    assert result.usable_for_lmpc is False
    assert result.success is False
    assert result.contraction_probe_success is False
    assert result.rejection_reason == "contraction_probe_failed"
    assert state.valid is False


def test_no_previous_target_rejected_does_not_store_state():
    A_aug, B_aug, C_aug = _augmented_model(A=1.2, B=1.0)
    result, state = select_gart_target(
        A_aug,
        B_aug,
        C_aug,
        np.array([1.0, 0.0]),
        np.array([0.0]),
        np.array([-0.1]),
        np.array([0.1]),
        state=None,
        config=_config(rho=0.1, eps=0.0),
        P_x=np.array([[1.0]]),
        K_x=np.array([[0.0]]),
    )
    assert result.solve_success is True
    assert result.accepted is False
    assert result.success is False
    assert state.valid is False


def test_lmpc_refuses_unaccepted_target():
    lmpc_obj = SimpleNamespace(
        A=np.eye(2),
        B=np.array([[1.0], [0.0]]),
        C=np.array([[1.0, 1.0]]),
        NP=2,
        NC=1,
    )
    target = {
        "success": False,
        "solve_success": True,
        "accepted": False,
        "usable_for_lmpc": False,
        "rejection_reason": "contraction_probe_failed",
        "status": "initial_target_rejected",
        "stage": "stage2",
        "target_error_inf": 0.0,
        "governor_alpha": 1.0,
        "governor_active": False,
        "hold_previous": False,
        "contraction_probe_success": False,
        "contraction_probe_margin_good": -1.0,
        "input_headroom_min": 1.0,
    }
    cfg = GARTMPCConfig(
        Q_raw_diag=np.array([1.0]),
        Q_target_diag=np.array([1.0]),
        R_us_diag=np.array([1.0]),
        Rdu_diag=np.array([1.0]),
        lyapunov_mode="hard",
    )
    u_apply, ic_next, step_info = solve_gart_lmpc_step(
        lmpc_obj,
        np.zeros(2),
        np.zeros(1),
        target,
        np.array([0.2]),
        np.zeros(1),
        ((-1.0, 1.0),),
        np.array([-1.0]),
        np.array([1.0]),
        cfg,
    )
    assert step_info["method"] == "gart_target_not_usable_hold_prev"
    assert step_info["message"] == "contraction_probe_failed"
    assert u_apply[0] == pytest.approx(0.2)
    assert ic_next[0] == pytest.approx(0.2)


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


def test_recursive_result_scanning_disabled_by_default():
    from utils.gart_defaults import discover_gart_case_values

    system_data = {
        "A_aug": np.eye(2),
        "B_aug": np.array([[1.0], [0.0]]),
        "C_aug": np.array([[1.0, 1.0]]),
        "b_min": np.array([-1.0]),
        "b_max": np.array([1.0]),
        "min_max_dict": {
            "x_min": np.array([-1.0, -1.0]),
            "x_max": np.array([1.0, 1.0]),
            "y_sp_min": np.array([-1.0]),
            "y_sp_max": np.array([1.0]),
        },
    }
    values = discover_gart_case_values(system_data, {"steady_states": {}}, results_roots=None)
    assert values["quantiles"]["d_q005"] is None
    assert values["quantiles"]["dy_abs_q95"] is None


def test_make_gart_target_config_accepts_absolute_dy_rate_override():
    from utils.gart_defaults import make_gart_target_config

    values = {
        "d_rate_max": np.array([0.5, 0.25]),
        "d_min": np.array([-1.0, -1.0]),
        "d_max": np.array([1.0, 1.0]),
        "dy_s_max": np.array([0.01, 0.02]),
        "du_s_max": np.array([0.1, 0.2]),
        "dx_s_max": np.ones(3),
        "Wy_diag": np.ones(2),
        "Q_raw_diag": np.ones(2),
        "Q_target_diag": np.ones(2),
        "R_us_diag": np.ones(2),
        "Rdu_diag": np.ones(2),
    }
    cfg = make_gart_target_config(values, dy_s_max_abs=0.1, dy_rate_scale=10.0)
    assert np.allclose(cfg.dy_s_max, np.array([0.1, 0.1]))

    cfg = make_gart_target_config(values, dy_s_max_abs=[0.1, 0.2])
    assert np.allclose(cfg.dy_s_max, np.array([0.1, 0.2]))
