"""Shared Direct LMPC selector defaults for online runs and pretraining."""

from __future__ import annotations


DIRECT_LMPC_TARGET_MODE = "bounded"
DIRECT_LMPC_TARGET_SELECTOR_VARIANT = "bounded_mixed_u0p1_x0p1"
DIRECT_LMPC_U_REF_WEIGHT = 0.1
DIRECT_LMPC_X_REF_WEIGHT = 0.1
DIRECT_LMPC_RHO_LYAP = 0.99
DIRECT_LMPC_LYAP_EPS = 1e-3
DIRECT_LMPC_LYAP_TOL = 1e-10
DIRECT_LMPC_SLACK_PENALTY = 1e6


def make_direct_lmpc_target_config() -> dict[str, float]:
    return {
        "u_ref_weight": float(DIRECT_LMPC_U_REF_WEIGHT),
        "x_ref_weight": float(DIRECT_LMPC_X_REF_WEIGHT),
    }


def _format_cache_value(value: float) -> str:
    text = f"{float(value):.6g}"
    return text.replace("-", "m").replace(".", "p")


def direct_lmpc_selector_cache_token() -> str:
    return (
        f"target_{DIRECT_LMPC_TARGET_MODE}_{DIRECT_LMPC_TARGET_SELECTOR_VARIANT}_"
        f"u{_format_cache_value(DIRECT_LMPC_U_REF_WEIGHT)}_"
        f"x{_format_cache_value(DIRECT_LMPC_X_REF_WEIGHT)}"
    )


def direct_lmpc_selector_metadata() -> dict[str, object]:
    return {
        "target_mode": DIRECT_LMPC_TARGET_MODE,
        "target_selector_variant": DIRECT_LMPC_TARGET_SELECTOR_VARIANT,
        "target_config": make_direct_lmpc_target_config(),
        "rho_lyap": float(DIRECT_LMPC_RHO_LYAP),
        "lyap_eps": float(DIRECT_LMPC_LYAP_EPS),
        "lyap_tol": float(DIRECT_LMPC_LYAP_TOL),
        "slack_penalty": float(DIRECT_LMPC_SLACK_PENALTY),
    }


__all__ = [
    "DIRECT_LMPC_LYAP_EPS",
    "DIRECT_LMPC_LYAP_TOL",
    "DIRECT_LMPC_RHO_LYAP",
    "DIRECT_LMPC_SLACK_PENALTY",
    "DIRECT_LMPC_TARGET_MODE",
    "DIRECT_LMPC_TARGET_SELECTOR_VARIANT",
    "DIRECT_LMPC_U_REF_WEIGHT",
    "DIRECT_LMPC_X_REF_WEIGHT",
    "direct_lmpc_selector_cache_token",
    "direct_lmpc_selector_metadata",
    "make_direct_lmpc_target_config",
]
