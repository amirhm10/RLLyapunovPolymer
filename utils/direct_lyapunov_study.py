from __future__ import annotations

from typing import Any, Dict, List

import numpy as np


DIRECT_TWO_SETPOINT_Y_PHYS = np.array(
    [
        [4.5, 324.0],
        [3.4, 321.0],
    ],
    dtype=float,
)

DIRECT_DISTURBANCE_N_TESTS = 300
DIRECT_DISTURBANCE_SETPOINT_LEN = 400
DIRECT_DISTURBANCE_WARM_START = 0
DIRECT_DISTURBANCE_SEED = 0
GOVERNED_REFERENCE_TARGET_MODE = "governed_reference"


def direct_disturbance_test_cycle(n_tests: int = DIRECT_DISTURBANCE_N_TESTS) -> List[bool]:
    return [False] * max(int(n_tests), 1)


def make_governed_reference_target_config(
    Qy_diag: Any | None = None,
    *,
    lambda_cmd_move: float = 1.0,
    u_ref_weight: float = 0.1,
    x_ref_weight: float = 0.1,
    input_headroom_frac: float = 0.03,
    one_step_probe: bool = True,
) -> Dict[str, Any]:
    """Default governed-reference target configuration for active LyapMPC runs."""
    cfg: Dict[str, Any] = {
        "governed_reference_enabled": True,
        "lambda_cmd_move": float(lambda_cmd_move),
        "Qr_diag": None,
        "W_r_diag": None,
        "u_ref_weight": float(u_ref_weight),
        "x_ref_weight": float(x_ref_weight),
        "input_headroom_frac": float(input_headroom_frac),
        "one_step_probe": bool(one_step_probe),
    }
    if Qy_diag is not None:
        qy = np.asarray(Qy_diag, dtype=float).copy()
        cfg["Qr_diag"] = qy.copy()
        cfg["W_r_diag"] = qy.copy()
    return cfg


def governed_reference_case_spec(
    Qy_diag: Any | None = None,
    *,
    case_name: str,
    controller_mode: str = "direct_lyapunov_mpc",
    lyapunov_mode: str = "hard",
    label: str | None = None,
    **config_overrides: Any,
) -> Dict[str, Any]:
    target_config = make_governed_reference_target_config(Qy_diag, **config_overrides)
    spec: Dict[str, Any] = {
        "case_name": str(case_name),
        "target_mode": GOVERNED_REFERENCE_TARGET_MODE,
        "lyapunov_mode": str(lyapunov_mode),
        "target_config": target_config,
        "controller_mode": str(controller_mode),
    }
    if label is not None:
        spec["label"] = str(label)
    return spec


def _weight_token(value: float) -> str:
    text = f"{float(value):.6f}".rstrip("0").rstrip(".")
    if text in {"", "-0"}:
        text = "0"
    return text.replace("-", "m").replace(".", "p")


def _normalize_case_variants(
    variants: str | tuple[str, ...] | list[str] | set[str] | None,
    *,
    allowed: set[str],
) -> set[str]:
    if variants is None:
        return set(allowed)

    if isinstance(variants, str):
        raw_items = [item.strip().lower() for item in variants.split(",")]
    else:
        raw_items = [str(item).strip().lower() for item in variants]

    normalized = [item for item in raw_items if item]
    if not normalized:
        allowed_str = ", ".join(sorted(allowed))
        raise ValueError(f"variants must contain at least one entry; allowed values are {allowed_str}.")

    unknown = [item for item in normalized if item not in allowed]
    if unknown:
        allowed_str = ", ".join(sorted(allowed))
        raise ValueError(f"variants contains unknown entries {unknown!r}; allowed values are {allowed_str}.")
    return set(normalized)


def direct_four_method_case_specs(
    *,
    anchor_weight: float = 0.1,
    smoothness_weight: float = 0.1,
    variants: str | tuple[str, ...] | list[str] | set[str] | None = None,
) -> List[Dict[str, Any]]:
    anchor_token = _weight_token(anchor_weight)
    smoothness_token = _weight_token(smoothness_weight)
    ordered_cases = [
        (
            "none",
            {
                "case_name": "bounded_hard",
                "target_mode": "bounded",
                "lyapunov_mode": "hard",
                "target_config": {},
                "label": "Bounded hard",
            },
        ),
        (
            "u_prev",
            {
                "case_name": f"bounded_hard_u_prev_{anchor_token}",
                "target_mode": "bounded",
                "lyapunov_mode": "hard",
                "target_config": {"u_ref_weight": float(anchor_weight)},
                "label": f"Previous-input anchor ({anchor_weight:g})",
            },
        ),
        (
            "xs_prev",
            {
                "case_name": f"bounded_hard_xs_prev_{smoothness_token}",
                "target_mode": "bounded",
                "lyapunov_mode": "hard",
                "target_config": {"x_ref_weight": float(smoothness_weight)},
                "label": f"State smoothness ({smoothness_weight:g})",
            },
        ),
        (
            "mixed",
            {
                "case_name": f"bounded_hard_u_prev_{anchor_token}_xs_prev_{smoothness_token}",
                "target_mode": "bounded",
                "lyapunov_mode": "hard",
                "target_config": {
                    "u_ref_weight": float(anchor_weight),
                    "x_ref_weight": float(smoothness_weight),
                },
                "label": f"Anchor + smoothness ({anchor_weight:g}, {smoothness_weight:g})",
            },
        ),
    ]
    allowed = {key for key, _ in ordered_cases}
    selected_variants = _normalize_case_variants(variants, allowed=allowed)
    return [dict(case_spec) for key, case_spec in ordered_cases if key in selected_variants]
