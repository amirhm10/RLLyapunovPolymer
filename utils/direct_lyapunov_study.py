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

DIRECT_DISTURBANCE_N_TESTS = 200
DIRECT_DISTURBANCE_SETPOINT_LEN = 400
DIRECT_DISTURBANCE_WARM_START = 0
DIRECT_DISTURBANCE_SEED = 0


def direct_disturbance_test_cycle(n_tests: int = DIRECT_DISTURBANCE_N_TESTS) -> List[bool]:
    return [False] * max(int(n_tests), 1)


def direct_four_method_case_specs(
    *,
    anchor_weight: float = 0.1,
    smoothness_weight: float = 0.1,
) -> List[Dict[str, Any]]:
    return [
        {
            "case_name": "bounded_hard",
            "target_mode": "bounded",
            "lyapunov_mode": "hard",
            "target_config": {},
            "label": "Bounded hard",
        },
        {
            "case_name": "bounded_hard_u_prev_0p1",
            "target_mode": "bounded",
            "lyapunov_mode": "hard",
            "target_config": {"u_ref_weight": float(anchor_weight)},
            "label": "Previous-input anchor",
        },
        {
            "case_name": "bounded_hard_xs_prev_0p1",
            "target_mode": "bounded",
            "lyapunov_mode": "hard",
            "target_config": {"x_ref_weight": float(smoothness_weight)},
            "label": "State smoothness",
        },
        {
            "case_name": "bounded_hard_u_prev_0p1_xs_prev_0p1",
            "target_mode": "bounded",
            "lyapunov_mode": "hard",
            "target_config": {
                "u_ref_weight": float(anchor_weight),
                "x_ref_weight": float(smoothness_weight),
            },
            "label": "Anchor + smoothness",
        },
    ]

