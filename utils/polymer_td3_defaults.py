from __future__ import annotations

import numpy as np


DEFAULT_TD3_STATE_X_MAX = np.array(
    [
        256.79686253,
        256.01560603,
        48.99447186,
        144.79949103,
        2.82199733,
        3.14014989,
        2.78866348,
        3.71691422,
        6.2029936,
    ],
    dtype=float,
)

DEFAULT_TD3_STATE_X_MIN = np.array(
    [
        -272.28060121,
        -1112.33972595,
        -76.63993491,
        -608.60327886,
        -3.94399122,
        -3.93115257,
        -2.9532091,
        -4.06547624,
        -28.25906582,
    ],
    dtype=float,
)

DEFAULT_TD3_SETPOINT_Y_PHYS = np.array(
    [
        [4.5, 324.0],
        [3.4, 321.0],
    ],
    dtype=float,
)

DEFAULT_U_MIN_PHYS = np.array([71.6, 78.0], dtype=float)
DEFAULT_U_MAX_PHYS = np.array([870.0, 670.0], dtype=float)


def default_td3_state_bounds(n_states: int) -> tuple[np.ndarray, np.ndarray]:
    """Return the repo-default TD3 augmented-state scaling bounds."""
    n_states = int(n_states)
    if n_states != DEFAULT_TD3_STATE_X_MIN.size:
        raise ValueError(
            "Default Polymer TD3 state bounds have length "
            f"{DEFAULT_TD3_STATE_X_MIN.size}, got augmented state dimension {n_states}."
        )
    return DEFAULT_TD3_STATE_X_MIN.copy(), DEFAULT_TD3_STATE_X_MAX.copy()
