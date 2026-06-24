from __future__ import annotations

from RunCyclePhase1Disturbance_Common import (
    SETPOINT_CYCLE_Y_PHYS,
    run_configured_cycle_study,
)


# Cold-start online TD3 pair on the 400-sample cycle scenario.
# No OF-MPC/LMPC pretrained checkpoint is loaded.
METHODS = (
    "cold_start_safety_gate",
    "cold_start_no_safety_gate",
)


def run_configured_study() -> dict:
    return run_configured_cycle_study(
        methods=METHODS,
        timestamp_label="cycle_phase1dist_cold_start_pair",
        setpoint_cycle_y_phys=SETPOINT_CYCLE_Y_PHYS,
    )


if __name__ == "__main__":
    run_configured_study()
