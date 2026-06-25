from __future__ import annotations

from RunCyclePhase1Disturbance_Common import (
    SETPOINT_CYCLE_Y_PHYS,
    run_configured_cycle_study,
)


# Deterministic GART-LMPC baseline on the same 400-sample cycle and Phase-1
# disturbance profile used by the saved-agent gate/no-gate comparison.
METHODS = (
    "gart_lmpc",
)


def run_configured_study() -> dict:
    return run_configured_cycle_study(
        methods=METHODS,
        timestamp_label="cycle_phase1dist_gart_lmpc",
        setpoint_cycle_y_phys=SETPOINT_CYCLE_Y_PHYS,
    )


if __name__ == "__main__":
    run_configured_study()
