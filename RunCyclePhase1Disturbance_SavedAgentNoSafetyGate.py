from __future__ import annotations

from RunCyclePhase1Disturbance_Common import (
    DEFAULT_SAVED_AGENT_PATH,
    SETPOINT_CYCLE_Y_PHYS,
    run_configured_cycle_study,
)


# Continue online training from the same saved seed agent with the safety gate
# disabled. This gives the paired comparison for the gated saved-agent runner.
METHODS = (
    "saved_agent_no_safety_gate",
)

SAVED_AGENT_PATH = DEFAULT_SAVED_AGENT_PATH


def run_configured_study() -> dict:
    return run_configured_cycle_study(
        methods=METHODS,
        timestamp_label="cycle_phase1dist_saved_agent_no_safety_gate",
        saved_agent_path=SAVED_AGENT_PATH,
        setpoint_cycle_y_phys=SETPOINT_CYCLE_Y_PHYS,
    )


if __name__ == "__main__":
    run_configured_study()
