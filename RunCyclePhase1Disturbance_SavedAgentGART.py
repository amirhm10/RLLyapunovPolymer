from __future__ import annotations

from RunCyclePhase1Disturbance_Common import (
    DEFAULT_SAVED_AGENT_PATH,
    SETPOINT_CYCLE_Y_PHYS,
    run_configured_cycle_study,
)


# Continue online training from a saved seed agent with the GART-LMPC safety gate.
# No teacher warm-up, behavior-cloning phase, or handoff phase is used.
METHODS = (
    "saved_agent_safety_gate",
)

SAVED_AGENT_PATH = DEFAULT_SAVED_AGENT_PATH


def run_configured_study() -> dict:
    return run_configured_cycle_study(
        methods=METHODS,
        timestamp_label="cycle_phase1dist_saved_agent_gart",
        saved_agent_path=SAVED_AGENT_PATH,
        setpoint_cycle_y_phys=SETPOINT_CYCLE_Y_PHYS,
    )


if __name__ == "__main__":
    run_configured_study()
