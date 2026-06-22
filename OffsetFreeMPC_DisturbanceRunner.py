from __future__ import annotations

from pathlib import Path
from pprint import pprint

import numpy as np

from utils.online_disturbance_runner import build_disturbance_context, run_offset_free_mpc_disturbance
from utils.two_phase_profiles import (
    TwoPhaseExperimentSpec,
    build_two_phase_profiles,
    jsonable_two_phase_profile,
)


# Offset-free MPC feasibility probe for the proposed Phase-2 profile.
# This runs only the Phase-2 setpoint schedule and disturbance ramp so you can
# quickly check whether plain OF-MPC can handle the continuation scenario.

EPISODES = 2
SET_POINTS_LEN = 400
SEED = 123
SAVE_PLOTS = True

OUTPUT_ROOT = Path.home() / "Desktop" / "Lyapunov_polymer_results"
STUDY_NAME = "OffsetFreeMPC_Phase2Feasibility"
CASE_NAME = "offset_free_mpc_phase2_feasibility"
TIMESTAMP = None

PHASE2_SETPOINTS_Y_PHYS = (
    (4.4, 321.5),
    (3.3, 324.5),
)

NOMINAL_QI = 108.0
NOMINAL_QS = 459.0
NOMINAL_HA = 1.05e6

PHASE1_QI_MULTIPLIER = 0.95
PHASE1_QS_MULTIPLIER = 1.05
PHASE1_HA_MULTIPLIER = 0.92

PHASE2_QI_MULTIPLIER = 1.05
PHASE2_QS_MULTIPLIER = 0.95
PHASE2_HA_MULTIPLIER = 0.88


def _phase2_probe_profiles() -> tuple[np.ndarray, dict[str, np.ndarray], dict]:
    context = build_disturbance_context()
    spec = TwoPhaseExperimentSpec(
        phase1_episodes=1,
        phase2_episodes=int(EPISODES),
        set_points_len=int(SET_POINTS_LEN),
        phase2_setpoints_y_phys=np.asarray(PHASE2_SETPOINTS_Y_PHYS, dtype=float),
        nominal_qi=float(NOMINAL_QI),
        nominal_qs=float(NOMINAL_QS),
        nominal_ha=float(NOMINAL_HA),
        phase1_qi_multiplier=float(PHASE1_QI_MULTIPLIER),
        phase1_qs_multiplier=float(PHASE1_QS_MULTIPLIER),
        phase1_ha_multiplier=float(PHASE1_HA_MULTIPLIER),
        phase2_qi_multiplier=float(PHASE2_QI_MULTIPLIER),
        phase2_qs_multiplier=float(PHASE2_QS_MULTIPLIER),
        phase2_ha_multiplier=float(PHASE2_HA_MULTIPLIER),
    )
    profile = build_two_phase_profiles(
        spec=spec,
        data_min=context.system_data["data_min"],
        data_max=context.system_data["data_max"],
        steady_outputs=context.setup.steady_states["y_ss"],
        n_inputs=context.dimensions.inputs_number,
    )
    start = int(profile["phase1_steps"])
    stop = int(profile["total_steps"])
    setpoint_profile = np.asarray(profile["setpoint_profile_scaled_dev"], dtype=float)[start:stop].copy()
    disturbance_profile = {
        name: np.asarray(values, dtype=float)[start:stop].copy()
        for name, values in profile["disturbance_profile"].items()
    }
    metadata = {
        "probe": "phase2_only_feasibility",
        "used_episodes": int(EPISODES),
        "used_set_points_len": int(SET_POINTS_LEN),
        "setpoints_y_phys": np.asarray(PHASE2_SETPOINTS_Y_PHYS, dtype=float).tolist(),
        "n_profile_steps": int(stop - start),
        "disturbance_start": {name: float(values[0]) for name, values in disturbance_profile.items()},
        "disturbance_end": {name: float(values[-1]) for name, values in disturbance_profile.items()},
        "source_two_phase_profile": jsonable_two_phase_profile(profile),
    }
    return setpoint_profile, disturbance_profile, metadata


def run_configured_study() -> dict:
    setpoint_profile, disturbance_profile, metadata = _phase2_probe_profiles()
    config = {
        "episodes": int(EPISODES),
        "set_points_len": int(SET_POINTS_LEN),
        "seed": int(SEED),
        "save_plots": bool(SAVE_PLOTS),
        "output_root": str(OUTPUT_ROOT),
        "study_name": STUDY_NAME,
        "case_name": CASE_NAME,
        "timestamp": TIMESTAMP,
        "phase2_setpoints_y_phys": PHASE2_SETPOINTS_Y_PHYS,
        "disturbance_start": metadata["disturbance_start"],
        "disturbance_end": metadata["disturbance_end"],
    }
    print("Offset-free MPC Phase-2 feasibility configuration:")
    pprint(config)
    return run_offset_free_mpc_disturbance(
        episodes=EPISODES,
        set_points_len=SET_POINTS_LEN,
        seed=SEED,
        save_plots=SAVE_PLOTS,
        timestamp=TIMESTAMP,
        output_root=OUTPUT_ROOT,
        study_name=STUDY_NAME,
        case_name=CASE_NAME,
        setpoint_profile=setpoint_profile,
        disturbance_profile=disturbance_profile,
        profile_metadata=metadata,
    )


if __name__ == "__main__":
    run_configured_study()
