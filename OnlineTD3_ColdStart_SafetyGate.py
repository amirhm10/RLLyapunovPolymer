from __future__ import annotations

from pprint import pprint

from utils.gart_defaults import GART_FINAL_LYAP_EPS, GART_FINAL_RHO_LYAP
from utils.direct_lyapunov_study import DIRECT_DISTURBANCE_N_TESTS
from utils.online_disturbance_runner import run_online_td3_disturbance_preset

# Cold-start online TD3 with the GART Section 16 safety gate.
#
# This root runner is intentionally editable, following the style of
# GARTLyapunovMPC.py. Change these values here for day-to-day experiments.

EPISODES = DIRECT_DISTURBANCE_N_TESTS
SET_POINTS_LEN = 400
SEED = 123
SAVE_PLOTS = True

# Set to None for an automatic timestamp, or use a fixed string to rerun into a
# predictable folder under results/OnlineTD3_ColdStart_SafetyGate/.
TIMESTAMP = None

# Section 16 defaults: the cold-start actor receives GART target information,
# and unsafe actor proposals are projected to the nearest certified input before
# the GART-LMPC fallback is allowed to take over.
RL_OBSERVATION_MODE = "gart"
PROJECTION_BACKEND = "gart_section16_projection"

RHO_LYAP = GART_FINAL_RHO_LYAP
LYAP_EPS = GART_FINAL_LYAP_EPS
LYAP_TOL = None

REWARD_FALLBACK_PENALTY_ENABLED = True
GAMMA_FALLBACK = 3.0
FALLBACK_EVENT_PENALTY = 10.0

TRAINING_PHASE_OVERRIDES = {
    # Examples:
    # "behavior_clone_teacher_episodes": 20,
    # "handoff_episodes": 5,
    # "full_rl_exploration_std_start": 0.1,
}

SECTION16_PROJECTION_CONFIG = {
    "candidate_weight_diag": [1.0, 1.0],
    "move_weight_diag": [0.0, 0.0],
    "steady_weight_diag": [0.0, 0.0],
    "output_weight_diag": [0.0, 0.0],
    "use_output_tracking_term": False,
    "allow_lyap_slack": False,
    "allow_trust_region_slack": False,
    "lyap_acceptance_mode": "hard_only",
    "solver_pref": None,
    "certificate_aware_exploration": True,
    "certificate_margin_scale": 1.0,
    "certificate_sigma_floor": 0.0,
}


def run_configured_study() -> dict:
    config = {
        "preset": "cold_start_safety_gate",
        "episodes": int(EPISODES),
        "set_points_len": int(SET_POINTS_LEN),
        "seed": int(SEED),
        "save_plots": bool(SAVE_PLOTS),
        "timestamp": TIMESTAMP,
        "rl_observation_mode": RL_OBSERVATION_MODE,
        "projection_backend": PROJECTION_BACKEND,
        "rho_lyap": float(RHO_LYAP),
        "lyap_eps": float(LYAP_EPS),
        "lyap_tol": LYAP_TOL,
        "reward_fallback_penalty_enabled": bool(REWARD_FALLBACK_PENALTY_ENABLED),
        "gamma_fallback": float(GAMMA_FALLBACK),
        "fallback_event_penalty": float(FALLBACK_EVENT_PENALTY),
        "training_phase_overrides": dict(TRAINING_PHASE_OVERRIDES),
        "section16_projection_config": dict(SECTION16_PROJECTION_CONFIG),
    }
    print("Online TD3 cold-start safety-gate configuration:")
    pprint(config)
    return run_online_td3_disturbance_preset(
        "cold_start_safety_gate",
        episodes=EPISODES,
        set_points_len=SET_POINTS_LEN,
        seed=SEED,
        save_plots=SAVE_PLOTS,
        timestamp=TIMESTAMP,
        rl_observation_mode=RL_OBSERVATION_MODE,
        projection_backend=PROJECTION_BACKEND,
        reward_fallback_penalty_enabled=REWARD_FALLBACK_PENALTY_ENABLED,
        gamma_fallback=GAMMA_FALLBACK,
        fallback_event_penalty=FALLBACK_EVENT_PENALTY,
        rho_lyap=RHO_LYAP,
        lyap_eps=LYAP_EPS,
        lyap_tol=LYAP_TOL,
        training_phase_overrides=TRAINING_PHASE_OVERRIDES,
        section16_projection_config=SECTION16_PROJECTION_CONFIG,
    )


if __name__ == "__main__":
    run_configured_study()
