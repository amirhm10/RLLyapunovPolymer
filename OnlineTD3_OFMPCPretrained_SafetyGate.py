from __future__ import annotations

from pprint import pprint

from utils.gart_defaults import GART_FINAL_LYAP_EPS, GART_FINAL_RHO_LYAP
from utils.online_disturbance_runner import run_online_td3_disturbance_preset

# OF-MPC-pretrained online TD3 with an active GART-LMPC safety gate.
#
# Keep RL_OBSERVATION_MODE as "standard" unless the checkpoint was trained with
# the larger GART observation dimension. The loaded actor can still train online.

EPISODES = 5
SET_POINTS_LEN = 400
SEED = 123
SAVE_PLOTS = True
TIMESTAMP = None

AGENT_PATH = None
RESET_PRETRAINED_CRITIC = True

RL_OBSERVATION_MODE = "standard"
PROJECTION_BACKEND = "direct_accept_or_fallback"

RHO_LYAP = GART_FINAL_RHO_LYAP
LYAP_EPS = GART_FINAL_LYAP_EPS
LYAP_TOL = None

REWARD_FALLBACK_PENALTY_ENABLED = True
GAMMA_FALLBACK = 3.0
FALLBACK_EVENT_PENALTY = 10.0

TRAINING_PHASE_OVERRIDES = {
    # Examples:
    # "behavior_clone_teacher_episodes": 20,
    # "handoff_episodes": 10,
    # "full_rl_exploration_std_start": 0.02,
}


def run_configured_study() -> dict:
    config = {
        "preset": "ofmpc_pretrained_safety_gate",
        "episodes": int(EPISODES),
        "set_points_len": int(SET_POINTS_LEN),
        "seed": int(SEED),
        "save_plots": bool(SAVE_PLOTS),
        "timestamp": TIMESTAMP,
        "agent_path": AGENT_PATH,
        "reset_pretrained_critic": bool(RESET_PRETRAINED_CRITIC),
        "rl_observation_mode": RL_OBSERVATION_MODE,
        "projection_backend": PROJECTION_BACKEND,
        "rho_lyap": float(RHO_LYAP),
        "lyap_eps": float(LYAP_EPS),
        "lyap_tol": LYAP_TOL,
        "reward_fallback_penalty_enabled": bool(REWARD_FALLBACK_PENALTY_ENABLED),
        "gamma_fallback": float(GAMMA_FALLBACK),
        "fallback_event_penalty": float(FALLBACK_EVENT_PENALTY),
        "training_phase_overrides": dict(TRAINING_PHASE_OVERRIDES),
    }
    print("Online TD3 OF-MPC-pretrained safety-gate configuration:")
    pprint(config)
    return run_online_td3_disturbance_preset(
        "ofmpc_pretrained_safety_gate",
        episodes=EPISODES,
        set_points_len=SET_POINTS_LEN,
        seed=SEED,
        save_plots=SAVE_PLOTS,
        agent_path=AGENT_PATH,
        reset_pretrained_critic=RESET_PRETRAINED_CRITIC,
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
    )


if __name__ == "__main__":
    run_configured_study()
