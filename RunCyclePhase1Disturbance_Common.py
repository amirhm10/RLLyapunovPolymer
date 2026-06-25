from __future__ import annotations

from argparse import Namespace
from datetime import datetime
from pathlib import Path
from pprint import pprint

from RunOnlineTD3TwoPhaseStudy import run_two_phase_study
import utils.online_disturbance_runner as online_runner


SETPOINT_CYCLE_Y_PHYS = (
    (4.0, 321.5),
    (3.3, 324.5),
)

PHASE1_EPISODES = 100
PHASE2_EPISODES = 1
PHASE1_SETPOINT_HOLD_STEPS = 400
REPORTING_WINDOW_STEPS = 800

OUTPUT_ROOT = Path("results")
SAVE_PLOTS = True
EXPORT_PROFILE = "compact"

N_SEEDS = 1
SEED_START = 0
SEEDS: tuple[int, ...] | None = (0,)

NOMINAL_QI = 108.0
NOMINAL_QS = 459.0
NOMINAL_HA = 1.05e6

PHASE1_QI_MULTIPLIER = 0.95
PHASE1_QS_MULTIPLIER = 1.05
PHASE1_HA_MULTIPLIER = 0.92

# Keep Phase 2 at the Phase-1 final disturbance value for this scenario.
PHASE2_QI_MULTIPLIER = PHASE1_QI_MULTIPLIER
PHASE2_QS_MULTIPLIER = PHASE1_QS_MULTIPLIER
PHASE2_HA_MULTIPLIER = PHASE1_HA_MULTIPLIER

DEFAULT_SAVED_AGENT_PATH = (
    Path("results")
    / "OnlineTD3_TwoPhaseStudy"
    / "20260623_092655_cold_start_safety_gate"
    / "seed_009"
    / "cold_start_safety_gate"
    / "onlinetd3_coldstart_safetygate"
    / "trained_agent_20260624_093558.pkl"
)

ACTOR_LAYER_SIZES = (512, 512, 512, 512, 512)
CRITIC_LAYER_SIZES = (512, 512, 512, 512, 512)
REPLAY_BUFFER_CAPACITY = 80000
BATCH_SIZE = 256
ACTOR_LR = 5e-5
CRITIC_LR = 5e-4

COLD_TARGET_POLICY_SMOOTHING_NOISE_STD = 0.1
COLD_TARGET_POLICY_NOISE_CLIP = 0.2
COLD_EXPLORATION_STD_START = 0.1

SAVED_TARGET_POLICY_SMOOTHING_NOISE_STD = 0.02
SAVED_TARGET_POLICY_NOISE_CLIP = 0.04
SAVED_EXPLORATION_STD_START = 0.01

EXPLORATION_STD_END = 0.01
RL_OBSERVATION_MODE = "standard"
REWARD_FALLBACK_PENALTY_ENABLED = False
GAMMA_FALLBACK = 0.0
FALLBACK_EVENT_PENALTY = 0.0
RHO_LYAP = online_runner.GART_FINAL_RHO_LYAP
LYAP_EPS = online_runner.GART_FINAL_LYAP_EPS
LYAP_TOL = None


def seed_arg(seeds: tuple[int, ...] | None = SEEDS) -> str | None:
    if seeds is None:
        return None
    return ",".join(str(seed) for seed in seeds)


def methods_label(methods: tuple[str, ...]) -> str:
    return "_".join(str(method).strip() for method in methods if str(method).strip())


def no_teacher_training_overrides(*, exploration_std_start: float) -> dict:
    return {
        "warmup_buffer_only_episodes": 0,
        "behavior_clone_teacher_episodes": 0,
        "handoff_episodes": 0,
        "warmup_behavior_source": "gart_lmpc",
        "bc_teacher_policy": "gart_lmpc",
        "bc_behavior_source": "gart_lmpc",
        "bc_update_mode": "critic_td_only",
        "handoff_update_mode": "td3_full",
        "bc_actor_updates_per_step": 0,
        "handoff_actor_bc_updates_per_step": 0,
        "warmup_behavior_noise": "none",
        "bc_behavior_noise": "none",
        "handoff_behavior_noise": "none",
        "warmup_exploration_std": float(exploration_std_start),
        "bc_exploration_std": float(exploration_std_start),
        "handoff_exploration_std_start": float(exploration_std_start),
        "handoff_exploration_std_end": float(EXPLORATION_STD_END),
        "full_rl_exploration_std_start": float(exploration_std_start),
        "full_rl_exploration_std_end": float(EXPLORATION_STD_END),
        "exploration_std_start": float(exploration_std_start),
        "exploration_std_end": float(EXPLORATION_STD_END),
    }


def apply_td3_defaults() -> None:
    online_runner.DEFAULT_ACTOR_LAYER_SIZES = tuple(int(v) for v in ACTOR_LAYER_SIZES)
    online_runner.DEFAULT_CRITIC_LAYER_SIZES = tuple(int(v) for v in CRITIC_LAYER_SIZES)
    online_runner.BUFFER_CAPACITY = int(REPLAY_BUFFER_CAPACITY)
    online_runner.BATCH_SIZE = int(BATCH_SIZE)
    online_runner.ACTOR_LR = float(ACTOR_LR)
    online_runner.CRITIC_LR = float(CRITIC_LR)

    online_runner.COLD_START_SMOOTHING_STD = float(COLD_TARGET_POLICY_SMOOTHING_NOISE_STD)
    online_runner.COLD_START_NOISE_CLIP = float(COLD_TARGET_POLICY_NOISE_CLIP)
    online_runner.COLD_START_EXPLORATION_STD_START = float(COLD_EXPLORATION_STD_START)
    online_runner.COLD_START_BC_EXPLORATION_STD = float(COLD_EXPLORATION_STD_START)
    online_runner.COLD_START_HANDOFF_EXPLORATION_STD_START = float(COLD_EXPLORATION_STD_START)
    online_runner.COLD_START_HANDOFF_EXPLORATION_STD_END = float(EXPLORATION_STD_END)
    online_runner.COLD_START_HANDOFF_EPISODES = 0

    online_runner.PRETRAINED_SMOOTHING_STD = float(SAVED_TARGET_POLICY_SMOOTHING_NOISE_STD)
    online_runner.PRETRAINED_NOISE_CLIP = float(SAVED_TARGET_POLICY_NOISE_CLIP)
    online_runner.PRETRAINED_EXPLORATION_STD_START = float(SAVED_EXPLORATION_STD_START)
    online_runner.PRETRAINED_BC_EXPLORATION_STD = float(SAVED_EXPLORATION_STD_START)
    online_runner.PRETRAINED_HANDOFF_EXPLORATION_STD_START = float(SAVED_EXPLORATION_STD_START)
    online_runner.PRETRAINED_HANDOFF_EXPLORATION_STD_END = float(EXPLORATION_STD_END)
    online_runner.PRETRAINED_HANDOFF_EPISODES = 0
    online_runner.GLOBAL_EXPLORATION_STD_END = float(EXPLORATION_STD_END)


def build_args(
    *,
    methods: tuple[str, ...],
    timestamp_label: str,
    saved_agent_path: str | Path | None = None,
    setpoint_cycle_y_phys=SETPOINT_CYCLE_Y_PHYS,
    n_seeds: int = N_SEEDS,
    seed_start: int = SEED_START,
    seeds: tuple[int, ...] | None = SEEDS,
) -> Namespace:
    checkpoint_initialized = any(str(method).startswith("saved_agent") for method in methods)
    exploration_start = SAVED_EXPLORATION_STD_START if checkpoint_initialized else COLD_EXPLORATION_STD_START
    timestamp = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{timestamp_label}"
    return Namespace(
        n_seeds=int(n_seeds),
        seed_start=int(seed_start),
        seeds=seed_arg(seeds),
        methods=",".join(methods),
        output_root=str(OUTPUT_ROOT),
        timestamp=timestamp,
        save_plots=bool(SAVE_PLOTS),
        export_profile=str(EXPORT_PROFILE),
        agent_path=None if saved_agent_path is None else str(saved_agent_path),
        reset_pretrained_critic=False,
        rl_observation_mode=RL_OBSERVATION_MODE,
        projection_backend=None,
        reward_fallback_penalty_enabled=bool(REWARD_FALLBACK_PENALTY_ENABLED),
        gamma_fallback=float(GAMMA_FALLBACK),
        fallback_event_penalty=float(FALLBACK_EVENT_PENALTY),
        rho_lyap=float(RHO_LYAP),
        lyap_eps=float(LYAP_EPS),
        lyap_tol=LYAP_TOL,
        training_phase_overrides=no_teacher_training_overrides(exploration_std_start=exploration_start),
        phase1_episodes=int(PHASE1_EPISODES),
        phase2_episodes=int(PHASE2_EPISODES),
        phase2_steps=None,
        set_points_len=int(PHASE1_SETPOINT_HOLD_STEPS),
        reporting_window_steps=int(REPORTING_WINDOW_STEPS),
        phase1_setpoints_y_phys=setpoint_cycle_y_phys,
        phase2_setpoints_y_phys=setpoint_cycle_y_phys,
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


def run_configured_cycle_study(
    *,
    methods: tuple[str, ...],
    timestamp_label: str,
    saved_agent_path: str | Path | None = None,
    setpoint_cycle_y_phys=SETPOINT_CYCLE_Y_PHYS,
) -> dict:
    apply_td3_defaults()
    args = build_args(
        methods=methods,
        timestamp_label=timestamp_label,
        saved_agent_path=saved_agent_path,
        setpoint_cycle_y_phys=setpoint_cycle_y_phys,
    )
    print("Cycle/Phase-1-disturbance TD3 configuration:")
    pprint(vars(args))
    return run_two_phase_study(args)
