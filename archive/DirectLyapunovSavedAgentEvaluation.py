# Saved-agent evaluation for direct Lyapunov safety-gate RL.
#
# This root entrypoint intentionally mirrors the converted notebook scripts:
# the full experiment setup is visible here, while reusable rollout, plotting,
# and export helpers live in Simulation/saved_agent_evaluation.py.

# %%
from utils.path_helpers import repo_path
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from pprint import pprint

import numpy as np

try:
    import pandas as pd
except Exception:
    pd = None

import torch

# %%
from TD3Agent.reward_functions import make_reward_fn_relative_QR
from Simulation.mpc import MpcSolver, compute_observer_gain
from Simulation.saved_agent_evaluation import (
    SavedAgentEvalContext,
    TD3AgentConfig,
    build_eval_disturbance_profile,
    jsonable,
    make_comparison_plots,
    make_unified_record,
    resolve_agent_paths,
    run_direct_lmpc_case,
    run_mpc_only_case,
    run_rl_saved_agent_case,
    write_csv,
)
from Simulation.system_functions import PolymerCSTR
from Lyapunov.direct_lyapunov_mpc import design_direct_lyapunov_mpc_solver
from utils.direct_lyapunov_study import DIRECT_TWO_SETPOINT_Y_PHYS
from utils.scaling_helpers import apply_min_max
from utils.td3_helpers import load_and_prepare_system_data

# %% [markdown]
# ## User-editable evaluation switches
#
# `AGENT_SOURCE_MODE = "latest"` searches the latest non-`mpc_only` trained
# agent under `results/ColdStart/...` and `results/Pretrain/...`.
# Set manual paths below when you want to freeze a specific checkpoint.

# %%
AGENT_SOURCE_MODE = "latest"
COLD_AGENT_PATH = None
PRETRAIN_AGENT_PATH = None

EVAL_N_EPISODES = 5
EVAL_SET_POINTS_LEN = 400
EVAL_SCENARIO_SUITE = "nominal_qi_qs_ha_all_step"

DRY_RUN = "--dry-run" in sys.argv
SAVE_CASE_PLOTS = True
FORCE_FINAL_TEST = False

study_name = "Compare"
study_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
study_root = Path(repo_path()) / "results" / study_name / study_timestamp

# %% [markdown]
# ## Direct Lyapunov and plant setup

# %%
predict_h = 9
cont_h = 3
rho_lyap = 0.99
lyap_eps = 1e-3
lyap_tol = 1e-10
slack_penalty = 1e6
plant_mode = "nominal"
disturbance_after_step = False
use_target_output_for_tracking = False

u_prev_penalty_weight = 0.1
xs_prev_penalty_weight = 0.1

Ad = 2.142e17
Ed = 14897
Ap = 3.816e10
Ep = 3557
At = 4.50e12
Et = 843
fi = 0.6
m_delta_H_r = -6.99e4
hA = 1.05e6
rhocp = 1506
rhoccpc = 4043
Mm = 104.14
system_params = np.array([Ad, Ed, Ap, Ep, At, Et, fi, m_delta_H_r, hA, rhocp, rhoccpc, Mm])

CIf = 0.5888
CMf = 8.6981
Qi = 108.0
Qs = 459.0
Tf = 330.0
Tcf = 295.0
V = 3000.0
Vc = 3312.4
system_design_params = np.array([CIf, CMf, Qi, Qs, Tf, Tcf, V, Vc])

Qm_ss = 378.0
Qc_ss = 471.6
system_steady_state_inputs = np.array([Qc_ss, Qm_ss])
delta_t = 0.5

steady_states = {"ss_inputs": system_steady_state_inputs.copy()}
cstr_ss = PolymerCSTR(system_params, system_design_params, system_steady_state_inputs, delta_t, deviation_form=False)
steady_states["y_ss"] = cstr_ss.y_ss.copy()

u_min = np.array([71.6, 78.0])
u_max = np.array([870.0, 670.0])
setpoint_y_phys = DIRECT_TWO_SETPOINT_Y_PHYS.copy()

# %% [markdown]
# ## Five-episode fixed disturbance test suite

# %%
n_tests = EVAL_N_EPISODES
set_points_len = EVAL_SET_POINTS_LEN
TEST_CYCLE = [True] * EVAL_N_EPISODES
warm_start = 0
time_in_sub_episodes = int(setpoint_y_phys.shape[0] * set_points_len)

nominal_qs = 459.0
nominal_qi = 108.0
nominal_hA = 1.05e6
qi_change = 0.95
qs_change = 1.05
ha_change = 0.92

scenarios, disturbance_profile = build_eval_disturbance_profile(
    n_eval_episodes=EVAL_N_EPISODES,
    episode_steps=time_in_sub_episodes,
    nominal_qi=nominal_qi,
    nominal_qs=nominal_qs,
    nominal_ha=nominal_hA,
)

# %% [markdown]
# ## Scaling, augmented model, and observer

# %%
system_data = load_and_prepare_system_data(
    steady_states=steady_states,
    setpoint_y=setpoint_y_phys,
    u_min=u_min,
    u_max=u_max,
    system_dict_path=os.path.join("Data", "system_dict"),
    augmentation_style="rawlings",
    augmentation_mode="output_disturbance",
)

A_aug = system_data["A_aug"]
B_aug = system_data["B_aug"]
C_aug = system_data["C_aug"]
data_min = system_data["data_min"]
data_max = system_data["data_max"]
min_max_dict = system_data["min_max_dict"]

inputs_number = int(B_aug.shape[1])
y_sp_scenario = apply_min_max(setpoint_y_phys, data_min[inputs_number:], data_max[inputs_number:]) - apply_min_max(
    steady_states["y_ss"],
    data_min[inputs_number:],
    data_max[inputs_number:],
)

poles = np.array([
    0.44619852,
    0.33547649,
    0.36380595,
    0.70467118,
    0.3562966,
    0.42900673,
    0.4228262,
    0.96916776,
    0.91230187,
])
L = compute_observer_gain(A_aug, C_aug, poles)

# %% [markdown]
# ## TD3 architecture used to reload saved agents

# %%
set_points_number = int(C_aug.shape[0])
STATE_DIM = int(A_aug.shape[0]) + set_points_number + inputs_number
ACTION_DIM = int(B_aug.shape[1])
ACTOR_LAYER_SIZES = [512, 512, 512, 512, 512]
CRITIC_LAYER_SIZES = [512, 512, 512, 512, 512]
BUFFER_CAPACITY = 40000
ACTOR_LR = 5e-5
CRITIC_LR = 5e-4
SMOOTHING_STD = 0.1
NOISE_CLIP = 0.01
GAMMA = 0.995
TAU = 0.005
MAX_ACTION = 1
POLICY_DELAY = 2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 256
STD_START = 0.0
STD_END = 0.01
STD_DECAY_RATE = 0.99992
STD_DECAY_MODE = "exp"
ACTOR_FREEZE = 0

td3_agent_config = TD3AgentConfig(
    state_dim=STATE_DIM,
    action_dim=ACTION_DIM,
    actor_hidden=ACTOR_LAYER_SIZES,
    critic_hidden=CRITIC_LAYER_SIZES,
    gamma=GAMMA,
    actor_lr=ACTOR_LR,
    critic_lr=CRITIC_LR,
    batch_size=BATCH_SIZE,
    policy_delay=POLICY_DELAY,
    target_policy_smoothing_noise_std=SMOOTHING_STD,
    noise_clip=NOISE_CLIP,
    max_action=MAX_ACTION,
    tau=TAU,
    std_start=STD_START,
    std_end=STD_END,
    std_decay_rate=STD_DECAY_RATE,
    std_decay_mode=STD_DECAY_MODE,
    buffer_size=BUFFER_CAPACITY,
    device=DEVICE,
    actor_freeze=ACTOR_FREEZE,
)

# %% [markdown]
# ## Reward function used for evaluation diagnostics

# %%
Qy_diag = np.array([8.0, 6.0])
Su_diag = np.array([1.0, 1.0])
Rdu_diag = np.array([1.0, 1.0])
k_rel = np.array([0.0015, 0.00015])
band_floor_phys = np.array([0.003, 0.035])
gamma_fallback = 3.0
fallback_event_penalty = 10.0

reward_config, reward_fn = make_reward_fn_relative_QR(
    data_min=data_min,
    data_max=data_max,
    n_inputs=inputs_number,
    k_rel=k_rel,
    band_floor_phys=band_floor_phys,
    Q_diag=Qy_diag,
    R_diag=Rdu_diag,
    tau_frac=0.5,
    gamma_out=1.0,
    gamma_in=3.0,
    beta=1.0,
    gate="geom",
    lam_in=3.0,
    bonus_kind="quadratic",
    gamma_fallback=gamma_fallback,
    fallback_event_penalty=fallback_event_penalty,
    R_fallback_diag=Rdu_diag,
    maintenance_band_scale=0.5,
    maintenance_move_weight=0.0,
    jitter_weight=0.0,
    dwell_bonus=0.0,
)

# %% [markdown]
# ## Direct Lyapunov MPC and offset-free MPC solvers

# %%
u_ss = apply_min_max(steady_states["ss_inputs"], data_min[:inputs_number], data_max[:inputs_number])
b_min = apply_min_max(u_min, data_min[:inputs_number], data_max[:inputs_number])
b_max = apply_min_max(u_max, data_min[:inputs_number], data_max[:inputs_number])
b1 = (b_min[0] - u_ss[0], b_max[0] - u_ss[0])
b2 = (b_min[1] - u_ss[1], b_max[1] - u_ss[1])
bnds = (b1, b2) * cont_h
IC_opt_template = np.zeros(inputs_number * cont_h)

u_min_scaled = apply_min_max(u_min, data_min[:inputs_number], data_max[:inputs_number])
u_max_scaled = apply_min_max(u_max, data_min[:inputs_number], data_max[:inputs_number])
u_dev_min = u_min_scaled - u_ss
u_dev_max = u_max_scaled - u_ss

LMPC_obj = design_direct_lyapunov_mpc_solver(
    A_aug=A_aug,
    B_aug=B_aug,
    C_aug=C_aug,
    Qy_diag=Qy_diag,
    NP=predict_h,
    NC=cont_h,
    Su_diag=Su_diag,
    u_min=u_dev_min,
    u_max=u_dev_max,
    Rdu_diag=Rdu_diag,
    terminal_set_on=True,
    terminal_alpha_scale=1.0,
)
MPC_obj_offset_free = MpcSolver(
    A_aug,
    B_aug,
    C_aug,
    Q_out=Qy_diag,
    R_in=Rdu_diag,
    NP=predict_h,
    NC=cont_h,
)

direct_target_config = {
    "u_ref_weight": float(u_prev_penalty_weight),
    "x_ref_weight": float(xs_prev_penalty_weight),
}

# %% [markdown]
# ## Evaluation context passed to reusable helpers

# %%
eval_context = SavedAgentEvalContext(
    study_name=study_name,
    study_root=study_root,
    scenario_suite=EVAL_SCENARIO_SUITE,
    n_tests=n_tests,
    set_points_len=set_points_len,
    test_cycle=TEST_CYCLE,
    warm_start=warm_start,
    time_in_sub_episodes=time_in_sub_episodes,
    force_final_test=FORCE_FINAL_TEST,
    save_case_plots=SAVE_CASE_PLOTS,
    td3_agent_config=td3_agent_config,
    system_params=system_params,
    system_design_params=system_design_params,
    system_steady_state_inputs=system_steady_state_inputs,
    delta_t=delta_t,
    steady_states=steady_states,
    min_max_dict=min_max_dict,
    data_min=data_min,
    data_max=data_max,
    inputs_number=inputs_number,
    y_sp_scenario=y_sp_scenario,
    L=L,
    LMPC_obj=LMPC_obj,
    MPC_obj_offset_free=MPC_obj_offset_free,
    reward_fn=reward_fn,
    reward_config=reward_config,
    gamma=GAMMA,
    rho_lyap=rho_lyap,
    lyap_eps=lyap_eps,
    lyap_tol=lyap_tol,
    slack_penalty=slack_penalty,
    fallback_event_penalty=fallback_event_penalty,
    plant_mode=plant_mode,
    disturbance_after_step=disturbance_after_step,
    use_target_output_for_tracking=use_target_output_for_tracking,
    IC_opt_template=IC_opt_template,
    bnds=bnds,
    direct_target_config=direct_target_config,
    nominal_qi=nominal_qi,
    nominal_qs=nominal_qs,
    nominal_hA=nominal_hA,
    qi_change=qi_change,
    qs_change=qs_change,
    ha_change=ha_change,
)

# %% [markdown]
# ## Resolve saved agents and show dry-run plan

# %%
cold_agent_path, pretrain_agent_path = resolve_agent_paths(
    agent_source_mode=AGENT_SOURCE_MODE,
    cold_agent_path=COLD_AGENT_PATH,
    pretrain_agent_path=PRETRAIN_AGENT_PATH,
)

planned = {
    "study_root": study_root,
    "cold_agent_path": cold_agent_path,
    "pretrain_agent_path": pretrain_agent_path,
    "scenario_suite": EVAL_SCENARIO_SUITE,
    "scenarios": scenarios,
    "n_eval_episodes": EVAL_N_EPISODES,
    "set_points_len": EVAL_SET_POINTS_LEN,
    "time_in_sub_episodes": time_in_sub_episodes,
    "n_steps": int(len(disturbance_profile["qi"])),
    "controllers": ["cold_saved_rl", "pretrained_saved_rl", "mpc_only", "direct_lmpc"],
}

if DRY_RUN:
    print("Saved-agent evaluation dry run:")
    pprint(jsonable(planned))

# %% [markdown]
# ## Run saved-agent evaluation and export artifacts

# %%
if not DRY_RUN:
    study_root.mkdir(parents=True, exist_ok=True)
    print(f"Saving saved-agent evaluation artifacts under: {study_root}")
    print("Using agents:")
    print(f"  cold: {cold_agent_path}")
    print(f"  pretrained: {pretrain_agent_path}")

    bundles = {}
    records = []
    summary_records = []

    print("Running cold_saved_rl")
    bundle, debug_dir, record = run_rl_saved_agent_case(
        eval_context,
        case_name="cold_saved_rl",
        agent_path=cold_agent_path,
        scenarios=scenarios,
        profile=disturbance_profile,
    )
    bundles["cold_saved_rl"] = bundle
    records.append(record)
    summary_records.append(
        make_unified_record(eval_context, "cold_saved_rl", "saved_rl_safety_gate", bundle, record, debug_dir)
    )
    pprint(summary_records[-1])

    print("Running pretrained_saved_rl")
    bundle, debug_dir, record = run_rl_saved_agent_case(
        eval_context,
        case_name="pretrained_saved_rl",
        agent_path=pretrain_agent_path,
        scenarios=scenarios,
        profile=disturbance_profile,
    )
    bundles["pretrained_saved_rl"] = bundle
    records.append(record)
    summary_records.append(
        make_unified_record(eval_context, "pretrained_saved_rl", "saved_rl_safety_gate", bundle, record, debug_dir)
    )
    pprint(summary_records[-1])

    print("Running mpc_only")
    bundle, debug_dir, record = run_mpc_only_case(
        eval_context,
        scenarios=scenarios,
        profile=disturbance_profile,
    )
    bundles["mpc_only"] = bundle
    records.append(record)
    summary_records.append(
        make_unified_record(eval_context, "mpc_only", "offset_free_mpc_diagnostic", bundle, record, debug_dir)
    )
    pprint(summary_records[-1])

    print("Running direct_lmpc")
    bundle, debug_dir, record = run_direct_lmpc_case(
        eval_context,
        scenarios=scenarios,
        profile=disturbance_profile,
    )
    bundles["direct_lmpc"] = bundle
    records.append(record)
    summary_records.append(
        make_unified_record(eval_context, "direct_lmpc", "direct_lyapunov_mpc", bundle, record, debug_dir)
    )
    pprint(summary_records[-1])

    comparison_csv = study_root / "comparison_table.csv"
    write_csv(comparison_csv, summary_records)

    raw_records_csv = study_root / "raw_comparison_records.csv"
    write_csv(raw_records_csv, records)

    scenarios_csv = study_root / "scenario_table.csv"
    write_csv(scenarios_csv, scenarios)

    figures_dir = study_root / "figures"
    plot_paths = make_comparison_plots(eval_context, summary_records, bundles, figures_dir)

    summary = {
        **planned,
        "completed_at": datetime.now().isoformat(timespec="seconds"),
        "comparison_table": comparison_csv,
        "raw_comparison_records": raw_records_csv,
        "scenario_table": scenarios_csv,
        "plot_paths": plot_paths,
    }
    with (study_root / "evaluation_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(jsonable(summary), handle, indent=2)

    print("Saved-agent evaluation complete.")
    print(f"Comparison table: {comparison_csv}")
