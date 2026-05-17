# Direct Disturbance Diagnosis And Literature Update

## What changed
- Added a new report:
  - `report/direct_lyapunov_disturbance_gap_analysis_2026-05-17.md`
- Extended the main method report:
  - `report/direct_lyapunov_direct_and_rl_method_2026-05-16.md`
- Added a new figure and CSV summary under:
  - `report/figures/2026-05-17_direct_disturbance_gap/`

## Why
- The direct no-RL notebook behaves well in nominal runs but degrades under disturbance.
- The update was meant to move from a purely empirical observation to a clearer control-theoretic diagnosis.
- The new analysis shows that the disturbed notebook currently combines:
  - raw-setpoint tracking in the MPC stage,
  - a Lyapunov certificate centered on an admissible target,
  - and a frozen output-disturbance model even though the simulated disturbance changes plant parameters and flow terms.

## Main conclusions recorded
- The disturbed problem is not explained by weight tuning alone.
- The best saved 3200-step disturbed run currently uses joint regularization with:
  - `u_ref_weight = 0.1`
  - `x_ref_weight = 0.1`
- A proper remedy path should first align the tracking reference with the admissible target in the disturbed test, then redesign the disturbance model and observer to match the actual disturbance channel, and only then consider a stronger robust tracking MPC architecture if needed.

## Literature added
- Muske and Badgwell (2002) on general disturbance models for offset-free MPC
- Pannocchia (2024) tutorial review on offset-free tracking MPC formulations
- Shead, Muske, and Rossiter (2008) on constrained target-selection failure modes
- Pannocchia and Bemporad (2007) on combined disturbance-model and observer design
- Limon et al. (2010) on robust tube-based tracking MPC
- Tatjewski (2026) on tunable disturbance estimates in offset-free MPC

## Validation
- Read the current disturbed notebook settings from `DirectLyapunovMPC_FourMethodDisturbance.ipynb`
- Verified the plant disturbance path in `Simulation/system_functions.py`
- Verified the disturbance-model assumptions in `Lyapunov/frozen_output_disturbance_target.py`
- Aggregated the saved 3200-step disturbed sweep summaries from:
  - `results/direct_lyapunov_mpc_bounded_four_method_two_setpoint_disturb/`
