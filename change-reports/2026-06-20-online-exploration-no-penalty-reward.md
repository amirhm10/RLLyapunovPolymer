# Online Exploration and No-Penalty Safety Reward

## Objective

Adjust the active online TD3 runners after the latest result review:

- increase Gaussian exploration because it is applied in scaled-deviation input
  coordinates with a wide range, about `[-10, 9.96]` and `[-7.5, 7.3]`
- make the safety-gate runners train on the same no-penalty reward used for
  fair cross-controller control-performance comparisons

## Evidence From Latest Results

The latest completed OF-MPC-pretrained safety-gate run inspected was:

```text
results/OnlineTD3_OFMPCPretrained_SafetyGate/20260619_174130
```

Its `run_summary.json` reported:

- `u_min_dev = [-10.0, -7.5]`
- `u_max_dev = [9.96, 7.3]`
- teacher-phase `bc_exploration_std = 0.002`
- full-RL exploration `0.02 -> 0.005`
- `reward_mean = -6.766`
- `reward_no_penalty_mean = -5.564`
- `fallback_penalty_sum = 288578.47`

The step table confirmed that the teacher phase used `input_dev` exploration
with `sigma = 0.002`. Relative to the input-deviation span, this was only about
0.01 percent of the available range in each input channel.

## Changes

`utils/online_disturbance_runner.py` now uses a larger input-deviation
exploration schedule:

```python
PRETRAINED_EXPLORATION_STD_START = 0.05
COLD_START_EXPLORATION_STD_START = 0.1
FULL_RL_EXPLORATION_STD_END = 0.02
PRETRAINED_BC_EXPLORATION_STD = 0.05
COLD_START_BC_EXPLORATION_STD = 0.05
PRETRAINED_HANDOFF_EXPLORATION_STD_END = 0.05
COLD_START_HANDOFF_EXPLORATION_STD_END = 0.05
```

The active safety-gate root runners now disable fallback/event reward penalties:

```python
REWARD_FALLBACK_PENALTY_ENABLED = False
GAMMA_FALLBACK = 0.0
FALLBACK_EVENT_PENALTY = 0.0
```

This applies to:

- `OnlineTD3_ColdStart_SafetyGate.py`
- `OnlineTD3_OFMPCPretrained_SafetyGate.py`

No-gate runners were already configured with no fallback penalty.

## Mathematical Interpretation

The behavior noise is now applied in scaled input-deviation coordinates:

$$
u_{\mathrm{beh},k}
= \operatorname{clip}(u_{\mathrm{nom},k}+\epsilon_k,
u_{\min},u_{\max}),
\qquad
\epsilon_k \sim \mathcal{N}(0,\sigma_u^2 I).
$$

The safety-runner reward now removes the fallback penalty from the scalar TD3
training reward:

$$
r_k = r_{\mathrm{track/move},k}.
$$

The safety gate still changes the executed action when needed, but the actor is
not additionally penalized by `gamma_fallback` or a fixed fallback event cost.

## Validation

Passed:

```powershell
python -m py_compile utils/online_disturbance_runner.py OnlineTD3_ColdStart_SafetyGate.py OnlineTD3_OFMPCPretrained_SafetyGate.py OnlineTD3_ColdStart_NoSafetyGate.py OnlineTD3_OFMPCPretrained_NoSafetyGate.py
```

The successful local run redirected `PYTHONPYCACHEPREFIX` to a Windows temp
directory to avoid the known OneDrive `.validation-pyc` pycache permission issue.
