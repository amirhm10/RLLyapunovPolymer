# Pretrained Online Critic Retention And Lower Exploration

## Objective

Evaluate the newly trained OF-MPC-pretrained TD3 agent online without resetting its critic, and reduce pretrained online exploration from `0.05` to `0.02`.

The goal is to test whether the freshly pretrained actor-critic pair can transfer more smoothly when online exploration is less aggressive and the critic is retained.

## Change

- `utils/online_disturbance_runner.py`
  - `PRETRAINED_EXPLORATION_STD_START`: `0.05` to `0.02`
  - `PRETRAINED_BC_EXPLORATION_STD`: `0.05` to `0.02`
  - `PRETRAINED_HANDOFF_EXPLORATION_STD_START`: `0.05` to `0.02`
  - `DEFAULT_RESET_PRETRAINED_CRITIC`: `True` to `False`
- `OnlineTD3_OFMPCPretrained_SafetyGate.py`
  - `RESET_PRETRAINED_CRITIC = False`
- `OnlineTD3_OFMPCPretrained_NoSafetyGate.py`
  - `RESET_PRETRAINED_CRITIC = False`

This keeps the pretrained actor and pretrained critic when running the OF-MPC-pretrained online runners.

## Method Interpretation

The pretrained TD3 checkpoint provides an initial actor and critic:

$$
\pi_{\theta_0}(s), \quad Q_{\phi_0}(s,a).
$$

Previously, the online pretrained runs retained $\pi_{\theta_0}$ but reset $Q_{\phi_0}$. With this change, both are retained:

$$
\theta_{\mathrm{online},0} = \theta_0,
\qquad
\phi_{\mathrm{online},0} = \phi_0.
$$

The behavior exploration during pretrained teacher warmup, actor-cloning correction, and handoff now starts from `0.02` in input-deviation space instead of `0.05`. TD3 target-policy smoothing remains unchanged.

## Validation

Passed syntax validation:

```powershell
$env:PYTHONPYCACHEPREFIX = Join-Path $env:TEMP 'codex-pycache-lyapunov-polymer'
& "C:\Users\HAMEDI\miniconda3\envs\rl\python.exe" -m py_compile "OnlineTD3_OFMPCPretrained_SafetyGate.py" "OnlineTD3_OFMPCPretrained_NoSafetyGate.py" "utils\online_disturbance_runner.py"
```

Passed an import-level configuration check confirming:

- safety-gate pretrained runner uses `RESET_PRETRAINED_CRITIC = False`
- no-safety-gate pretrained runner uses `RESET_PRETRAINED_CRITIC = False`
- pretrained warmup, BC, and handoff exploration values are `0.02`

## Recommended Comparison

Run both OF-MPC-pretrained online variants with the latest checkpoint:

- `OnlineTD3_OFMPCPretrained_SafetyGate.py`
- `OnlineTD3_OFMPCPretrained_NoSafetyGate.py`

Compare against the previous reset-critic runs using:

- no-penalty reward
- output IAE/RMSE
- fallback or would-be fallback count
- critic and actor loss trends
- early handoff tracking error

The key diagnostic is whether retaining the critic reduces the early online relearning transient without locking the policy into a bad value estimate.
