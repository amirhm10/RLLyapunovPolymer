# Online TD3 Target Smoothing Clips

## Objective

Correct the online TD3 target-policy smoothing clips so they scale with the target smoothing standard deviation.

The previous online runner used one shared clip:

```text
NOISE_CLIP = 0.01
```

That made cold-start TD3 inconsistent because its target smoothing standard deviation was `0.1`, but the target noise was clipped at only `0.01`.

## Change

`utils/online_disturbance_runner.py` now uses preset-specific TD3 target smoothing values:

| Runner family | Target smoothing std | Target noise clip |
|---|---:|---:|
| OF-MPC-pretrained | `0.02` | `0.04` |
| Cold-start | `0.1` | `0.2` |

The online TD3 agent constructor now receives `noise_clip` explicitly from the selected preset family instead of using a single shared global value.

## Method Interpretation

TD3 target-policy smoothing uses:

$$
a' = \mathrm{clip}\left(\pi_{\bar{\theta}}(s') + \epsilon\right),
\qquad
\epsilon \sim \mathrm{clip}\left(\mathcal{N}(0,\sigma), [-c,c]\right).
$$

With this update:

$$
c = 2\sigma
$$

for both pretrained and cold-start online TD3.

This changes only the critic target-action smoothing used in TD learning. It does not change behavior exploration, the PER+recency replay buffer, actor/critic learning rates, or the pretrained critic-retention setting.

## Validation

Passed syntax validation:

```powershell
$env:PYTHONPYCACHEPREFIX = Join-Path $env:TEMP 'codex-pycache-lyapunov-polymer'
& "C:\Users\HAMEDI\miniconda3\envs\rl\python.exe" -m py_compile "utils\online_disturbance_runner.py" "OnlineTD3_OFMPCPretrained_SafetyGate.py" "OnlineTD3_OFMPCPretrained_NoSafetyGate.py" "OnlineTD3_ColdStart_SafetyGate.py" "OnlineTD3_ColdStart_NoSafetyGate.py"
```

Passed a construction smoke test confirming:

- pretrained online agent: `t_std = 0.02`, `noise_clip = 0.04`
- cold-start online agent: `t_std = 0.1`, `noise_clip = 0.2`

## Recommended Comparison

Use the next pretrained online runs to compare:

- retained pretrained critic with lower behavior exploration
- updated target smoothing `0.02/0.04`
- safety-gate versus no-safety-gate behavior

Watch the critic loss and early handoff tracking. If the retained critic is useful, the early online transient should improve without a large value-loss spike.
