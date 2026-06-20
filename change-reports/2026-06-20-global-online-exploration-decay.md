# Global Online Exploration Decay

## Objective

Make online TD3 exploration use one global decay schedule from episode 1 to the
end of the run, including teacher/warm-up, handoff, and full-RL phases.

## Change

The active online runner defaults now use:

```python
PRETRAINED_EXPLORATION_STD_START = 0.05
COLD_START_EXPLORATION_STD_START = 0.1
GLOBAL_EXPLORATION_STD_END = 0.005
PRETRAINED_BC_EXPLORATION_STD = 0.05
COLD_START_BC_EXPLORATION_STD = 0.1
PRETRAINED_HANDOFF_EXPLORATION_STD_START = 0.05
PRETRAINED_HANDOFF_EXPLORATION_STD_END = 0.005
COLD_START_HANDOFF_EXPLORATION_STD_START = 0.1
COLD_START_HANDOFF_EXPLORATION_STD_END = 0.005
```

`utils.online_disturbance_runner._training_phase_config(...)` now emits:

```python
"global_exploration_schedule": True
"exploration_decay_mode": "linear"
"exploration_std_start": exploration_std
"exploration_std_end": GLOBAL_EXPLORATION_STD_END
```

The noisy-teacher override helpers also report handoff start sigma equal to the
same initial teacher sigma instead of `0.0`.

`Simulation.run_rl_lyapunov._phase_exploration_sigma(...)` now honors that
global schedule before phase-specific constants whenever the current behavior
uses Gaussian exploration.

## Mathematical Interpretation

For a run with total step count `N`, the exploration scale is:

$$
\sigma_k = \sigma_0 + (\sigma_f-\sigma_0)
\frac{k}{N-1},
\qquad 0 \le k < N.
$$

The schedule is:

- pretrained online runners: $\sigma_0 = 0.05$, $\sigma_f = 0.005$
- cold-start online runners: $\sigma_0 = 0.1$, $\sigma_f = 0.005$

The same $\sigma_k$ is used in the teacher/BC, handoff, and full-RL phases when
their behavior noise mode is Gaussian.

## Validation

Passed:

```powershell
python -m py_compile utils/online_disturbance_runner.py Simulation/run_rl_lyapunov.py OnlineTD3_ColdStart_SafetyGate.py OnlineTD3_OFMPCPretrained_SafetyGate.py OnlineTD3_ColdStart_NoSafetyGate.py OnlineTD3_OFMPCPretrained_NoSafetyGate.py
```

The successful local run redirected `PYTHONPYCACHEPREFIX` to a Windows temp
directory to avoid the known OneDrive `.validation-pyc` pycache permission issue.

An optional import-level schedule sanity check was not completed because the
default Python in this shell does not have `torch` installed.
