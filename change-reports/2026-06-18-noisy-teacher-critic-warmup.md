# Noisy Teacher-Critic Warmup for Online TD3

## Objective

Replace the active online TD3 runners' behavior-cloning teacher phase with a safer noisy teacher-behavior critic warmup.

## Change

The four active online TD3 root runners now use the shared Alternative B phase template:

```python
default_noisy_teacher_critic_warmup_overrides(...)
```

The effective default schedule is:

- 10 noisy GART-LMPC teacher-behavior episodes
- `bc_update_mode = "critic_td_only"`
- no actor behavioral-cloning updates during the teacher phase
- 10 linear handoff episodes
- full TD3 actor updates only during handoff and full-RL phases

The shared runner also exposes Alternative A:

```python
noisy_teacher_buffer_warmup_overrides(...)
```

Alternative A uses noisy teacher behavior to fill replay only before handoff. `Simulation/run_rl_lyapunov.py` now supports `warmup_exploration_std` so this buffer-only warmup uses the intended teacher-noise scale instead of the generic full-RL exploration schedule.

## Active Runners Updated

- `OnlineTD3_ColdStart_SafetyGate.py`
- `OnlineTD3_ColdStart_NoSafetyGate.py`
- `OnlineTD3_OFMPCPretrained_SafetyGate.py`
- `OnlineTD3_OFMPCPretrained_NoSafetyGate.py`

## Archive Update

The LMPC-pretrained root wrappers were moved to `archive/`:

- `archive/OnlineTD3_LMPCPretrained_SafetyGate.py`
- `archive/OnlineTD3_LMPCPretrained_NoSafetyGate.py`

The shared LMPC-pretrained presets and main functions remain available in `utils/online_disturbance_runner.py` for historical analysis and compatibility scripts.

## Validation

Passed:

```powershell
C:\Users\hamediaa\.conda\envs\rl-env\python.exe -m py_compile utils\online_disturbance_runner.py Simulation\run_rl_lyapunov.py OnlineTD3_ColdStart_SafetyGate.py OnlineTD3_ColdStart_NoSafetyGate.py OnlineTD3_OFMPCPretrained_SafetyGate.py OnlineTD3_OFMPCPretrained_NoSafetyGate.py archive\OnlineTD3_LMPCPretrained_SafetyGate.py archive\OnlineTD3_LMPCPretrained_NoSafetyGate.py
```

Short 1-episode, `SET_POINTS_LEN = 2`, no-plot smoke runs passed for all four active runners:

- `OnlineTD3_ColdStart_SafetyGate.py`
- `OnlineTD3_ColdStart_NoSafetyGate.py`
- `OnlineTD3_OFMPCPretrained_SafetyGate.py`
- `OnlineTD3_OFMPCPretrained_NoSafetyGate.py`

Run-summary and first-step table readback confirmed:

- `behavior_clone_teacher_episodes = 10`
- `bc_update_mode = "critic_td_only"`
- `handoff_episodes = 10`
- `actor_bc_update_active = False` during the initial noisy teacher phase
- `td3_full_update_active = False` during the initial noisy teacher phase
