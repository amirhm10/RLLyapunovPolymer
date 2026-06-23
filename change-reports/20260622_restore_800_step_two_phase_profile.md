# Restore 800-Step Two-Phase Profile

Date: 2026-06-22

## Objective

Restore the two-phase online study to true 800-sample episodes and use the more stable Data-folder pretrained TD3 agent. Phase 2 now keeps the same two-setpoint schedule as Phase 1 and changes only the disturbance profile.

## Changes

- Switched pretrained OF-MPC online runners to:

```python
AGENT_PATH = Path("Data") / "agent_2507171027.pkl"
```

- Matched all TD3 runner network sizes to the stable checkpoint:

```python
ACTOR_LAYER_SIZES = (512, 512, 512, 512, 512)
CRITIC_LAYER_SIZES = (512, 512, 512, 512, 512)
```

- Restored 800-sample reporting/training episodes:

```python
PHASE1_SETPOINT_HOLD_STEPS = 400
REPORTING_WINDOW_STEPS = 800
```

- Replaced fixed `PHASE2_STEPS = 10000` with:

```python
PHASE2_EPISODES = 50
```

- Set Phase 2 setpoints equal to Phase 1:

```python
PHASE2_SETPOINTS_Y_PHYS = (
    (4.5, 324.0),
    (3.4, 321.0),
)
```

- Updated the shared profile builder to support `phase2_episodes` directly while keeping optional fixed-step Phase 2 support.

## Resulting Default Profile

- Phase 1: 150 episodes x 800 steps = 120000 steps.
- Phase 2: 50 episodes x 800 steps = 40000 steps.
- Total: 200 episodes/windows = 160000 steps.
- Rollout call: `n_tests = 200`, `set_points_len = 400`.
- Disturbance is continuous at the Phase 1 to Phase 2 boundary and ramps from the Phase-1 final disturbance to the Phase-2 final disturbance over the 50 Phase-2 episodes.

## Validation

- Verified the old Data checkpoint resolves as:
  - state dimension: 13
  - action dimension: 2
  - actor layers: `(512, 512, 512, 512, 512)`
  - critic layers: `(512, 512, 512, 512, 512)`
- Verified the built profile:
  - `phase1_episode_len = 800`
  - `reporting_window_steps = 800`
  - `phase1_reporting_windows = 150`
  - `phase2_reporting_windows = 50`
  - `total_steps = 160000`
  - `rollout_n_tests = 200`
  - `rollout_set_points_len = 400`
- Compiled touched modules and runners with bytecode redirected away from OneDrive:

```powershell
$env:PYTHONPYCACHEPREFIX="$env:TEMP\codex_pycache"
C:\Users\HAMEDI\miniconda3\envs\rl\python.exe -m py_compile RunOnlineTD3TwoPhaseStudy.py utils\two_phase_profiles.py RunTwoPhase_OFMPCPretrained_SafetyGate.py RunTwoPhase_OFMPCPretrained_NoSafetyGate.py RunTwoPhase_ColdStart_SafetyGate.py RunTwoPhase_ColdStart_NoSafetyGate.py RunTwoPhase_GART_LMPC.py
```
