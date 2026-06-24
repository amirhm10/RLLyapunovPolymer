# Cycle Phase-1-Disturbance Study Runners

## Objective

Add a separate set of two-phase study entrypoints for the new constrained
tracking scenario:

- raw setpoint cycle `(4.0, 321.5) -> (3.3, 324.5)`
- 400 samples per setpoint
- 100 online-training cycles
- no teacher warm-up
- no behavior-cloning/critic-only teacher phase
- no handoff phase
- disturbance profile held to the Phase-1 disturbance family, with Phase 2
  staying at the Phase-1 final disturbance value

This is intended as a third discussion scenario focused on online adaptation
and safe action selection near an input-constrained reference cycle.

## Files Added

- `RunCyclePhase1Disturbance_Common.py`
- `RunCyclePhase1Disturbance_ColdStartStudy.py`
- `RunCyclePhase1Disturbance_SavedAgentGART.py`

## Files Updated

- `RunOnlineTD3TwoPhaseStudy.py`
- `utils/online_disturbance_runner.py`

## Study Entrypoints

### Cold-start safety/no-safety pair

Run:

```powershell
C:\Users\HAMEDI\miniconda3\envs\rl\python.exe .\RunCyclePhase1Disturbance_ColdStartStudy.py
```

Methods:

- `cold_start_safety_gate`
- `cold_start_no_safety_gate`

The two methods share the same setpoint and disturbance profiles. Neither
loads an OF-MPC/LMPC pretrained checkpoint.

### Saved-agent GART safety-gate continuation

Run:

```powershell
C:\Users\HAMEDI\miniconda3\envs\rl\python.exe .\RunCyclePhase1Disturbance_SavedAgentGART.py
```

Method:

- `saved_agent_safety_gate`

Default saved-agent checkpoint:

```text
results/OnlineTD3_TwoPhaseStudy/20260623_092655_cold_start_safety_gate/seed_009/cold_start_safety_gate/onlinetd3_coldstart_safetygate/trained_agent_20260624_093558.pkl
```

The checkpoint is a previously online-trained TD3 agent from a seed folder.
It is not treated as an OF-MPC pretrained checkpoint, and its critic is not
reset by default.

## Scenario Definition

The shared scenario constants live in `RunCyclePhase1Disturbance_Common.py`:

```python
SETPOINT_CYCLE_Y_PHYS = (
    (4.0, 321.5),
    (3.3, 324.5),
)

PHASE1_EPISODES = 100
PHASE2_EPISODES = 1
PHASE1_SETPOINT_HOLD_STEPS = 400
REPORTING_WINDOW_STEPS = 800
```

This gives:

- Phase-1 online training: `100 * 800 = 80000` samples
- Phase-2 continuation/evaluation: `1 * 800 = 800` samples
- Total profile length: `80800` samples

The disturbance multipliers are:

```python
PHASE1_QI_MULTIPLIER = 0.95
PHASE1_QS_MULTIPLIER = 1.05
PHASE1_HA_MULTIPLIER = 0.92

PHASE2_QI_MULTIPLIER = PHASE1_QI_MULTIPLIER
PHASE2_QS_MULTIPLIER = PHASE1_QS_MULTIPLIER
PHASE2_HA_MULTIPLIER = PHASE1_HA_MULTIPLIER
```

Thus Phase 2 remains at:

- `qi = 102.6`
- `qs = 481.95`
- `ha = 966000.0`

## No-Teacher Online Training

The new wrappers intentionally set:

- `warmup_buffer_only_episodes = 0`
- `behavior_clone_teacher_episodes = 0`
- `handoff_episodes = 0`

This means the online TD3 policy is trained directly under the selected
safety/no-safety mode, instead of starting with teacher-generated replay or a
teacher-to-RL transition period.

## Saved-Agent Method Support

Added two method keys to `utils.online_disturbance_runner.ONLINE_TD3_PRESETS`:

- `saved_agent_safety_gate`
- `saved_agent_no_safety_gate`

These load a TD3 checkpoint supplied by `agent_path` or by the
`SAVED_ONLINE_TD3_AGENT_PATH` environment variable. They use the GART target
selector family but do not imply an OF-MPC/LMPC pretrained checkpoint.

`RunOnlineTD3TwoPhaseStudy.py` now recognizes the saved-agent methods and
passes the checkpoint path through to the online TD3 runner.

## Validation

Compiled:

```powershell
C:\Users\HAMEDI\miniconda3\envs\rl\python.exe -X pycache_prefix="$env:TEMP\codex_pycache" -m py_compile RunOnlineTD3TwoPhaseStudy.py utils\online_disturbance_runner.py RunCyclePhase1Disturbance_Common.py RunCyclePhase1Disturbance_ColdStartStudy.py RunCyclePhase1Disturbance_SavedAgentGART.py
```

Profile-only validation was run without launching training. It confirmed:

- setpoints: `((4.0, 321.5), (3.3, 324.5))`
- total steps: `80800`
- Phase-1 steps: `80000`
- Phase-2 steps: `800`
- Phase-1 final disturbance equals final disturbance
- default saved-agent checkpoint exists

The profile check emitted the existing observer pole-placement convergence
warning from `Simulation/mpc.py`, but it completed successfully.

## Notes

These wrappers do not launch automatically. They are entrypoints for the next
experiment stage. The output root is `results/`, so new runs will be stored
under:

```text
results/OnlineTD3_TwoPhaseStudy/
```
