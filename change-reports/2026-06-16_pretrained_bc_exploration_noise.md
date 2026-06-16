# Increase Pretrained BC Exploration Noise

## Summary

The pretrained online TD3 behavior-cloning phase now uses a modest Gaussian
behavior-action exploration standard deviation instead of an almost
deterministic value.

## Change

- Updated `PRETRAINED_BC_EXPLORATION_STD` from `1.0e-4` to `0.002`.
- Cold-start BC exploration remains `0.005`.
- The clean teacher/demo action remains unchanged; this noise applies only to
  the behavior action executed around the teacher during BC rollout.

## Expected Effect

The replay buffer should get more local action coverage around the teacher
policy during pretrained BC, which should help critic learning without making
the pretrained actor fight a highly noisy teacher rollout.

## Validation

Completed checks:

```powershell
python -m py_compile utils/online_disturbance_runner.py Simulation/run_rl_lyapunov.py
& "C:\Users\hamed\miniconda3\envs\rlenv\python.exe" -c "<assert pretrained BC std is 0.002 and cold-start BC std remains 0.005>"
```

The config audit confirmed:

- pretrained BC: `bc_exploration_std=0.002`, `bc_behavior_noise="gaussian"`
- cold-start BC: `bc_exploration_std=0.005`, `bc_behavior_noise="gaussian"`
