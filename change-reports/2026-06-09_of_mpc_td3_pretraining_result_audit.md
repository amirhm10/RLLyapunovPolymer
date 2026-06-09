# OF-MPC TD3 Pretraining Result Audit

Date: 2026-06-09

## Summary

Analyzed the latest OF-MPC-pretrained TD3 checkpoint and its comparison against OF-MPC, then extended the pretraining process report with the results and critic-loss interpretation.

## Reviewed Artifacts

- Pretraining bundle: `results/PretrainOFMPC/20260609_002245/`
- Checkpoint: `results/PretrainOFMPC/20260609_002245/of_mpc_pretrained_td3_20260609_134202.pkl`
- Comparison bundle: `results/PretrainOFMPCComparison/20260609_155747/`
- Metrics: `results/PretrainOFMPCComparison/20260609_155747/comparison_metrics.csv`

## Key Findings

- The new `[256, 256, 256]` TD3 checkpoint closely matches OF-MPC on the direct two-setpoint comparison.
- Nominal mean RMSE is `0.3562` for TD3 versus `0.3554` for OF-MPC.
- Disturbance mean RMSE is `0.3594` for TD3 versus `0.3569` for OF-MPC.
- The earlier smaller checkpoint `results/PretrainOFMPC/20260608_171525/` remains much worse on the same corrected-scaler comparison.
- The current `loss_arrays.json` is empty, so the completed run does not contain a recoverable critic-loss curve.

## Critic-Loss Diagnosis

The critic-loss concern is valid, but the current artifact cannot answer it directly because `TD3Agent.pretrain_from_buffer(...)` printed epoch losses without appending them to `agent.critic_losses`.

As an indirect check, Q-values on saved comparison rollout states were around `-1.22e5`, consistent with the broad replay reward mean of about `-604` and `gamma = 0.995`. Mean absolute rollout TD residuals were around `570-663`, which is small relative to the Q-value scale and does not suggest obvious critic divergence.

## Code Change

Updated `TD3Agent/agent.py` so future `pretrain_from_buffer(...)` calls append epoch-averaged losses:

- actor behavioral-cloning loss to `actor_losses` and `actor_bc_losses`
- critic TD loss to `critic_losses`

Future pretraining runs should therefore produce non-empty `loss_arrays.json` and `loss_arrays.csv`.

## Documentation

Extended `report/of_mpc_td3_pretraining_process_2026-06-08.md` with:

- current checkpoint configuration
- nominal and disturbance comparison metrics
- physical trajectory/input closeness to OF-MPC
- critic-loss logging limitation
- proxy critic TD diagnostic interpretation

## Validation

Completed validation:

```powershell
python -m py_compile TD3Agent/agent.py utils/of_mpc_td3_workflow.py utils/td3_helpers.py
```
