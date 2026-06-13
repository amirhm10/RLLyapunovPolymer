# Deep Research Prompt: Better Direct LMPC Target Selector

I am working on safe RL/MPC for a polymer CSTR. The current Direct LMPC TD3
pretraining workflow is not giving a good pretrained actor, even after
increasing the actor and critic to `[512, 512, 512, 512, 512]`.

Please use the attached bundle to help design a better target selector.

## Evidence To Use

- `README.md`: high-level diagnosis.
- `tables/pretrain.csv`: pretraining configurations and losses.
- `tables/comparison.csv`: closed-loop comparison metrics.
- `tables/gap.csv`: TD3-vs-expert gaps.
- `tables/target.csv`: Direct LMPC target-selector diagnostics.
- `tables/failures.csv`: LMPC label rejection reasons.
- `figures/*.png`: visual summaries.

## Important Facts

- OF-MPC TD3 pretraining works well under the same plant, scaler, TD3 state/action dimensions, and comparison setpoints.
- Direct LMPC and OF-MPC baselines track almost identically.
- LMPC TD3 imitation is poor for both governed-reference and bounded-mixed selectors.
- Increasing the network to 512x5 lowered supervised BC loss but worsened closed-loop comparison.
- The bounded-mixed selector often uses a bounded least-squares target instead of the exact raw-setpoint steady target.
- The governed-reference selector also produced poor LMPC-TD3 imitation.
- Direct LMPC uses the selected target for Lyapunov certification but still tracks the raw setpoint in the MPC objective.

## Research Questions

1. What target-selector formulations are common in offset-free tracking MPC,
   reference governors, command governors, and Lyapunov MPC when the raw
   setpoint may be unreachable?
2. How can a target selector preserve practical Lyapunov contraction while
   producing a smoother expert action map for offline RL imitation?
3. Should the selector be lexicographic: first minimize raw setpoint mismatch,
   then use input/state regularization only as a tie-breaker?
4. Would a multi-step reachable reference or dynamic reference governor be more
   suitable than a steady target selector?
5. How should rejected/infeasible LMPC label regions be represented during
   offline pretraining?
6. What label-quality metrics should be logged and filtered before actor BC?
7. What concrete ablation plan should be run next?

Please produce a literature-backed target-selector redesign plan with equations,
implementation-level details, and a small ablation matrix.
