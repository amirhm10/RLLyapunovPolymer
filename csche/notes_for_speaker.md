# Speaker Notes For CSChE Draft

## Talk Length

The main deck is designed for a 12 to 15 minute conference talk. The target pace is roughly one minute per main technical slide, with backup slides available for questions.

## Core Story

The talk should not be framed as "RL beats MPC." The stronger and more honest story is:

1. MPC is the practical baseline for constrained process control.
2. RL can propose actions and adapt online, but it needs a deployment safety layer.
3. The Lyapunov safety gate gives model-based final authority.
4. Pretraining improves full-horizon tracking and reward.
5. Cold-start RL currently has better safety-gate authority.
6. The unresolved issue is target selection, not just reward tuning or fallback penalty size.

## Slide-By-Slide Intent

1. **Title**  
   Introduce the method as a practical RL/MPC integration, not a replacement of MPC.

2. **Technical objective**  
   Start with the closed-loop optimization problem and the certification inequality. This should feel like a technical CSChE talk rather than a broad motivation slide.

3. **Polymer CSTR problem**  
   Make the variables clear: outputs are eta and T, inputs are Qc and Qm. Emphasize physical units versus scaled deviation coordinates.

4. **Offset-free control structure**  
   Explain that the observer and disturbance estimate feed the target selector. This is the bridge between offset-free control and Lyapunov MPC.

5. **Direct target selection**  
   Stress that the target package is `x_s, d_s, u_s, y_s`. The disturbance target is frozen as `d_s = d_hat`. This is useful but can create alignment problems.

6. **Direct LMPC formulation**  
   Present the first-step contraction inequality. The key line is that the Lyapunov certificate is computed around the selected target.

7. **RL safety-gate architecture**  
   Walk through the actor, gate, fallback, replay, and demo buffer. Mention that BC is agent-authority: the actor proposes the candidate even during BC.

8. **Cold-start versus pretrained RL**  
   Explain that pretraining improves early behavior, while cold-start explores more and can learn a different gate-compatible action style.

9. **Gate authority result**  
   Use the phase-authority figure. Point out that cold-start BC is fragile, the final evaluation is stable for both agents, and the full-run authority/tracking tradeoff is still visible.

10. **Tracking and runtime result**  
   Pretrained RL has the best learned full-horizon RMSE. Direct LMPC is slower and worse over the full horizon.

11. **Tail offset result**  
   Use the custom final-evaluation tracking figure. Separate output tracking over the final episode from the final 100-step offset bars.

12. **Target-selector limitation**  
   This is the most important slide. Explain that an admissible Lyapunov target can still be misaligned with the raw process objective.

13. **Scientific interpretation**  
   Be balanced. What works: safety gate, pretraining, diagnostics. What remains unresolved: target quality and reward-authority mismatch.

14. **Next experiments**  
   Keep the plan small and target-focused. Avoid broad reward sweeps.

15. **Closing**  
   Repeat the one-sentence message: the main challenge is choosing a Lyapunov target that is both admissible and aligned with the raw process objective.

## Phrases To Use

- "MPC remains the deployment reference point."
- "The actor has proposal authority, but not final authority."
- "MPC-only fallback should be read as would-be gate activation, not actual fallback."
- "The target can be Lyapunov-admissible but process-objective poor."
- "The next experiment should change target selection in a measurable way."

## Phrases To Avoid

- "RL solves the control problem."
- "The method is proven safe."
- "Direct LMPC is bad."  
  Better: "Direct LMPC settles well in the final tail, but has poor full-horizon raw-setpoint RMSE in the analyzed run."

- "Increasing fallback penalty is the next fix."  
  That was already tried up to `fallback_event_penalty = 10.0`.

## Most Important Manual Review

Review Slide 12 first. It carries the central scientific limitation. If that slide is unclear, the talk will sound like a generic safe-RL story rather than the actual current research bottleneck.

Second review priority: Slides 9 through 11. These now use the custom CSChE result figures and should be checked for projection readability.
