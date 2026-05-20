# Prompt For ChatGPT Deep Repository Analysis

Use this prompt when sharing `report/rl_agent_authority_bc_latest_analysis_2026-05-19.html` and connecting the GitHub repository.

```text
I am working on a polymer CSTR control research repository:

GitHub repository:
https://github.com/amirhm10/RLLyapunovPolymer

I am also attaching this HTML report:
rl_agent_authority_bc_latest_analysis_2026-05-19.html

Please do a deep research-style analysis of the full repository and the attached report. I want you to reconstruct the methods, audit the implementation, analyze the findings, and propose technically serious next ideas, especially for the target selector problem that we have not solved yet.

Important: read the attached HTML report first, especially the final section called "Project History And Already-Tried Changes". Do not suggest things that are already listed there as tried or implemented unless you have a specific new variant and can explain why it is meaningfully different.

Project context:
- The system is a polymer CSTR.
- Outputs are viscosity-like eta and reactor temperature T.
- Manipulated inputs are usually Qc and Qm.
- Most control logic uses scaled deviation coordinates, while reports often discuss physical output units.
- The active root scripts are:
  - DirectLyapunovMPC.py
  - DirectLyapunovSafetyGateRL_ColdStart.py
  - DirectLyapunovSafetyGateRL_Pretrained.py
- The current active Lyapunov/RL study uses two setpoints, 200 episodes, setpoint length 400, rho_lyap = 0.99, and lyap_eps = 1e-3.
- The latest report says cold-start RL currently has better safety-gate authority, pretrained RL has better full-horizon reward/RMSE, and direct LMPC has good final-tail offset but worse full-horizon raw-setpoint RMSE and slower runtime.

Please inspect these files carefully:
- AGENTS.md
- report/rl_agent_authority_bc_latest_analysis_2026-05-19.md
- report/rl_agent_authority_bc_latest_analysis_2026-05-19.html, if available from the attachment
- DirectLyapunovMPC.py
- DirectLyapunovSafetyGateRL_ColdStart.py
- DirectLyapunovSafetyGateRL_Pretrained.py
- Lyapunov/target_selector.py
- Lyapunov/frozen_output_disturbance_target.py
- Lyapunov/direct_lyapunov_mpc.py
- Lyapunov/lyapunov_core.py
- Lyapunov/run_lyap_mpc.py
- Simulation/run_rl_lyapunov.py
- TD3Agent/reward_functions.py
- TD3Agent/agent.py
- utils/direct_lyapunov_study.py
- utils/lyapunov_utils.py
- change-reports/ directory, especially the recent 2026-05-18, 2026-05-19, and 2026-05-20 reports

Main goal:
Help me understand what is scientifically wrong or incomplete in the current target-selection and direct Lyapunov/RL safety-gate methodology, then propose better methods that are new relative to what we already tried.

Please structure your response like this:

1. Executive summary
- What is the current method trying to do?
- What appears to work?
- What appears to fail?
- What are the top 3 root-cause hypotheses?

2. Mathematical reconstruction
- Define the plant state, outputs, inputs, setpoints, scaled/deviation coordinates, observer/disturbance estimate, and steady target variables.
- Reconstruct the offset-free MPC baseline.
- Reconstruct the refined Step A target selector.
- Reconstruct the direct frozen-output-disturbance target selector.
- Reconstruct the direct Lyapunov MPC objective and contraction condition.
- Reconstruct the RL safety-gate flow, behavioral cloning setup, handoff, and reward.
- Be explicit about physical units versus scaled/deviation variables.

3. Implementation audit
- Check whether the code matches the intended mathematics.
- Look for sign errors, coordinate mismatches, wrong disturbance treatment, target/reference mismatch, wrong use of y_sp versus y_s, action scaling mistakes, replay-buffer mistakes, reward/logging mismatches, or diagnostics that might be misleading.
- Cite specific files, functions, and line numbers when possible.
- Separate confirmed bugs from plausible concerns.

4. Result interpretation
- Use the attached report figures/tables and repository result logic to explain why:
  - cold-start RL can have better safety-gate authority than pretrained RL,
  - pretrained RL can still have better full-horizon reward/RMSE,
  - direct LMPC can have good final-tail offset but worse full-horizon raw-setpoint RMSE,
  - MPC-only diagnostic should use would-be gate activation instead of actual fallback count.
- Explain which claims are strongly supported and which require another run.

5. Target selector deep dive
- Focus heavily on the target selector problem.
- Explain why the target can become "admissible" for Lyapunov contraction but still poor for raw setpoint tracking.
- Analyze the roles of:
  - frozen d_hat,
  - bounded target projection,
  - u_prev anchoring,
  - x_s smoothing,
  - raw y_sp tracking versus target-output y_s tracking,
  - target-quality bypass,
  - lexicographic bounded target option,
  - disturbance model mismatch.
- Identify whether the main issue is feasibility, objective alignment, disturbance modeling, target-center drift, or online controller aggressiveness.

6. New method proposals
- Propose 3 to 6 genuinely new target-selector or safety-gate designs that are not just repeats of what was already tried.
- For each proposal include:
  - mathematical formulation,
  - why it addresses the observed failure mode,
  - expected benefits,
  - risks or failure cases,
  - what code files would need to change,
  - a small validation experiment.
- Please prioritize ideas that preserve offset-free tracking and avoid steady-state target drift.

Possible directions are welcome only if you make them concrete and distinguish them from what is already tried:
- command governor or reference governor around the raw setpoint,
- target selector with explicit raw-setpoint offset penalty plus Lyapunov-center quality constraint,
- terminal/invariant-set-aware steady target selection,
- robust/tube target selector under disturbance-estimation uncertainty,
- two-layer target selector that first finds the closest trackable raw target and then selects the least-moving Lyapunov center,
- adaptive disturbance model or disturbance regularization,
- residual RL around a verified target-centered controller,
- target selector with a hard bound on y_s - y_sp or with a "reject poor target" policy.

7. Next experiment plan
- Give a staged plan with low-cost experiments first.
- Include exactly what metrics to compare:
  - raw-setpoint RMSE,
  - y_s-target RMSE,
  - final-tail offset,
  - target residual,
  - target movement,
  - input saturation,
  - Lyapunov contraction margin,
  - fallback/would-be activation rate,
  - wall-clock seconds per step.
- Include a minimal ablation matrix that does not explode into too many runs.

8. Concrete implementation sketch
- Give pseudocode or patch-level guidance for the best target-selector idea.
- Identify the exact config knobs and default values to try first.
- Explain how to log diagnostics so we can prove whether the new idea worked.

9. Things not to suggest again
- Explicitly list recommendations you are not making because the attached report says they were already tried.
- If you think one of those old ideas should be revisited, explain the new evidence or new variant that makes it different.

Formatting requirements:
- Use clear Markdown.
- Use equations in LaTeX.
- Avoid very wide tables.
- Avoid emojis.
- Be precise and skeptical. Do not overclaim.
- If you cannot inspect a file or result, say so and explain what evidence is missing.
```
