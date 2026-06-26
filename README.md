# Stability-Gated TD3 for Polymer CSTR Control

This repository contains the simulation and controller code for the manuscript
**"Stability-Guaranteed RL Control from Step-Test Data"**. The study evaluates
a safety-gated Twin Delayed Deep Deterministic Policy Gradient (TD3) controller
for a nonlinear polymerization continuous stirred-tank reactor (CSTR).

The main idea is to let TD3 propose a fast candidate input while a
Guarded Admissible Reachable Target Lyapunov MPC (GART-LMPC) layer decides what
input is actually applied to the plant. The safety layer uses a
step-test-identified scaled-deviation state-space model, an offset-free
observer, a GART target selector, and a one-step Lyapunov contraction check. If
the TD3 proposal is not certified, the applied input is replaced by a
GART-LMPC move tied to the requested setpoint.

## Method Summary

Most calculations are performed in scaled-deviation coordinates. Let
$\bar{x}_k = [\Delta x_k^T, d_k^T]^T$ denote the augmented observer state,
where $d_k$ is the output-disturbance estimate, and let $\Delta u_k$ be the
input deviation from the nominal steady input. The identified model used by the
offset-free MPC and GART-LMPC layers has the form

```math
\bar{x}_{k+1}=A_a\bar{x}_k+B_a\Delta u_k,\qquad
\Delta y_k=C_a\bar{x}_k .
```

At each sampling instant, the requested setpoint $y_{sp,k}$ remains the
tracking objective, but the Lyapunov certificate is centered on an accepted
GART equilibrium target $(x_{s,k}, u_{s,k}, y_{s,k})$. For a certified
disturbance value $d^c_k$, the target selector searches for a reachable target
that is close to the requested setpoint while satisfying steady-state,
input-bound, target-motion, and contraction-probe checks:

```math
x_{s,k}=Ax_{s,k}+Bu_{s,k},\qquad
y_{s,k}=Cx_{s,k}+d^c_k .
```

If the requested setpoint is not immediately certifiable, the command governor
searches between the previous accepted command and the requested setpoint. This
keeps the safety layer tied to the tracking objective without claiming that the
raw setpoint is always reachable or certifiable.

The safety gate uses a target-centered Lyapunov function

```math
V_k=(\hat{x}_k-x_{s,k})^T P(\hat{x}_k-x_{s,k})
```

and accepts a TD3 candidate input $\tilde{u}_k$ only if the model-predicted next
state satisfies the one-step decrease condition

```math
V(\hat{x}_{k+1|k}(\tilde{u}_k)-x_{s,k})
\le \rho V(\hat{x}_k-x_{s,k})+\epsilon .
```

When this test fails, the applied input is replaced by the first move from the
GART-LMPC fallback problem when that move is certified. In compact form,

```math
u_k =
\begin{cases}
\tilde{u}_k, & \text{if the TD3 proposal is certified},\\
u^{\mathrm{GART}}_{0|k}, & \text{if fallback GART-LMPC is certified},\\
\operatorname{clip}(u_{k-1}), & \text{otherwise}.
\end{cases}
```

TD3 is therefore used as a candidate-action generator, not as the final
authority on the plant input. Offline pretraining initializes the actor with
offset-free MPC first-move labels. During online operation, reward evaluation
and replay storage use the executed plant transition and the applied input, so
the learning signal reflects the controller that the plant actually saw.

## Repository Scope

The public repository is focused on code needed to inspect or reproduce the
controllers used in the paper. It intentionally excludes manuscript drafts,
slide decks, local reports, archived notebooks, raw result bundles, and
workflow-specific agent files.

Included:

- polymer CSTR simulation utilities
- offset-free MPC and GART-LMPC controller code
- TD3 agent, replay buffer, and reward logic
- active experiment runners for the paper studies
- minimal data/scaling assets used by the case study

Not included:

- manuscript source and paper workflow files
- generated result folders and trained-agent bundles
- local change reports, analysis reports, and slide materials
- archived notebooks and exploratory scripts

## Main Components

```text
Simulation/       Plant model, MPC rollout code, observer and closed-loop logic
TD3Agent/         TD3 actor/critic networks, replay buffer, reward utilities
Lyapunov/         Lyapunov target selector and tracking MPC ingredients
utils/            Shared scaling, profile, GART, and runner utilities
experiments/      Helper routines for GART target-selector studies
Plotting_fns/     Plotting utilities used by retained runners
Data/             Small retained case-study data/scaling assets
```

## Key Entrypoints

Scheduled operating-change study:

```powershell
python .\RunTwoPhase_OFMPCPretrained_SafetyGate.py
python .\RunTwoPhase_OFMPCPretrained_NoSafetyGate.py
python .\RunTwoPhase_ColdStart_SafetyGate.py
python .\RunTwoPhase_ColdStart_NoSafetyGate.py
python .\RunTwoPhase_GART_LMPC.py
```

New setpoint-cycle safety study:

```powershell
python .\RunCyclePhase1Disturbance_SavedAgentSafetyGate.py
python .\RunCyclePhase1Disturbance_SavedAgentNoSafetyGate.py
python .\RunCyclePhase1Disturbance_GARTLMPC.py
```

The saved-agent cycle runners expect trained TD3 checkpoints. The repository
contains the runner logic, but generated trained-agent files are not included in
the public tree. To rerun those scripts, first generate compatible agents with
the online TD3 runners or edit `RunCyclePhase1Disturbance_Common.py` to point to
local checkpoint paths.

## Installation

No pinned environment file is currently provided. A typical Python environment
needs:

```powershell
pip install numpy scipy matplotlib pandas torch cvxpy control scikit-learn joblib
```

Optional but useful:

```powershell
pip install jupyter
```

The local development runs used a Conda environment with the scientific Python
stack installed. `cvxpy` is required for the Lyapunov target-selector path.

## Quick Validation

Before running long experiments, a low-cost syntax check is:

```powershell
python -m py_compile RunOnlineTD3TwoPhaseStudy.py RunCyclePhase1Disturbance_Common.py
python -m py_compile Simulation\run_rl_lyapunov.py utils\two_phase_profiles.py
```

Long TD3 studies can take substantial time. Generated outputs are written under
`results/`, which is ignored by Git.

## Notes on Coordinates and Objectives

Most controller calculations use scaled deviation coordinates, not raw plant
units. Outputs are viscosity-like `eta` and reactor temperature `T`, and the
manipulated inputs are typically `Qc` and `Qm`.

The controller objectives and RL rewards are intentionally separated:

- MPC, OF-MPC, and GART-LMPC use MPC tracking and move-suppression penalties.
- Offline TD3 pretraining uses expert labels from offset-free MPC.
- Online TD3 uses a shaped tracking reward and stores the executed action, not
  the rejected candidate action, in replay.

This distinction matters when comparing tracking performance, reward, safety
gate intervention rate, and no-gate unsafe-action diagnostics.

## Citation

If you use this repository, please cite the associated manuscript when it
becomes available. The current manuscript title is:

```text
Stability-Guaranteed RL Control from Step-Test Data
```

## License

No open-source license has been added yet. Until a license is provided, all
rights are reserved by the repository owner.
