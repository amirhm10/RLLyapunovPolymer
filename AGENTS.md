# Repository Guide

## Scope
This repository is a notebook-driven research codebase for polymer CSTR control. The active workstreams are:

- baseline offset-free MPC
- TD3-based RL control
- standard Lyapunov tracking MPC
- plotting and export utilities

For Lyapunov work, the preferred implementation is now the consolidated `Lyapunov/` directory.

## Sibling Repository Boundary
This code repository is tightly coupled to the companion paper workspace, but the two folders must be treated as separate write targets:

- Code repo: `C:\Users\hamediaa\Desktop\Lyapunov_polymer`
- Paper workspace: `C:\Users\hamediaa\Desktop\Lyapunov Paper`

Use this repository for controller code, simulations, experiment runners, result exports, plotting utilities, and code-side reports. Use the paper workspace for manuscript prose, paper memory, literature packets, citation ledgers, figure/table registries, and paper workflow notes.

Before editing, infer the intended target from explicit paths, the active IDE file, and the artifact type. If a request is ambiguous or could touch both folders, state which repo will be edited before making changes, and ask if the write target is still unclear.

Do not write into the paper workspace from a code task unless Amir explicitly requests a paper update. Treat the paper workspace as read-only context when the requested work is code, controller, simulation, or experiment work.

## Current State
- This folder is a Git repository and is connected to a GitHub remote.
- `README.md` is still minimal.
- There is no pinned environment file such as `requirements.txt` or `environment.yml`.
- The default `python` in this environment does not have the scientific stack installed, so runtime validation is limited unless dependencies are installed first.

## Inferred Dependencies
Install these before running notebooks or scripts:

- `numpy`
- `scipy`
- `matplotlib`
- `pandas`
- `torch`
- `cvxpy`
- `control`
- `scikit-learn`
- `joblib`
- `jupyter`

Suggested bootstrap on Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install numpy scipy matplotlib pandas torch cvxpy control scikit-learn joblib jupyter
```

`cvxpy` is required for the standard Lyapunov target-selector path.

## Main Layout
- `Simulation/`
  Plant dynamics, baseline MPC helpers, rollout loops, and system identification.
- `TD3Agent/`
  TD3 agent, actor/critic models, replay buffers, and reward logic.
- `utils/`
  Shared helpers.
  Preferred Lyapunov shared helpers now live in `utils/lyapunov_utils.py`.
- `Lyapunov/`
  Canonical standard Lyapunov MPC implementation.
  - `target_selector.py`: refined target selector
  - `lyapunov_core.py`: terminal ingredients and tracking MPC solver
  - `run_lyap_mpc.py`: closed-loop standard Lyapunov MPC rollout
- `Plotting_fns/`
  MPC and RL plotting utilities.
- `Data/`
  Core runtime data assets and serialized model/scaling files.
- `results/`
  Saved experiment bundles and retained latest run exports.

## Preferred Lyapunov Files
When a request is about Lyapunov MPC, use these files first:

- `Lyapunov/target_selector.py`
- `Lyapunov/lyapunov_core.py`
- `Lyapunov/run_lyap_mpc.py`
- `utils/lyapunov_utils.py`

Legacy or secondary Lyapunov files still present at the top level are:

- `standard_lyap_tracking_mpc_v2.py`
- `safe_mpc_with_lyapunov_filter.py`
- `safe_mpc_with_lyapunov_filter_v2.py`

If a change only concerns the standard Lyapunov path, do not start from those files.

## Important Entrypoints
These are the main active experiment entrypoints in the repository root:

- `DirectLyapunovMPC.py`
- `OnlineTD3_OFMPCPretrained_SafetyGate.py`
- `OnlineTD3_OFMPCPretrained_NoSafetyGate.py`
- `OnlineTD3_ColdStart_SafetyGate.py`
- `OnlineTD3_ColdStart_NoSafetyGate.py`
- `DirectLyapunovMPC_DisturbanceRunner.py`
- `OffsetFreeMPC_DisturbanceRunner.py`

Legacy notebooks and compatibility runner wrappers are archived under `archive/` and should not be used as active entrypoints unless explicitly restored.

Notebook cells often import modules directly, so keep public function names stable unless the notebook code is updated too.

## Core Conventions
- The plant is a polymer CSTR with two manipulated inputs typically treated as `Qc` and `Qm`.
- Outputs are typically viscosity-like `eta` and reactor temperature `T`.
- Most control code uses scaled deviation coordinates, not raw plant units.
- `steady_states["ss_inputs"]` and `steady_states["y_ss"]` are the steady-state anchors.
- `xhatdhat` denotes the augmented observer state: physical state estimate plus disturbance estimate.
- Setpoint schedules are usually generated with `generate_setpoints_training_rl_gradually(...)`.
- For governed-reference/direct Lyapunov work, be precise about stability language. The current controller enforces model-based practical first-step Lyapunov contraction around a moving governed steady target `(x_s, u_s, y_s)`, not a global nonlinear asymptotic stability proof to the raw setpoint. Reports, papers, and slides should explicitly account for target movement or phrase the result as practical/recursive Lyapunov contraction.
- Keep controller objectives and RL rewards explicitly separated in all future runners and scripts:
  - MPC and Direct LMPC optimization objectives use the MPC penalties `Q = [5, 1]` and `R`/`Rdu = [1, 1]`.
  - Offline TD3 pretraining rewards use the one-step MPC quadratic stage cost with the same `Q = [5, 1]` and `R = [1, 1]`, because there is no closed-loop rollout reward shaping in the replay-label phase.
  - Online RL training/evaluation may use the shaped reward family with its own reward weights, currently `Q_reward = [12, 6]` and `R_reward = [1, 1]`, plus fallback/event/bonus terms. These reward-shaping parameters must not overwrite MPC, OF-MPC, LMPC, target-selector, or safety-gate objective weights.
  - Reports and configs should record both controller objective weights and RL reward weights when both are present.

Before editing controller logic, confirm whether each variable is:

- physical units
- min-max scaled to `[0, 1]`
- scaled to `[-1, 1]`
- deviation from steady state
- augmented with disturbance states

Most control bugs here come from mixing those representations.

## Where To Edit
- Plant dynamics: `Simulation/system_functions.py`
- Baseline MPC formulation: `Simulation/mpc.py`
- Baseline MPC rollout: `Simulation/mpc_run.py`
- TD3 logic: `TD3Agent/`
- Scaling and generic helpers: `utils/scaling_helpers.py`, `utils/helpers.py`
- Standard Lyapunov target selection: `Lyapunov/target_selector.py`
- Standard Lyapunov solver and terminal ingredients: `Lyapunov/lyapunov_core.py`
- Standard Lyapunov closed-loop rollout: `Lyapunov/run_lyap_mpc.py`
- Lyapunov shared helpers: `utils/lyapunov_utils.py`
- Plotting/export tooling: `lyap_debug_export.py`, `standard_lyap_debug_export.py`, `standard_lyap_plinter.py`, `target_selector_diagnostics.py`

## Modification Rules
- Prefer editing Python modules instead of notebooks unless the user explicitly requests notebook changes.
- For standard Lyapunov MPC, edit `Lyapunov/` and `utils/lyapunov_utils.py` first.
- Preserve public signatures that notebooks already import.
- Keep optional `cvxpy` imports guarded.
- Avoid renaming the Lyapunov rollout return tuple unless notebook callers and exporters are updated too.
- Do not rewrite files under `Data/` unless the user explicitly asks for regenerated assets.
- Prefer writing new experiment bundles under `results/`, not `Data/`.
- Ignore `__pycache__/`.

## Markdown Report Formatting
- Avoid wide pipe tables in Markdown reports. If a table grows beyond about 6-7 columns, split it into smaller themed tables such as performance, reliability, tracking, target diagnostics, and input ranges.
- Keep numeric columns right-aligned, use consistent precision within each table, and keep units or shorthand terms defined in the surrounding text.
- Put long text fields such as failure clusters, interpretation notes, or caveats in bullets below the table instead of forcing them into wide table cells.
- Prefer compact labels such as `S1 high` or `Residual mean` when the meaning is clear from the surrounding paragraph.
- Use rendered Markdown math for equations in `.md` reports. Use `$...$` for short inline symbols and `$$...$$` for displayed equations. Do not put equations in fenced `text` or code blocks unless the content is actual code or terminal output.
- For multi-line equations, use readable display math such as `aligned` blocks. Keep notation compact and define symbols in prose or bullets around the equation. Avoid very wide equations that force horizontal scrolling in GitHub or IDE previews.
- Keep code/config examples in fenced blocks, but keep mathematical method statements, optimization problems, observer equations, constraints, and reward definitions as rendered math.
- When a report uses figures, embed them inline in the Markdown report with relative image paths near the relevant discussion. Do not leave figures only as bare file links at the end.
- For MPC-only cases in safety-gate RL reports, do not plot fallback count as only zero unless the plot is explicitly labeled as actual fallback. Use the diagnostic Lyapunov contraction failure or `diagnostic_unsafe_count` as the MPC-only "would-be fallback if the gate were active" count. Keep actual fallback and would-be fallback clearly separated in labels, legends, and tables.
- For safety-gate RL comparisons, report both the actual training reward and `reward_no_penalty`. Use `reward_no_penalty` for cross-method control-performance comparisons because RL training reward includes fallback/event penalties that Direct Lyapunov MPC does not use.
- Before finishing a Markdown report, scan the rendered table shape in plain text: it should remain readable in an IDE preview and on GitHub without horizontal scrolling for the main conclusions.

## Commit And Change-Report Workflow
- For any major code, notebook, or controller update, create a Git commit at the end of the task unless the user explicitly says not to commit.
- After applying a requested code/config/report change, create a local Git commit for the intended files before the final response whenever it is safe to do so. Do not leave intended changes only unstaged or staged.
- If unrelated dirty worktree changes make a safe commit risky, stage only the intended files or hunks; if that is not possible, explain clearly in the final response why the commit was skipped.
- Use a descriptive commit message that matches the main technical change. Prefer messages like `Refine Step A selector tuning`, `Add RL paper-style debug export plots`, or `Fix safety-filter target backup logic`.
- For every major committed change, create or update a matching Markdown report under `change-reports/`.
- The relevant `change-reports/...md` file should be included in the same commit as the code change so the history stays paired.
- If a task naturally splits into distinct major updates, use separate commits and separate change reports rather than bundling unrelated work together.
- Before committing, run the low-cost validation that fits the change, typically `python -m py_compile` on touched modules.
- In the final response, report the commit hash and the matching change-report path.

## Validation Strategy
There is no formal test suite. Use low-cost validation:

- `python -m py_compile` on touched modules
- import-only checks when dependencies are installed
- small synthetic calls for target selector or solver helpers

Useful diagnostics already in the repo:

- `standard_lyap_exact_mpc_objective_test.py`
- `target_selector_diagnostics.py`
- `run_standard_lyap_export.py`
- `standard_lyap_plinter.py`

Avoid long notebook runs or retraining unless the user explicitly asks for them.

## Agent Workflow
- First decide whether the task belongs to baseline MPC, RL, or standard Lyapunov MPC.
- For standard Lyapunov work, read `Lyapunov/target_selector.py`, `Lyapunov/lyapunov_core.py`, and `Lyapunov/run_lyap_mpc.py` first.
- Use `utils/lyapunov_utils.py` for shared Lyapunov helpers instead of duplicating small helper functions.
- Use `LyapDetails.md` as the step-by-step functional reference for the current controller flow.

## Known Gaps
- No environment lockfile
- No automated regression suite
- Notebook-heavy orchestration
- Legacy and current Lyapunov files coexist

When unsure, preserve the standard Lyapunov path in `Lyapunov/` and keep compatibility wrappers thin.
