# Repository Guide

## Scope
This repository is a notebook-driven research codebase for polymer CSTR control. The active workstreams are:

- baseline offset-free MPC
- TD3-based RL control
- standard Lyapunov tracking MPC
- plotting and export utilities

For Lyapunov work, the preferred implementation is now the consolidated `Lyapunov/` directory.

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
  Runtime data assets and serialized model/scaling files.

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

## Important Notebooks
These appear to be the main experiment entrypoints:

- `LyapMPC.ipynb`
- `StandardLyapMPC.ipynb`
- `MPCOffsetFree.ipynb`
- `MPCOffsetFree1.ipynb`
- `OnlineTrainingWPreTrain.ipynb`
- `OnlineTrainingWPreTrainLyapunov.ipynb`
- `ComparePlots.ipynb`

Notebook cells often import modules directly, so keep public function names stable unless the notebook code is updated too.

## Core Conventions
- The plant is a polymer CSTR with two manipulated inputs typically treated as `Qc` and `Qm`.
- Outputs are typically viscosity-like `eta` and reactor temperature `T`.
- Most control code uses scaled deviation coordinates, not raw plant units.
- `steady_states["ss_inputs"]` and `steady_states["y_ss"]` are the steady-state anchors.
- `xhatdhat` denotes the augmented observer state: physical state estimate plus disturbance estimate.
- Setpoint schedules are usually generated with `generate_setpoints_training_rl_gradually(...)`.

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
- Ignore `__pycache__/`.

## Commit And Change-Report Workflow
- For any major code, notebook, or controller update, create a Git commit at the end of the task unless the user explicitly says not to commit.
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
