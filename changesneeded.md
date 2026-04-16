# Changes Needed Audit

## Scope
This note covers three things:

1. a static audit of the code path behind `pretraining_rl_controller.ipynb` and the main downstream RL notebooks/runners,
2. a project-wide scaling consistency check, limited to source files and notebook contents,
3. a literature check on whether the current offline MPC-to-TD3 pretraining design is already strong enough or should be improved.

This is a **static audit**, not a runtime regression. In this environment, the default `python` does not have `numpy`, so I could not directly load and compare the serialized scaler pickle files under `Data/`. That means:

- code-path consistency findings below are high-confidence,
- notebook constant comparisons are high-confidence,
- direct verification of `Data/min_max_states.pickle` and `Data/scaling_factor.pickle` against the hard-coded notebook overrides is **not** runtime-verified here.

## Bottom Line
The current pretraining block is **good enough as a baseline**, but it is **not yet strong enough as the final training story**.

The strongest local problems are not in the basic idea. The main issues are:

- the pretraining notebook is **not connected cleanly** to the downstream evaluation/training notebooks,
- there are a few **real code inconsistencies** in scaling/state handling,
- the offline dataset is generated from **independent random samples**, not from teacher trajectory distributions,
- there is **no holdout or closed-loop validation** in the pretraining notebook itself.

Research-wise, the literature supports the general approach of imitating MPC, but it also strongly supports the view that **plain behavior cloning on offline expert data is usually not enough by itself** because of distribution shift and compounding error.

## 1. What Is Correct and Consistent Already

### 1.1 Pretraining notebook intent is clear
`pretraining_rl_controller.ipynb` is not online RL training. It is an **offline MPC-to-TD3 pretraining notebook**:

- build the augmented linear model and scaling metadata,
- generate synthetic samples,
- label them with the first MPC move,
- behavior-clone the actor,
- then pretrain the critic offline on the same dataset.

That interpretation is consistent with:

- `utils/td3_helpers.py`
- `TD3Agent/agent.py`
- `TD3Agent/replay_buffer.py`
- `Simulation/mpc.py`

### 1.2 Active RL state layout is mostly consistent
For the active RL path, the state layout is consistent:

\[
s_k = [\,\mathrm{scaled}(x_{aug,k}),\ \mathrm{scaled}(y_{sp,k}),\ \mathrm{scaled}(u_{k-1})\,]
\]

The active Python runners use `utils/scaling_helpers.apply_rl_scaled(...)`, and the pretraining notebook uses the same layout when it builds buffer states and next-states.

### 1.3 Legacy augmentation is still the default in the pretraining path
`pretraining_rl_controller.ipynb` calls `load_and_prepare_system_data(...)` without overriding augmentation style, so it uses the default `augmentation_style="legacy"` path in `utils/td3_helpers.py`.

That is still consistent with the plain RL notebooks and with the current safe-RL notebook when it is configured for the legacy RL backend.

### 1.4 The hard-coded RL state min/max arrays are internally consistent across notebooks
The same hard-coded `x_min` / `x_max` override arrays appear repeatedly across the main RL/MPC/Lyapunov notebooks, including:

- `pretraining_rl_controller.ipynb`
- `PretrainAgentPerformance.ipynb`
- `OnlineTrainingWPreTrain.ipynb`
- `OnlineTrainingWPreTrain+BC.ipynb`
- `OnlineTrainingWPreTrainLyapunov.ipynb`
- `LyapunovSafetyFilterRL.ipynb`
- `MPCOffsetFree.ipynb`
- `MPCOffsetFree1.ipynb`
- `LyapMPC.ipynb`
- `StandardLyapMPC.ipynb`

So the project is not randomly using different hard-coded RL state bounds in different notebooks. It is using the **same override** repeatedly.

### 1.5 Training and evaluation setpoint ranges are compatible
The pretraining notebook builds data over:

\[
y_{sp} \in \{[2.8, 320],\ [5.0, 326]\}
\]

while the downstream RL evaluation/training notebooks typically use scenarios like:

\[
[4.5, 324],\ [3.4, 321]
\]

These lie inside the pretraining range, so there is no obvious out-of-range setpoint mismatch.

## 2. Confirmed Inconsistencies or Weak Spots

### 2.1 The pretraining notebook is disconnected from the downstream RL workflow
This is the biggest local inconsistency.

`pretraining_rl_controller.ipynb` trains and saves a checkpoint using:

- actor/critic hidden sizes: `[256, 256, 256]`
- `mode="mpc"`
- save name pattern: `td3_YYYYMMDD_HHMMSS.pkl`

But the downstream notebooks currently instantiate and load a different family of agent:

- actor/critic hidden sizes: `[512, 512, 512, 512, 512]`
- fixed checkpoint name/path: `agent_2507171027.pkl`

This means the pretraining notebook is **not structurally feeding the active downstream path**.

So even if the pretraining notebook works, it is not currently the source of the model used in:

- `PretrainAgentPerformance.ipynb`
- `OnlineTrainingWPreTrain.ipynb`
- `OnlineTrainingWPreTrainLyapunov.ipynb`
- `LyapunovSafetyFilterRL.ipynb`

This is not just a naming issue. It is an architecture-level disconnect.

### 2.2 There are two different `apply_rl_scaled(...)` implementations
There are two copies:

- `utils/scaling_helpers.py`
- `utils/helpers.py`

They are not exactly the same.

`utils/scaling_helpers.py` uses the safer min-max helpers and guards zero-width ranges via `eps`.

`utils/helpers.py` uses the raw formula directly:

\[
2 \cdot \frac{x - x_{\min}}{x_{\max}-x_{\min}} - 1
\]

with no zero-width protection.

The active runners import from `utils/scaling_helpers.py`, so this is not currently breaking the main RL path. But it is still a **real project inconsistency** and an unnecessary future bug source.

### 2.3 `Simulation/run_rl.py` and `Simulation/run_rl_lyapunov.py` do not define the TD3 next-state identically
This is a real semantic mismatch.

In `Simulation/run_rl.py`, the pushed next state uses the **current** setpoint:

\[
s_{k+1}^{RL} = [x_{k+1},\ y_{sp,k},\ u_k]
\]

But in `Simulation/run_rl_lyapunov.py`, the pushed next state uses the **next** setpoint:

\[
s_{k+1}^{RL} = [x_{k+1},\ y_{sp,k+1},\ u_k]
\]

This means the plain RL runner and the safe RL runner are training on **different transition definitions**, even if everything else matches.

That is exactly the kind of subtle mismatch that can make a pretrained policy look worse once moved into a different runner.

### 2.4 `TD3Agent.load(...)` does not restore hyperparameters or optimizer state
`TD3Agent.load(...)` restores:

- actor weights
- critic weights

and then hard-copies those into targets.

It does **not** restore:

- architecture metadata,
- optimizer state,
- saved hyperparameters,
- exploration schedule state,
- counters such as `steps`, `train_steps`, `total_it`.

This is acceptable only if the caller reconstructs the exact same architecture and hyperparameters before calling `load(...)`.

That is fragile, especially because the project currently has multiple different TD3 configurations across notebooks.

### 2.5 `Simulation/sys_ids.py` has a broken scaling import
`Simulation/sys_ids.py` imports:

```python
from utils.td3_helpers import apply_min_max, reverse_min_max
```

But `utils/td3_helpers.py` imports `apply_min_max` from `utils.scaling_helpers.py` and does **not** expose `reverse_min_max`.

So this is a real project inconsistency and likely a broken import path if that file is executed.

### 2.6 Old absolute OneDrive paths still exist in several notebooks
Several notebooks still reference absolute paths such as:

`C:\Users\HAMEDI\OneDrive - McMaster University\PythonProjects\Polymer_example\...`

This is not a control-theory bug, but it is a reproducibility and portability problem. It also makes it harder to know which checkpoint or exported result is actually the authoritative one.

### 2.7 The pretraining notebook has no holdout validation or closed-loop validation
`print_accuracy(...)` only measures in-sample actor agreement against stored MPC actions.

That means the notebook ends with:

- no train/validation split,
- no held-out action prediction metric,
- no closed-loop test after pretraining,
- no comparison of teacher MPC vs cloned policy on trajectories.

This is a major methodological gap.

### 2.8 The critic pretraining dataset has effectively no terminal structure
The replay buffer used for pretraining stores `done = 0` for all samples.

That is not automatically wrong for this control setting, but it means the TD-pretrained critic never sees terminal structure or block boundaries. So the critic objective is effectively an infinite-horizon bootstrap objective over a synthetic one-step teacher dataset.

This is a design choice, but it should be treated as a deliberate one, not as a default truth.

## 3. Scaling Audit

## 3.1 What is consistent
The active RL scaling pipeline is mostly consistent:

- raw inputs/outputs are scaled to `[0,1]` using `data_min` / `data_max`,
- RL state blocks are then mapped to `[-1,1]`,
- the state ordering is consistent across the pretraining notebook and active RL runners.

For the active RL path, the main scaling source is:

- `utils/td3_helpers.load_and_prepare_system_data(...)`

and the main RL-state scaling implementation is:

- `utils/scaling_helpers.apply_rl_scaled(...)`

## 3.2 What is duplicated
The most important duplication is:

- file-based state bounds from `Data/min_max_states.pickle`,
- then notebook-level hard-coded overrides of `min_max_dict["x_min"]` / `["x_max"]`.

Because the hard-coded arrays are repeated consistently, this is not currently random drift. But it is still a maintenance risk:

- if the pickle file changes and the notebooks do not,
- or if one notebook gets edited and another does not,

the project can silently fork into incompatible scaler definitions.

## 3.3 What could not be verified here
Because the current environment lacks `numpy`, I could not load and compare:

- `Data/min_max_states.pickle`
- `Data/scaling_factor.pickle`

against the hard-coded notebook arrays.

So the exact statement I can support is:

- the **source-code copies** of the RL state bounds are consistent across the main notebooks,
- but I cannot certify from runtime loading here that those bounds still match the serialized project assets under `Data/`.

## 4. Research Check: Is This Pretraining Part Strong Enough?

## 4.1 Short answer
It is **strong enough as a reasonable baseline**, but **not strong enough to stop here**.

The literature supports:

- imitating MPC as a valid controller-learning route,
- offline expert pretraining as a useful starting point,

but it also strongly warns that:

- plain behavior cloning suffers from distribution shift,
- offline data alone usually underestimates rollout error,
- trajectory-distributed expert data is better than purely i.i.d. sampled states,
- some form of on-policy data aggregation or closed-loop correction is often needed.

## 4.2 What the papers support

### A. Behavior cloning alone is vulnerable to covariate shift
Ross, Gordon, and Bagnell (2011) show that imitation learning breaks standard i.i.d. assumptions because future observations depend on the learner’s own previous actions, and they motivate DAgger-style data aggregation for this reason.

Source:
- Ross et al., 2011, PMLR 15
  https://proceedings.mlr.press/v15/ross11a.html

### B. This point applies directly to MPC imitation
Ahn et al. (2023) study imitation learning for MPC explicitly and state that behavior cloning can mimic MPC, but it is data-inefficient and suffers from distribution shift; they show that an on-policy imitation method is superior to plain behavior cloning for MPC.

Source:
- Ahn et al., 2023, PMLR 211
  https://proceedings.mlr.press/v211/ahn23a.html

### C. Recent theory says the execution gap can be fundamental in continuous control
Simchowitz, Pfrommer, and Jadbabaie (2025) show that in continuous control, smooth deterministic imitators can suffer rollout error much larger than their training error unless the policy/data structure is richer or the expert distribution is sufficiently spread.

Source:
- Simchowitz et al., 2025, PMLR 291
  https://proceedings.mlr.press/v291/simchowitz25a.html

### D. Offline RL on fixed datasets has known failure modes
Offline RL reviews emphasize distribution shift, bootstrapping error, and out-of-distribution action evaluation as the major challenges when learning purely from fixed data.

Sources:
- Levine et al., 2020
  DOI: 10.48550/arXiv.2005.01643
- Kim et al., 2024, ScienceDirect review
  https://www.sciencedirect.com/science/article/pii/S0065245823000372

## 4.3 Research interpretation for this project
The current notebook is strongest as:

- a way to initialize the actor close to the MPC teacher,
- a way to reduce early online training burden,
- a way to seed a policy with the right input/output geometry.

It is weakest as:

- the final proof that the policy will behave like MPC in closed loop,
- a trustworthy estimate of rollout performance,
- a complete offline RL solution.

So this part is **worth keeping**, but it should be treated as:

`pretraining / initialization`

not as:

`full validation of the learned controller`.

## 5. Recommended Changes, Ordered by Priority

## Priority 1: Connect pretraining to the actual downstream workflow
Right now, the pretraining notebook and the downstream RL notebooks are not aligned on:

- architecture,
- checkpoint naming,
- checkpoint consumption.

This should be fixed first. Otherwise, improvements to pretraining may never reach the policy that is actually being used later.

## Priority 2: Add a true validation stage to the pretraining notebook
At minimum:

- train/validation split for action imitation,
- held-out `R^2` or MSE,
- closed-loop rollout against teacher MPC on fixed scenarios.

Without this, the notebook cannot tell whether it learned a good controller or just memorized the buffer.

## Priority 3: Replace i.i.d. random-state sampling with trajectory-distributed MPC data
The current buffer generation samples:

- augmented states,
- setpoints,
- previous inputs

independently and uniformly.

That is convenient, but it does not match the state distribution the learned policy will actually induce.

The literature strongly suggests moving toward:

- MPC closed-loop trajectories,
- maybe a mixture of random-state coverage plus trajectory rollouts,
- possibly DAgger-like aggregation later.

## Priority 4: Unify the TD3 transition definition between runners
The plain RL and safe RL runners should not disagree on whether the TD3 next-state contains `y_sp[k]` or `y_sp[k+1]`, unless that is explicitly intentional and documented.

This is a real consistency issue.

## Priority 5: Eliminate duplicated scaling logic
There should be one canonical RL scaling helper only.

Specifically:

- keep `utils/scaling_helpers.py`,
- retire the duplicate `apply_rl_scaled(...)` in `utils/helpers.py`,
- avoid notebook-level scaler duplication where possible.

## Priority 6: Make checkpoint loading self-checking
`TD3Agent.load(...)` should eventually verify:

- architecture match,
- state/action dimensions,
- key hyperparameters,
- optionally scaler metadata or a scaler checksum.

Without that, it is too easy to silently load a policy into the wrong runtime context.

## Priority 7: Decide deliberately what critic pretraining is supposed to mean here
The current critic pretraining is not obviously wrong, but its meaning is limited:

- fixed offline dataset,
- all `done = 0`,
- synthetic one-step teacher transitions.

This may still help, but it should be tested rather than assumed. The actor imitation stage is currently much more clearly justified than the critic pretraining stage.

## 6. What I Would Call “Strong Enough” vs “Not Yet Strong Enough”

### Strong enough already
- centralized system/scaler loading through `load_and_prepare_system_data(...)`
- a coherent legacy augmentation path
- a consistent RL state layout across the main active runners
- MPC imitation as a controller initialization strategy
- repeated notebook use of the same hard-coded `x_min/x_max` override

### Not yet strong enough
- downstream connection from pretraining to actual active checkpoints
- validation methodology
- dataset realism
- runner-to-runner consistency for TD3 transitions
- scaler governance
- checkpoint portability and reproducibility

## 7. Recommended Next Discussion
Before changing code, the best next design decision is:

1. decide whether `pretraining_rl_controller.ipynb` should become the **authoritative source** of the pretrained agent actually used downstream,
2. decide whether the next dataset should be:
   - trajectory-distributed MPC only, or
   - mixed random coverage + trajectory MPC,
3. decide whether critic pretraining stays in scope or whether the next round should focus only on actor imitation + downstream online fine-tuning.

That decision will determine whether the next implementation round should focus on:

- data generation,
- validation,
- checkpoint integration,

or all three.
