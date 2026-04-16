# Pretraining RL Controller Notebook Review

## Scope
This note reviews the notebook [pretraining_rl_controller.ipynb](C:/Users/HAMEDI/Desktop/Lyapunov_polymer/pretraining_rl_controller.ipynb) line by line and traces the actual Python code it calls in:

- [utils/td3_helpers.py](C:/Users/HAMEDI/Desktop/Lyapunov_polymer/utils/td3_helpers.py)
- [TD3Agent/agent.py](C:/Users/HAMEDI/Desktop/Lyapunov_polymer/TD3Agent/agent.py)
- [TD3Agent/replay_buffer.py](C:/Users/HAMEDI/Desktop/Lyapunov_polymer/TD3Agent/replay_buffer.py)
- [Simulation/mpc.py](C:/Users/HAMEDI/Desktop/Lyapunov_polymer/Simulation/mpc.py)

The goal is to capture what the notebook is really doing now, not what it appears to be doing at first glance.

## Executive Summary
This notebook is not online RL training.

It is an **offline MPC-to-TD3 pretraining pipeline**:

1. build an offset-free augmented linear model and scaling metadata,
2. define a TD3 actor/critic,
3. fill a replay buffer with synthetic state/setpoint/input samples,
4. label those samples with the **first move of an MPC solve**,
5. pretrain the actor by behavioral cloning,
6. freeze the actor and pretrain the critic with TD targets on the same offline dataset,
7. save the checkpoint,
8. report an in-sample `R^2` agreement with the MPC actions.

So the notebook is best understood as:

`supervised imitation of an MPC policy + offline critic fitting`

not as:

`closed-loop RL training on the polymer plant`

## Cell-By-Cell Flow

### Cells 1 to 7: plant and steady-state definition
The notebook imports:

- `PolymerCSTR`
- `load_and_prepare_system_data`
- `filling_the_buffer`
- `add_steady_state_samples`
- `print_accuracy`
- `ReplayDataset`
- `TD3Agent`
- `MpcSolver`

Then it defines the polymer CSTR parameters and creates:

- `system_params`
- `system_design_params`
- `system_steady_state_inputs`
- `delta_t = 0.5`

Finally it instantiates:

```python
cstr = PolymerCSTR(system_params, system_design_params, system_steady_state_inputs, delta_t)
steady_states = {
    "ss_inputs": cstr.ss_inputs,
    "y_ss": cstr.y_ss,
}
```

At this stage, the notebook uses the nonlinear plant only to recover steady-state anchors. It does **not** run closed-loop simulation in this notebook.

### Cells 9 to 16: model loading and scaling
The notebook calls:

```python
system_data = load_and_prepare_system_data(
    steady_states=steady_states,
    setpoint_y=setpoint_y,
    u_min=u_min,
    u_max=u_max,
)
```

Because no augmentation arguments are passed, the helper uses the default:

- `augmentation_style="legacy"`
- `augment_state_space(A, B, C)`

That means the augmented model is:

\[
z_{k+1} =
\begin{bmatrix}
A & 0 \\
0 & I
\end{bmatrix} z_k +
\begin{bmatrix}
B \\
0
\end{bmatrix} u_k,
\qquad
y_k = [\,C \;\; I\,] z_k
\]

So the final augmented state is:

\[
z_k = [x_k; d_k]
\]

with an output-disturbance integrator.

The helper also loads:

- [Data/system_dict](C:/Users/HAMEDI/Desktop/Lyapunov_polymer/Data/system_dict)
- [Data/scaling_factor.pickle](C:/Users/HAMEDI/Desktop/Lyapunov_polymer/Data/scaling_factor.pickle)
- [Data/min_max_states.pickle](C:/Users/HAMEDI/Desktop/Lyapunov_polymer/Data/min_max_states.pickle)

and constructs:

- `A_aug`, `B_aug`, `C_aug`
- `data_min`, `data_max`
- `b_min`, `b_max`
- `min_max_dict`

Important scaling convention:

- physical plant outputs and inputs are min-max scaled to `[0,1]`
- then the RL state and action channels are mapped to `[-1,1]` using `apply_min_max_pm1(...)`
- the controller is trained in **scaled deviation coordinates**

The RL state built by the helper stack is:

\[
s_k = \big[
\text{scaled}(x\_d),\;
\text{scaled}(y_{sp}),\;
\text{scaled}(u_{k-1})
\big]
\]

where all three blocks are mapped to `[-1,1]`.

The notebook then manually overwrites:

- `min_max_dict["x_max"]`
- `min_max_dict["x_min"]`

with hard-coded arrays. That means the actual RL state scaling for the augmented-state block is **not** taken purely from the file-loaded `min_max_states.pickle`; it is explicitly forced by the notebook.

This is important. Any future change to pretraining behavior must treat these manual overrides as part of the pipeline, not as an incidental notebook detail.

### Cells 18 to 19: TD3 agent configuration
The notebook defines:

- `STATE_DIM = A_aug.shape[0] + n_outputs + n_inputs`
- `ACTION_DIM = B_aug.shape[1]`

With the current default augmentation, this implies:

- augmented state dimension = `9`
- output dimension = `2`
- input dimension = `2`
- total RL state dimension = `13`

The network sizes are:

- actor: `[256, 256, 256]`
- critic: `[256, 256, 256]`

Other important hyperparameters:

- `BUFFER_CAPACITY = 5_000_000`
- `ACTOR_LR = 1e-3`
- `CRITIC_LR = 1e-3`
- `GAMMA = 0.99`
- `TAU = 0.005`
- `BATCH_SIZE = 128`
- `POLICY_DELAY = 2`
- `SMOOTHING_STD = 1e-6`
- `NOISE_CLIP = 1e-6`
- exploration std schedule from `0.02` down to `0.001`

The agent is instantiated with:

```python
mode="mpc"
```

That single line changes behavior in a meaningful way:

- the buffer becomes the simple uniform [ReplayBuffer](C:/Users/HAMEDI/Desktop/Lyapunov_polymer/TD3Agent/replay_buffer.py)
- not the prioritized/recent replay buffer
- `train_step(...)` uses the `mode == "mpc"` branch
- for pretraining this is appropriate, because the data is synthetic MPC data rather than online RL experience

### Cells 21 to 25: teacher MPC and replay-buffer generation
The teacher controller is a simple finite-horizon MPC:

- prediction horizon `NP = 9`
- control horizon `NC = 3`
- `Q_out = [5, 1]`
- `R_in = [1, 1]`

The MPC cost implemented in [Simulation/mpc.py](C:/Users/HAMEDI/Desktop/Lyapunov_polymer/Simulation/mpc.py) is:

\[
J =
\sum_{j=1}^{N_P} \| y_{k+j|k} - y_{sp} \|_{Q}^2
+
\sum_{j=0}^{N_C-1} \| \Delta u_{k+j|k} \|_{R}^2
\]

where:

- the teacher solves over a stacked move vector,
- and the label used for imitation is **only the first control move**.

The notebook reserves:

- `4,900,000` samples for generic MPC imitation
- `100,000` samples for near-steady-state augmentation

#### `filling_the_buffer(...)`
This helper:

1. samples random augmented states uniformly between `x_min` and `x_max`,
2. samples random setpoints uniformly in the deviation setpoint box,
3. samples random previous inputs uniformly in the admissible input-deviation box,
4. solves an MPC problem for each sample,
5. stores:
   - state
   - action = first MPC move
   - reward
   - next_state

The reward is not the plant reward from an actual rollout. It is a synthetic one-step quadratic teacher reward:

\[
r = - \left(
\Delta y^\top Q_{\text{rew}} \Delta y
+
\Delta u^\top R_{\text{rew}} \Delta u
\right)
\]

with:

- \(\Delta y = y_{\text{pred}} - y_{sp}\)
- \(\Delta u = u_{\text{prev}} - u_{\text{mpc}}\)

So the dataset is generated from the linear teacher model directly, not from plant rollouts.

#### `add_steady_state_samples(...)`
This helper adds near-equilibrium samples by:

- sampling `x_d_states` from a tiny Gaussian centered at zero,
- using `y_sp = 0`,
- using almost-zero previous inputs in deviation coordinates

This is meant to densify the dataset around the steady operating region.

### Cells 27 to 28: offline pretraining
The notebook converts the full buffer into tensors and builds a `DataLoader` with:

- `batch_size=8192`
- `shuffle=True`
- `drop_last=True`

Then it calls:

```python
td3_agent.pretrain_from_buffer(
    num_actor_epochs=1000,
    num_critic_epochs=500,
    data_loader=data_loader,
    use_target_noise_critic=True,
    log_interval=10,
)
```

This function has two distinct stages.

#### Stage 1: actor behavioral cloning
The actor is trained with plain MSE:

\[
\mathcal{L}_{\text{BC}} =
\| \pi_\theta(s) - a_{\text{MPC}} \|_2^2
\]

This is pure supervised imitation of the teacher action labels.

No critic term is used in this stage.

After each epoch block, the actor target network is hard-updated to the online actor.

#### Stage 2: critic TD fitting with frozen actor
The actor is frozen and the critic is trained using TD targets:

\[
y = r + \gamma (1-d) Q_{\phi^-}(s', \pi_{\theta^-}(s'))
\]

where:

- the next action is generated by `actor_target(ns)`,
- target noise is optionally added,
- `done` is effectively zero for the pretraining data path

Important subtlety:

Because the notebook-pretraining dataset is created by `ReplayBuffer.pretrain_add(...)`, the `done` array is never meaningfully populated. In the DataLoader path, `done` is treated as zero if missing. So this critic pretraining is effectively an infinite-horizon-style bootstrap on synthetic one-step transitions.

Also note:

- `use_target_noise_critic=True`
- but `SMOOTHING_STD = 1e-6`
- and `NOISE_CLIP = 1e-6`

So the target-policy smoothing is technically on but numerically almost zero.

### Cells 30 to 32: save and evaluate
The checkpoint is saved with:

```python
filename_agent = td3_agent.save(dir_path)
```

The current `save(...)` method stores:

- `actor_state_dict`
- `critic_state_dict`
- target-network weights
- some hparams metadata

but the current `load(...)` path only restores:

- actor weights
- critic weights

and then hard-copies those to the target networks.

Finally the notebook calls:

```python
print_accuracy(td3_agent, n_samples=20, device=DEVICE)
```

This computes `R^2` between:

- the actor predictions
- and the stored MPC teacher actions

using samples already inside the replay buffer.

So this is an **in-sample imitation metric**, not a held-out generalization metric and not a closed-loop performance metric.

## What The Notebook Is Really Training
The notebook is training a policy approximation:

\[
\pi_{\theta}(s)
\approx
u^{\text{MPC}}_0(s)
\]

where:

- \(s\) is the scaled concatenation of augmented state, setpoint, and previous input
- \(u^{\text{MPC}}_0(s)\) is the first move of a teacher MPC solve

So the actor is trying to be a **fast surrogate for the first control move of MPC**.

The critic is then fitted on the same synthetic one-step transition dataset so that the pretrained actor/critic pair is numerically initialized before later online fine-tuning.

## Why It Works Partly
The notebook can work reasonably well because:

1. the policy class is expressive enough for a smooth first-move surrogate,
2. the state, setpoint, and previous-input channels are normalized to `[-1,1]`,
3. the teacher action labels come from a consistent optimization problem,
4. the dataset is extremely large,
5. the additional steady-state samples reduce the risk of poor behavior near the operating point.

This is a sensible way to initialize a controller before online training.

## Main Weaknesses And Risks

### 1. The dataset distribution is synthetic, not closed-loop
The random samples are drawn independently across:

- augmented state,
- setpoint,
- previous input.

That means the training distribution is not the actual on-policy or even typical closed-loop state distribution. Many sampled points may be dynamically implausible.

This can make the actor good as a box-wide interpolator while still being weak on the true trajectory distribution that matters online.

### 2. `print_accuracy(...)` is an in-sample metric
The notebook currently checks agreement against samples from the same replay buffer that was used for pretraining. That can overstate quality.

There is no:

- holdout split,
- validation loader,
- closed-loop benchmark in this notebook.

### 3. The critic pretraining signal is bootstrapped on synthetic data
The critic is not trained against true long-horizon returns from plant rollouts. It is trained on:

- one-step model-generated transitions
- with bootstrap targets from the frozen cloned actor

This can initialize the critic numerically, but it does not guarantee that the critic is meaningful for later online control.

### 4. The action-noise smoothing settings are effectively off
With:

- `SMOOTHING_STD = 1e-6`
- `NOISE_CLIP = 1e-6`

the TD3 target-noise mechanism is almost irrelevant during pretraining.

### 5. The notebook hard-overrides state scaling
The manual overwrite of:

- `min_max_dict["x_min"]`
- `min_max_dict["x_max"]`

is part of the actual pipeline. If this is forgotten or changed inconsistently, the pretrained policy can degrade sharply.

### 6. Saving/loading is only partial
The current save file contains more metadata than the current load path actually restores. This is fine for inference, but it is not a full training-state checkpoint.

### 7. Compute cost is very high
The dataset size and epoch counts are large:

- 5 million samples
- 1000 actor epochs
- 500 critic epochs

This is expensive and raises the question of whether the same practical quality could be reached with:

- a smaller filtered dataset,
- better trajectory-distributed sampling,
- and a proper validation-based early stopping rule.

## Most Important Improvement Opportunities

### A. Add a real validation split
This is the cleanest first improvement.

What to add:

- hold out a fraction of the generated MPC dataset,
- report actor MSE and `R^2` on train and validation,
- save the best checkpoint by validation metric.

Without this, you do not really know whether the surrogate generalizes.

### B. Move part of the sampling to trajectory-consistent states
Instead of relying almost entirely on uniformly sampled synthetic states, add datasets collected from:

- baseline MPC rollouts,
- setpoint-change trajectories,
- disturbance-change trajectories.

That should align the teacher dataset with the states the controller actually visits online.

### C. Separate the goals of actor pretraining and critic pretraining
Right now the actor pretraining is the strong part. The critic pretraining is much weaker in terms of physical meaning.

So one good improvement path is:

- keep strong behavioral cloning for the actor,
- make the critic pretraining lighter or more targeted,
- then let online RL reshape the critic later.

### D. Add direct closed-loop evaluation in this notebook
This notebook should not end at `print_accuracy(...)`.

It should also answer:

- does the pretrained actor track like the teacher MPC?
- what is the average reward on a held-out scenario?
- how much worse is it than MPC?
- where does it fail?

### E. Revisit the state-distribution weighting
Right now every synthetic sample is treated equally. It may be better to weight samples so that:

- near-trajectory states matter more,
- near-constraint states matter more,
- steady-state samples remain important but not dominant.

## What I Think The Real Gist Is
The notebook is a teacher-student compression pipeline:

- teacher: offset-free linear MPC
- student: TD3 actor
- auxiliary critic: bootstrapped offline initialization

The notebook is trying to produce a policy that can later be fine-tuned online, but the strongest part of it is still the teacher imitation stage, not the critic stage.

So if we want to improve this part, the biggest leverage is probably not in changing TD3 details first. The biggest leverage is in:

1. improving the teacher dataset distribution,
2. adding validation and closed-loop evaluation,
3. making scaling and state construction absolutely stable,
4. deciding whether the critic pretraining is truly helping enough to justify its cost.

## Bottom Line
Yes, I got the gist:

- the notebook is not doing plant RL yet,
- it is building an offline MPC-labeled dataset,
- cloning the first MPC move into a TD3 actor,
- then fitting a critic on the same synthetic transitions,
- and finally checking in-sample agreement with the teacher.

That is a valid and useful pipeline, but it is only a first stage. The main improvement bottlenecks are dataset realism, validation, and closed-loop evaluation, not just network hyperparameters.
