# MACC 2026 3-Minute Talk Notes

## Core message
This work makes reinforcement learning practical for chemical process control by keeping MPC in the loop instead of replacing it.

## Timed outline
- `0:00-0:25` Problem and motivation
- `0:25-1:10` MPC-pretrained RL
- `1:10-1:55` RL-assisted MPC
- `1:55-2:35` Lyapunov-filtered safe RL
- `2:35-3:00` Closing message

## Full script
Good afternoon. My research is about making reinforcement learning practical for industrial process control.

In chemical plants, linear MPC is widely used because it is structured, interpretable, and already accepted in industry. But its performance can degrade when the process is nonlinear, highly coupled, or operating conditions change. Reinforcement learning can adapt from data, but pure RL usually needs too much unsafe exploration. So the main idea in my work is simple: keep MPC in the loop, and let RL improve performance in practical ways.

The first part of my work is MPC-pretrained RL. Here, a TD3 policy is first trained to imitate offset-free MPC and then refined online using an offset-aware reward and a mixed replay strategy. We tested this on a styrene polymerization reactor and on an Aspen Dynamics C2 splitter. Compared with the earlier pretrained RL baseline, the average final offset was reduced by 88 percent on the polymer system and 86 percent on the C2 case, while the faster settling relative to MPC was retained.

The second part is RL-assisted MPC. In this framework, RL does not replace the controller. Instead, it tunes interpretable MPC variables such as model multipliers, weights, horizons, or residual corrections, while MPC still computes the final constrained move. On the polymer case, the combined supervisor gives about 43 percent higher full-run reward than nominal MPC and beats nominal MPC in 98 percent of full runs. For the C2 splitter, the same trend is visible, and the final comparison packaging is still ongoing.

The third part is Lyapunov-filtered safe RL. Here, RL proposes a candidate action, but a model-based safety layer checks whether that move should reach the plant. If the action fails the Lyapunov contraction check, the controller falls back to MPC. This part is completed on the polymer case, and the C2 extension is in progress.

Overall, the contribution is a practical RL control framework for complex process systems. Depending on the application, we can start from MPC, assist MPC, or protect RL with an MPC-based safety layer. The main message is that RL becomes much more realistic for chemical process control when we improve it around MPC instead of trying to discard MPC completely.

## Memory cues
- `Problem`: MPC is practical, pure RL is unsafe.
- `Main idea`: keep MPC in the loop.
- `Project 1`: pretrain TD3 from OF-MPC, then refine online.
- `Project 1 result`: 88 percent polymer, 86 percent C2 offset reduction.
- `Project 2`: RL tunes MPC variables, MPC still gives the final move.
- `Project 2 result`: 43 percent higher full-run reward, 98 percent win rate on polymer.
- `Project 3`: RL proposes, Lyapunov filter checks, MPC fallback protects.
- `Close`: practical RL for chemical plants means working with MPC, not replacing it.

## Short version for practice
My work is about making reinforcement learning practical for chemical process control by keeping MPC in the loop. Linear MPC is still the industrial standard, but it can lose performance under nonlinearities and changing operation. Pure RL can adapt, but unsafe exploration is a major barrier.

So I study three directions. First, MPC-pretrained RL starts from offset-free MPC behavior and improves online. On the polymer reactor and the Aspen C2 splitter, this reduced final offset by 88 percent and 86 percent, while keeping faster settling than MPC. Second, in RL-assisted MPC, RL tunes MPC variables while MPC still computes the control move. On the polymer case, the combined supervisor gives about 43 percent higher full-run reward than nominal MPC and wins in 98 percent of runs. Third, in Lyapunov-filtered safe RL, RL proposes a move, a safety layer certifies it, and MPC takes over if needed.

The overall message is simple: RL becomes much more practical for chemical engineering when it is built around MPC instead of trying to replace it.
