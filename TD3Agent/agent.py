import torch
import torch.nn as nn
from typing import List, Literal, Optional
from dataclasses import dataclass
import torch.optim as optim
import numpy as np
import pickle
import math
from torch.nn.utils import parameters_to_vector, vector_to_parameters
import torch.nn.functional as F
from copy import deepcopy

# importing nets and buffer
from TD3Agent.critic import Critic
from TD3Agent.actor import Actor
from TD3Agent.replay_buffer import PERRecentReplayBuffer, ReplayBuffer

import os
from datetime import datetime


# ----------------
# Utilities
# ----------------
def get_device() -> torch.device:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return device


def hard_update(target: nn.Module, online: nn.Module) -> None:
    target.load_state_dict(online.state_dict())


@torch.no_grad()
def soft_update(target: nn.Module, online: nn.Module, tau: float) -> None:
    for tp, p in zip(target.parameters(), online.parameters()):
        tp.data.mul_(1.0 - tau).add_(tau * p.data)


# exploration schedule for Gaussian action noise
@dataclass
class GaussianNoiseSchedule:
    std_start: float = 0.2
    std_end: float = 0.02
    mode: Literal["linear", "exp", "cosine"] = "exp"
    decay_steps: int = 200_000
    decay_rate: float = 0.99995

    def value(self, step: int) -> float:
        if self.mode == "linear":
            t = min(1.0, step / max(1, self.decay_steps))
            return self.std_start + (self.std_end - self.std_start) * t
        if self.mode == "exp":
            return self.std_end + (self.std_start - self.std_end) * (self.decay_rate ** step)
        if self.mode == "cosine":
            t = min(1.0, step / max(1, self.decay_steps))
            return self.std_end + 0.5 * (self.std_start - self.std_end) * (1 + math.cos(math.pi * t))
        raise ValueError("mode must be 'linear' | 'exp' | 'cosine'")


@dataclass
class ParameterNoiseAdaptation:
    initial_std: float = 0.01
    min_std: float = 0.002
    max_std: float = 0.05
    target_action_std: float = 0.05
    adapt_up: float = 1.05
    adapt_down: float = 0.95


def col(x: torch.Tensor) -> torch.Tensor:
    return x if x.ndim == 2 else x.view(-1, 1)


def copy_params_by_order(new_model: nn.Module, old_model: nn.Module):
    with torch.no_grad():
        vec = parameters_to_vector(list(old_model.parameters()))
        vector_to_parameters(vec, list(new_model.parameters()))


class TD3Agent(nn.Module):
    def __init__(
            self,
            state_dim: int,
            action_dim: int,
            actor_hidden: List[int],
            critic_hidden: List[int],
            # learning
            gamma: float = 0.99,
            actor_lr: float = 1e-4,
            critic_lr: float = 1e-3,
            batch_size: int = 256,
            grad_clip_norm: Optional[float] = 10.0,
            # TD3
            policy_delay: int = 2,
            target_policy_smoothing_noise_std: float = 0.2,
            noise_clip: float = 0.5,
            max_action: float = 1.0,
            # targets update
            target_update: Literal["soft", "hard"] = "soft",
            tau: float = 0.005,
            hard_update_interval: int = 10_000,
            target_combine: Literal["min", "max", "mean", "q1"] = "min",
            # architecture of the actor and critic
            activation: str = "relu",
            use_layernorm: bool = False,
            dropout: float = 0.0,
            squash: str = "tanh",
            # exploration
            exploration_schedule: Optional[GaussianNoiseSchedule] = None,
            std_start: float = 1.0,
            std_end: float = 0.05,
            std_decay_rate: float = 0.99,
            std_decay_steps: int = 100_000,
            std_decay_mode: Literal["linear", "exp", "cosine"] = "exp",
            # buffer
            buffer_size: int = 1_000_000,
            # device/opt
            device: Optional[torch.device] = None,
            use_adamw: bool = True,
            # actor freeze
            actor_freeze: int = 0,
            mode: str = None,
            # Regularization
            beta_near: float = 2.0,
            beta_far: float = 0.05,
            alpha_lambda: float = 2.5,
            lambda_eps: float = 1e-6,
            lambda_ema: float = 0.01,
            std_near: float = 0.001,
    ):
        super(TD3Agent, self).__init__()
        self.device = device if device is not None else get_device()

        # --- hparams ---
        self.gamma = gamma
        self.batch_size = batch_size
        self.grad_clip_norm = grad_clip_norm
        self.policy_delay = policy_delay
        self.t_std = target_policy_smoothing_noise_std
        self.noise_clip = noise_clip
        self.max_action = float(max_action)
        self.target_update = target_update
        self.tau = tau
        self.hard_update_interval = hard_update_interval
        self.target_combine = target_combine
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.use_adamw = bool(use_adamw)

        self.steps = 0
        self.train_steps = 0
        self.total_it = 0

        self.actor_freeze = actor_freeze
        # Train with MPC reward or not
        self.mode = mode

        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.actor_hidden = [int(v) for v in actor_hidden]
        self.critic_hidden = [int(v) for v in critic_hidden]
        self.activation = str(activation)
        self.use_layernorm = bool(use_layernorm)
        self.dropout = float(dropout)
        self.squash = str(squash)
        self.loss_fn_actor = nn.MSELoss()

        # --- actors and critic networks ---
        self.actor = Actor(state_dim, action_dim, actor_hidden,
                           activation=activation, use_layernorm=use_layernorm,
                           dropout=dropout, max_action=max_action, squash=squash).to(self.device)
        self.actor_target = Actor(state_dim, action_dim, actor_hidden,
                                  activation=activation, use_layernorm=use_layernorm,
                                  dropout=dropout, max_action=max_action, squash=squash).to(self.device)
        self.critic = Critic(state_dim, action_dim, critic_hidden,
                             activation=activation, use_layernorm=use_layernorm,
                             dropout=dropout).to(self.device)
        self.critic_target = Critic(state_dim, action_dim, critic_hidden,
                                    activation=activation, use_layernorm=use_layernorm,
                                    dropout=dropout).to(self.device)
        self.behavior_actor = deepcopy(self.actor).to(self.device)
        self.behavior_actor.eval()

        hard_update(self.actor_target, self.actor)
        hard_update(self.critic_target, self.critic)
        hard_update(self.behavior_actor, self.actor)

        # --- optimizers and loss function ---
        if self.use_adamw:
            self.actor_optimizer = optim.AdamW(self.actor.parameters(), lr=actor_lr, weight_decay=0.0)
            self.critic_optimizer = optim.AdamW(self.critic.parameters(), lr=critic_lr, weight_decay=0.0)
        else:
            self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
            self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.loss_fn_critic = nn.SmoothL1Loss(reduction="none")  # per-sample Huber



        # Buffer initialization
        if self.mode == "mpc":
            self.buffer = ReplayBuffer(buffer_size, state_dim, action_dim)
        else:
            self.buffer = PERRecentReplayBuffer(buffer_size, state_dim, action_dim)
        self.bc_buffer = ReplayBuffer(buffer_size, state_dim, action_dim)

        # logs
        self.actor_losses, self.critic_losses = [], []
        self.actor_bc_losses = []

        # decay scheduler
        self.expl_sched = exploration_schedule if exploration_schedule is not None else GaussianNoiseSchedule(
            std_start=std_start, std_end=std_end,
            decay_steps=std_decay_steps, mode=std_decay_mode,
            decay_rate=std_decay_rate,
        )
        self.param_noise_cfg = ParameterNoiseAdaptation()
        self.param_noise_std = float(self.param_noise_cfg.initial_std)
        self._parameter_noise_active = False
        self._parameter_noise_last_action_deviation = 0.0
        self._parameter_noise_last_resampled = False
        self._behavior_noise_mode = "none"

    def freeze_actor(self) -> None:
        for p in self.actor.parameters():
            p.requires_grad = False
        self.actor.eval()

    def unfreeze_actor(self) -> None:
        for p in self.actor.parameters():
            p.requires_grad = True
        self.actor.train()

    # -------- interactions ------
    @torch.no_grad()
    def act_eval(self, state: np.ndarray, sigma_eval: float = 0.0) -> np.ndarray:
        s = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        a = self.actor(s)
        if sigma_eval > 0.0:
            a = a + torch.randn_like(a) * sigma_eval
        return a.clamp(-self.max_action, self.max_action).cpu().numpy()

    @torch.no_grad()
    def act_target_eval(self, state: np.ndarray, sigma_eval: float = 0.0) -> np.ndarray:
        s = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        a = self.actor_target(s)
        if sigma_eval > 0.0:
            a = a + torch.randn_like(a) * sigma_eval
        return a.clamp(-self.max_action, self.max_action).cpu().numpy()

    def sync_behavior_actor(self) -> None:
        hard_update(self.behavior_actor, self.actor)
        self.behavior_actor.eval()

    @torch.no_grad()
    def clear_parameter_noise(self) -> None:
        self.sync_behavior_actor()
        self._parameter_noise_active = False
        self._parameter_noise_last_resampled = False

    @torch.no_grad()
    def configure_parameter_noise(
        self,
        initial_std: Optional[float] = None,
        min_std: Optional[float] = None,
        max_std: Optional[float] = None,
        target_action_std: Optional[float] = None,
        adapt_up: Optional[float] = None,
        adapt_down: Optional[float] = None,
    ) -> None:
        if initial_std is not None:
            self.param_noise_cfg.initial_std = float(max(0.0, initial_std))
        if min_std is not None:
            self.param_noise_cfg.min_std = float(max(0.0, min_std))
        if max_std is not None:
            self.param_noise_cfg.max_std = float(max(self.param_noise_cfg.min_std, max_std))
        if target_action_std is not None:
            self.param_noise_cfg.target_action_std = float(max(0.0, target_action_std))
        if adapt_up is not None:
            self.param_noise_cfg.adapt_up = float(max(1.0, adapt_up))
        if adapt_down is not None:
            self.param_noise_cfg.adapt_down = float(min(1.0, max(0.0, adapt_down)))
        self.param_noise_std = float(
            np.clip(self.param_noise_std, self.param_noise_cfg.min_std, self.param_noise_cfg.max_std)
        )

    @torch.no_grad()
    def resample_parameter_noise(self, std: Optional[float] = None) -> float:
        if std is not None:
            self.param_noise_std = float(std)
        self.param_noise_std = float(
            np.clip(self.param_noise_std, self.param_noise_cfg.min_std, self.param_noise_cfg.max_std)
        )
        self.sync_behavior_actor()
        if self.param_noise_std <= 0.0:
            self._parameter_noise_active = False
            self._parameter_noise_last_resampled = True
            return self.param_noise_std

        with torch.no_grad():
            vec = parameters_to_vector(list(self.behavior_actor.parameters()))
            noise = torch.randn_like(vec) * float(self.param_noise_std)
            vector_to_parameters(vec + noise, list(self.behavior_actor.parameters()))
        self.behavior_actor.eval()
        self._parameter_noise_active = True
        self._parameter_noise_last_resampled = True
        return self.param_noise_std

    @torch.no_grad()
    def adapt_parameter_noise(self, states: np.ndarray) -> float:
        states = np.asarray(states, dtype=np.float32)
        if states.size == 0:
            self._parameter_noise_last_action_deviation = 0.0
            return self.param_noise_std

        s = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        nominal = self.actor(s)
        perturbed = self.behavior_actor(s)
        denom = max(float(self.max_action), 1e-6)
        diff = (perturbed - nominal) / denom
        deviation = torch.sqrt(torch.mean(diff * diff, dim=1)).mean().item()
        self._parameter_noise_last_action_deviation = float(deviation)

        if deviation < float(self.param_noise_cfg.target_action_std):
            updated = float(self.param_noise_std) * float(self.param_noise_cfg.adapt_up)
        else:
            updated = float(self.param_noise_std) * float(self.param_noise_cfg.adapt_down)
        self.param_noise_std = float(
            np.clip(updated, self.param_noise_cfg.min_std, self.param_noise_cfg.max_std)
        )
        return self.param_noise_std

    def get_behavior_noise_diagnostics(self) -> dict:
        return {
            "behavior_noise_mode": str(self._behavior_noise_mode),
            "parameter_noise_active": bool(self._parameter_noise_active),
            "parameter_noise_std": float(self.param_noise_std),
            "parameter_noise_last_action_deviation": float(self._parameter_noise_last_action_deviation),
            "parameter_noise_last_resampled": bool(self._parameter_noise_last_resampled),
        }

    @torch.no_grad()
    def apply_exploration(
        self,
        action: np.ndarray,
        sigma_override: Optional[float] = None,
        advance_step: bool = False,
    ) -> np.ndarray:
        if advance_step:
            self.steps += 1

        if sigma_override is None:
            sigma = float(self.expl_sched.value(self.steps))
        else:
            sigma = float(max(0.0, sigma_override))

        self._behavior_noise_mode = "gaussian"
        self._parameter_noise_last_resampled = False
        self._expl_sigma = sigma
        a = np.asarray(action, dtype=np.float32).copy()
        if sigma > 0.0:
            a = a + np.random.randn(*a.shape).astype(np.float32) * sigma
        return np.clip(a, -self.max_action, self.max_action)

    @torch.no_grad()
    def take_behavior_action(
        self,
        state: np.ndarray,
        behavior_noise_mode: str = "gaussian",
        sigma_override: Optional[float] = None,
        advance_step: bool = True,
    ) -> np.ndarray:
        mode = str(behavior_noise_mode).strip().lower()
        if advance_step:
            self.steps += 1

        s = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        if mode == "none":
            self._behavior_noise_mode = "none"
            self._parameter_noise_last_resampled = False
            self._expl_sigma = 0.0
            self._parameter_noise_active = False
            return self.actor(s).detach().cpu().numpy().clip(-self.max_action, self.max_action)
        if mode == "gaussian":
            a = self.actor(s).detach().cpu().numpy()
            return self.apply_exploration(a, sigma_override=sigma_override, advance_step=False)
        if mode == "parameter":
            self._behavior_noise_mode = "parameter"
            self._expl_sigma = 0.0
            a = self.behavior_actor(s).detach().cpu().numpy()
            return a.clip(-self.max_action, self.max_action)
        raise ValueError("behavior_noise_mode must be 'none', 'gaussian', or 'parameter'.")

    @torch.no_grad()
    def take_action(
        self,
        state: np.ndarray,
        explore: bool = False,
        sigma_override: Optional[float] = None,
    ) -> np.ndarray:
        s = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        a = self.actor(s).detach().cpu().numpy()
        if explore:
            return self.apply_exploration(
                a,
                sigma_override=sigma_override,
                advance_step=True,
            )
        self._behavior_noise_mode = "none"
        self._parameter_noise_last_resampled = False
        self._expl_sigma = 0.0
        return np.clip(a, -self.max_action, self.max_action)


    def push(self, s, a, r, ns, done):
        self.buffer.push(s, a, r, ns, done)

    def push_actor_demo(self, s, a):
        self.bc_buffer.push(s, a, 0.0, s, 0.0)

    def pretrain_push(self, s, a, r, ns,):
        self.buffer.pretrain_add(s, a, r, ns)

    def set_actor_lr(self, new_lr: float):
        for g in self.actor_optimizer.param_groups:
            g['lr'] = new_lr
        if self.total_it % 800 == 1:
            print(f"Actor learning rate changed to: {new_lr:.2e}")


    # ------ training ------
    def train_step(self, actor_update: bool = True) -> Optional[float]:
        # check if the buffer has enough info
        if len(self.buffer) < self.batch_size:
            return None

        if self.mode == "mpc":
            s, a, r, ns, done = self.buffer.sample(self.batch_size, device=self.device)
        else:
            s, a, r, ns, done, idx, is_w = self.buffer.sample(self.batch_size, device=self.device)
            is_w = col(is_w.to(self.device, non_blocking=True).float())  # [B, 1]

        s = s.to(self.device, non_blocking=True).float()  # [B, S]
        a = a.to(self.device, non_blocking=True).float()  # [B, A]
        r = col(r.to(self.device, non_blocking=True).float())  # [B, 1]
        ns = ns.to(self.device, non_blocking=True).float()  # [B, S]
        done = col(done.to(self.device, non_blocking=True).float())  # [B, 1]


        with torch.no_grad():
            # target policy smoothing
            base_next = self.actor_target(ns)
            noise = torch.empty_like(base_next).normal_(0.0, self.t_std)
            noise.clamp_(-self.noise_clip, self.noise_clip)
            next_action = (base_next + noise).clip(-self.max_action, self.max_action)

            # Next q value
            q_next = self.critic_target.combined_forward(ns, next_action, mode=self.target_combine)
            if self.mode == "mpc":
                y = r + self.gamma * q_next
            else:
                y = r + self.gamma * (1.0 - done) * q_next



        # critic update
        q1, q2 = self.critic(s, a)
        q1 = col(q1)
        q2 = col(q2)
        td = torch.max((y - q1).abs(), (y - q2).abs()).detach().view(-1)
        l1 = self.loss_fn_critic(q1, y)
        l2 = self.loss_fn_critic(q2, y)
        if self.mode == "mpc":
            critic_loss = (l1 + l2).mean()
        else:
            critic_loss = (is_w * (l1 + l2)).mean()

        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        if self.grad_clip_norm is not None:
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_clip_norm)
        self.critic_optimizer.step()
        self.critic_losses.append(float(critic_loss.item()))

        # ------ Delayed actor + target update -------
        if self.total_it % self.policy_delay == 0:
            curr = self.actor(s)
            q_for_actor = self.critic.q1_forward(s, curr)
            actor_loss = -torch.mean(q_for_actor)

            do_actor = actor_update and (self.total_it >= self.actor_freeze)
            if do_actor:
                self.actor_optimizer.zero_grad(set_to_none=True)
                actor_loss.backward()
                if self.grad_clip_norm is not None:
                    nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip_norm)
                self.actor_optimizer.step()
            self.actor_losses.append(float(actor_loss.item()))

            if self.target_update == "soft":
                soft_update(self.actor_target, self.actor, self.tau)
                soft_update(self.critic_target, self.critic, self.tau)
            else:
                if self.train_steps % self.hard_update_interval == 0:
                    hard_update(self.actor_target, self.actor)
                    hard_update(self.critic_target, self.critic)

        self.total_it += 1
        self.train_steps += 1

        # --------- PER: update priorities from |TD| ----------
        if self.mode != "mpc":
            if hasattr(self.buffer, "update_priorities"):
                self.buffer.update_priorities(idx, td.abs())

        return float(critic_loss.item())

    def train_actor_bc_step(self, batch_size: Optional[int] = None) -> Optional[float]:
        batch_size = int(batch_size or self.batch_size)
        if len(self.bc_buffer) < batch_size:
            return None

        s, a, _r, _ns, _done = self.bc_buffer.sample(batch_size, device=self.device)
        s = s.to(self.device, non_blocking=True).float()
        a = a.to(self.device, non_blocking=True).float()

        pred = self.actor(s)
        bc_loss = self.loss_fn_actor(pred, a)

        self.actor_optimizer.zero_grad(set_to_none=True)
        bc_loss.backward()
        if self.grad_clip_norm is not None:
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip_norm)
        self.actor_optimizer.step()

        if self.target_update == "soft":
            soft_update(self.actor_target, self.actor, self.tau)
        else:
            hard_update(self.actor_target, self.actor)

        bc_value = float(bc_loss.item())
        self.actor_losses.append(bc_value)
        self.actor_bc_losses.append(bc_value)
        return bc_value

    def pretrain_from_buffer(
            self,
            num_actor_epochs: int = 50,
            num_critic_epochs: int = 20,
            data_loader=None,
            use_target_noise_critic: bool = False,
            log_interval: int = 1,
            mode: str = "mpc",
            sched_kind: str = "cosine",
    ) -> dict:
        """
        Two-stage pretraining:
        Stage 1: actor behavioral cloning (pure imitation).
        Stage 2: freeze actor, train critic using TD targets under cloned policy.

        If data_loader is provided, it must yield either:
          (s, a, r, ns) or (s, a, r, ns, done).
        Otherwise, samples come from self.buffer (random batches).
        """
        self.mode = mode
        history = {
            "mode": mode,
            "sched_kind": sched_kind,
            "actor_bc_losses": [],
            "critic_losses": [],
            "actor_bc_lrs": [],
            "critic_lrs": [],
            "actor_bc_samples": [],
            "critic_samples": [],
        }

        if data_loader is None and len(self.buffer) < self.batch_size:
            raise RuntimeError("Buffer is less than the batch size")

        def make_scheduler(opt, epochs: int):
            if sched_kind == "cosine":
                return optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))
            if sched_kind == "step":
                return optim.lr_scheduler.StepLR(opt, step_size=max(1, epochs // 3), gamma=0.5)
            return None

        # -------------------------
        # Stage 1: actor BC only
        # -------------------------
        self.actor.train()
        self.critic.eval()

        sch_a = make_scheduler(self.actor_optimizer, num_actor_epochs)

        for ep in range(1, num_actor_epochs + 1):
            bc_loss_sum = 0.0
            n_sum = 0

            if data_loader is not None:
                batches = data_loader
            else:
                batches = range(max(1, len(self.buffer) // self.batch_size))

            for batch in batches:
                if data_loader is not None:
                    pack = batch
                    if len(pack) == 4:
                        s, a, r, ns = pack
                        done = None
                    else:
                        s, a, r, ns, done = pack
                    s = s.to(self.device).float()
                    a = a.to(self.device).float()
                else:
                    s, a, r, ns, done = self.buffer.sample(self.batch_size, device=self.device)
                    s = s.to(self.device).float()
                    a = a.to(self.device).float()

                pred = self.actor(s)
                bc_loss = self.loss_fn_actor(pred, a)

                self.actor_optimizer.zero_grad(set_to_none=True)
                bc_loss.backward()
                if self.grad_clip_norm is not None:
                    nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip_norm)
                self.actor_optimizer.step()

                bs = int(s.shape[0])
                bc_loss_sum += float(bc_loss.item()) * bs
                n_sum += bs

            if sch_a is not None:
                sch_a.step()

            hard_update(self.actor_target, self.actor)
            epoch_bc_loss = bc_loss_sum / max(1, n_sum)
            self.actor_losses.append(float(epoch_bc_loss))
            self.actor_bc_losses.append(float(epoch_bc_loss))
            lr_a = float(self.actor_optimizer.param_groups[0]["lr"])
            history["actor_bc_losses"].append(float(epoch_bc_loss))
            history["actor_bc_lrs"].append(lr_a)
            history["actor_bc_samples"].append(int(n_sum))

            if log_interval and (ep == 1 or ep % log_interval == 0):
                print(f"[pretrain][actor_bc] ep={ep} loss={epoch_bc_loss:.4e} lr={lr_a:.2e}")

        # -------------------------
        # Stage 2: critic TD with frozen actor
        # -------------------------
        self.freeze_actor()
        self.critic.train()

        sch_c = make_scheduler(self.critic_optimizer, num_critic_epochs)

        for ep in range(1, num_critic_epochs + 1):
            q_loss_sum = 0.0
            n_sum = 0

            if data_loader is not None:
                batches = data_loader
            else:
                batches = range(max(1, len(self.buffer) // self.batch_size))

            for batch in batches:
                if data_loader is not None:
                    pack = batch
                    if len(pack) == 4:
                        s, a, r, ns = pack
                        done = None
                    else:
                        s, a, r, ns, done = pack
                    s = s.to(self.device).float()
                    a = a.to(self.device).float()
                    r = col(r.to(self.device).float())
                    ns = ns.to(self.device).float()
                    if done is None:
                        done = torch.zeros((s.shape[0],), device=self.device)
                    done = col(done.to(self.device).float())
                else:
                    s, a, r, ns, done = self.buffer.sample(self.batch_size, device=self.device)
                    s = s.to(self.device).float()
                    a = a.to(self.device).float()
                    r = col(r.to(self.device).float())
                    ns = ns.to(self.device).float()
                    done = col(done.to(self.device).float())

                with torch.no_grad():
                    base_next = self.actor_target(ns)
                    if use_target_noise_critic:
                        noise = torch.empty_like(base_next).normal_(0.0, self.t_std)
                        noise.clamp_(-self.noise_clip, self.noise_clip)
                        next_action = (base_next + noise).clamp(-self.max_action, self.max_action)
                    else:
                        next_action = base_next.clamp(-self.max_action, self.max_action)

                    q_next = self.critic_target.combined_forward(ns, next_action, mode=self.target_combine)
                    y = r + self.gamma * (1.0 - done) * q_next

                q1, q2 = self.critic(s, a)
                q1 = col(q1)
                q2 = col(q2)

                l1 = self.loss_fn_critic(q1, y)
                l2 = self.loss_fn_critic(q2, y)
                critic_loss = (l1 + l2).mean()

                self.critic_optimizer.zero_grad(set_to_none=True)
                critic_loss.backward()
                if self.grad_clip_norm is not None:
                    nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_clip_norm)
                self.critic_optimizer.step()

                if self.target_update == "soft":
                    soft_update(self.critic_target, self.critic, self.tau)
                else:
                    if self.train_steps % self.hard_update_interval == 0:
                        hard_update(self.critic_target, self.critic)

                bs = int(s.shape[0])
                q_loss_sum += float(critic_loss.item()) * bs
                n_sum += bs

                self.train_steps += 1

            if sch_c is not None:
                sch_c.step()

            epoch_critic_loss = q_loss_sum / max(1, n_sum)
            self.critic_losses.append(float(epoch_critic_loss))
            lr_c = float(self.critic_optimizer.param_groups[0]["lr"])
            history["critic_losses"].append(float(epoch_critic_loss))
            history["critic_lrs"].append(lr_c)
            history["critic_samples"].append(int(n_sum))

            if log_interval and (ep == 1 or ep % log_interval == 0):
                print(f"[pretrain][critic_td] ep={ep} loss={epoch_critic_loss:.4e} lr={lr_c:.2e}")

        hard_update(self.actor_target, self.actor)
        hard_update(self.critic_target, self.critic)
        self.unfreeze_actor()
        return history



    def load(self, path: str):
        with open(path, 'rb') as f:
            d = pickle.load(f)

        self.actor.load_state_dict(d['actor_state_dict'])
        self.critic.load_state_dict(d['critic_state_dict'])
        hard_update(self.actor_target, self.actor)
        hard_update(self.critic_target, self.critic)

        # re-init optimizers; checkpoint optimizer states are intentionally not restored.
        if self.use_adamw:
            self.actor_optimizer = torch.optim.AdamW(self.actor.parameters(), lr=self.actor_lr, weight_decay=0.0)
            self.critic_optimizer = torch.optim.AdamW(self.critic.parameters(), lr=self.critic_lr, weight_decay=0.0)
        else:
            self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.actor_lr)
            self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.critic_lr)

        print(f"Agent loaded successfully from: {path}")

    def reset_critic(self) -> None:
        """Reinitialize both critic networks while preserving the actor state."""
        self.critic = Critic(
            self.state_dim,
            self.action_dim,
            self.critic_hidden,
            activation=self.activation,
            use_layernorm=self.use_layernorm,
            dropout=self.dropout,
        ).to(self.device)
        self.critic_target = Critic(
            self.state_dim,
            self.action_dim,
            self.critic_hidden,
            activation=self.activation,
            use_layernorm=self.use_layernorm,
            dropout=self.dropout,
        ).to(self.device)
        hard_update(self.critic_target, self.critic)
        if self.use_adamw:
            self.critic_optimizer = optim.AdamW(self.critic.parameters(), lr=self.critic_lr, weight_decay=0.0)
        else:
            self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.critic_lr)
        self.critic_losses = []

    def save(
            self,
            directory: str,
            prefix: str = "td3",
            include_optim: bool = False,
    ) -> str:
        """
        Save a checkpoint to `directory` with filename based on current time.
        Returns the full path to the saved pickle.

        Notes:
        - Your existing `load(...)` only reads 'actor_state_dict' and 'critic_state_dict'.
          Extra keys saved here are harmless and simply ignored by that loader.
        - Set `include_optim=True` if you later add a loader that restores optimizers.
        """
        os.makedirs(directory, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(directory, f"{prefix}_{timestamp}.pkl")

        payload = {
            # what your current `load(...)` expects:
            "actor_state_dict": self.actor.state_dict(),
            "critic_state_dict": self.critic.state_dict(),

            # nice-to-have extras (ignored by your current loader):
            "actor_target_state_dict": self.actor_target.state_dict(),
            "critic_target_state_dict": self.critic_target.state_dict(),
            "hparams": {
                "state_dim": self.state_dim,
                "action_dim": self.action_dim,
                "actor_hidden": list(self.actor_hidden),
                "critic_hidden": list(self.critic_hidden),
                "gamma": self.gamma,
                "actor_lr": self.actor_lr,
                "critic_lr": self.critic_lr,
                "batch_size": self.batch_size,
                "grad_clip_norm": self.grad_clip_norm,
                "policy_delay": self.policy_delay,
                "target_update": self.target_update,
                "tau": self.tau,
                "hard_update_interval": self.hard_update_interval,
                "target_combine": self.target_combine,
                "t_std": self.t_std,
                "noise_clip": self.noise_clip,
                "max_action": self.max_action,
                "actor_freeze": self.actor_freeze,
                "mode": self.mode,
                "activation": self.activation,
                "use_layernorm": self.use_layernorm,
                "dropout": self.dropout,
                "squash": self.squash,
                "steps": self.steps,
                "train_steps": self.train_steps,
                "total_it": self.total_it,
            },
            "training_losses": {
                "actor_losses": [float(v) for v in self.actor_losses],
                "actor_bc_losses": [float(v) for v in self.actor_bc_losses],
                "critic_losses": [float(v) for v in self.critic_losses],
            },
        }

        if include_optim:
            payload["actor_optimizer_state_dict"] = self.actor_optimizer.state_dict()
            payload["critic_optimizer_state_dict"] = self.critic_optimizer.state_dict()

        with open(path, "wb") as f:
            pickle.dump(payload, f)

        print(f"Saved TD3 checkpoint to: {path}")
        return path

