"""算法实现（CTDE-SAC / CTDE-MAPPO）。"""

from __future__ import annotations

import os

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from .buffer import CTDEReplayBuffer
from .config import (
    ACTION_DIM,
    ALPHA_LR,
    BATCH_SIZE,
    DEVICE,
    GAMMA,
    GLOBAL_STATE_DIM,
    GRADIENT_STEPS,
    HIDDEN_DIM,
    INIT_ALPHA,
    LEARNING_RATE,
    NUM_AGENTS,
    NUM_FOLLOWERS,
    POLICY_DELAY,
    PPO_CLIP_EPS,
    PPO_ENTROPY_COEF,
    PPO_EPOCHS,
    PPO_GAE_LAMBDA,
    PPO_LR,
    PPO_MAX_GRAD_NORM,
    PPO_MINIBATCH_SIZE,
    PPO_ROLLOUT_STEPS,
    PPO_TARGET_KL,
    PPO_VALUE_COEF,
    STATE_DIM,
    TARGET_ENTROPY_RATIO,
    TARGET_UPDATE_INTERVAL,
    TAU,
)
from .networks import CentralizedCritic, CentralizedValue, DecentralizedActor


class ActorOnlyPolicy:
    """只加载共享 Actor 的评估策略（支持任意 follower 数）。

    设计目标：
    - **评估/部署**场景不需要 centralized critic/value，也不需要 replay/rollout buffer。
    - 允许你在训练结束后，把 `num_followers` 设得很大（重新创建 topology/env），
      仍然能用同一个共享 actor 做闭环仿真。

    约束：
    - `STATE_DIM`/`ACTION_DIM`/`MAX_NEIGHBORS` 等网络输入输出维度必须与训练时一致。
    - `local_states` 约定 index=0 为 leader，follower 为 1..N。
    """

    def __init__(self, actor: DecentralizedActor, use_amp: bool = False):
        self.actor = actor.to(DEVICE)
        self.actor.eval()
        self.use_amp = bool(use_amp and DEVICE.type == "cuda")
        self._autocast = None
        if self.use_amp:
            from torch.amp import autocast

            self._autocast = lambda: autocast("cuda")

    @staticmethod
    def _extract_actor_state_dict(checkpoint) -> dict:
        # 兼容两种保存形式：
        # 1) {'actor': actor_state_dict, ...}
        # 2) 直接保存 actor_state_dict
        if isinstance(checkpoint, dict) and ("actor" in checkpoint) and isinstance(checkpoint["actor"], dict):
            return checkpoint["actor"]
        if isinstance(checkpoint, dict) and any(k.startswith("local_encoder") or k.startswith("neighbor_encoder") for k in checkpoint.keys()):
            return checkpoint
        raise ValueError("Unrecognized checkpoint format: cannot find actor state_dict")

    @classmethod
    def from_checkpoint(cls, path: str, hidden_dim: int = HIDDEN_DIM, strict: bool = True, use_amp: bool = False):
        ckpt = torch.load(path, map_location=DEVICE)
        actor = DecentralizedActor(hidden_dim=hidden_dim).to(DEVICE)
        actor_sd = cls._extract_actor_state_dict(ckpt)
        actor.load_state_dict(actor_sd, strict=bool(strict))
        return cls(actor=actor, use_amp=use_amp)

    @torch.inference_mode()
    def select_action(self, local_states: torch.Tensor, deterministic: bool = True):
        """根据本地状态选择动作（面向 follower）。

        Args:
            local_states:
                - 单环境：shape=(num_agents, STATE_DIM)
                - 并行环境：shape=(E, num_agents, STATE_DIM)
            deterministic: 评估默认 True。

        Returns:
            follower 动作张量：
                - 单环境：shape=(num_followers, ACTION_DIM)
                - 并行环境：shape=(E, num_followers, ACTION_DIM)
        """
        is_batched = local_states.dim() == 3

        if is_batched:
            E = int(local_states.shape[0])
            num_followers = int(local_states.shape[1]) - 1
            follower_states = local_states[:, 1:, :]
            flat_states = follower_states.reshape(-1, STATE_DIM)

            if self.use_amp:
                with self._autocast():
                    action, _ = self.actor(flat_states, deterministic=bool(deterministic))
            else:
                action, _ = self.actor(flat_states, deterministic=bool(deterministic))

            action = action.view(E, num_followers, ACTION_DIM)
            return action.float()

        num_followers = int(local_states.shape[0]) - 1
        follower_states = local_states[1:, :]
        if self.use_amp:
            with self._autocast():
                action, _ = self.actor(follower_states, deterministic=bool(deterministic))
        else:
            action, _ = self.actor(follower_states, deterministic=bool(deterministic))
        return action.float()


class CTDESACAgent:
    """CTDE-SAC 智能体（集中训练、分散执行）。

    - Actor：对每个 follower 使用本地 flat state 生成动作。
    - Critic：集中式 Q 网络，输入为 global state + joint action。

    Args:
        topology: `CommunicationTopology` 实例（提供 follower 数量等信息）。
        auto_entropy: 是否启用 entropy 系数自动调节（SAC 的 temperature）。
        use_amp: 是否启用 AMP 混合精度（仅 CUDA 环境有效）。

    Attributes:
        actor: `DecentralizedActor`，用于执行/策略更新。
        q1/q2: `CentralizedCritic`，双 Q 网络。
        buffer: `CTDEReplayBuffer`。
        value_net: 为了评估工具统一保留（SAC 不使用，恒为 None）。
    """

    def __init__(self, topology, auto_entropy: bool = True, use_amp: bool = True):
        self.topology = topology
        self.num_followers = topology.num_followers
        self.num_agents = topology.num_agents
        self.auto_entropy = bool(auto_entropy)

        self.use_amp = bool(use_amp and DEVICE.type == "cuda")
        self.scaler = None
        self._autocast = None
        if self.use_amp:
            from torch.amp import GradScaler
            from torch.amp import autocast

            self.scaler = GradScaler("cuda")
            self._autocast = lambda: autocast("cuda")
            print("🚀 AMP (混合精度训练) 已启用 - CTDE 架构")

        # 标准化接口：SAC 没有 value_net，但为了评估工具一致，保留该属性。
        self.value_net = None

        self.actor = DecentralizedActor(hidden_dim=HIDDEN_DIM).to(DEVICE)

        self.q1 = CentralizedCritic(global_state_dim=GLOBAL_STATE_DIM, hidden_dim=HIDDEN_DIM).to(DEVICE)
        self.q2 = CentralizedCritic(global_state_dim=GLOBAL_STATE_DIM, hidden_dim=HIDDEN_DIM).to(DEVICE)
        self.q1_target = CentralizedCritic(global_state_dim=GLOBAL_STATE_DIM, hidden_dim=HIDDEN_DIM).to(DEVICE)
        self.q2_target = CentralizedCritic(global_state_dim=GLOBAL_STATE_DIM, hidden_dim=HIDDEN_DIM).to(DEVICE)

        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        for p in self.q1_target.parameters():
            p.requires_grad = False
        for p in self.q2_target.parameters():
            p.requires_grad = False

        self.target_entropy = -float(ACTION_DIM * self.num_followers) * float(TARGET_ENTROPY_RATIO)
        self.log_alpha = torch.tensor(np.log(INIT_ALPHA), requires_grad=True, device=DEVICE)
        self.alpha = float(self.log_alpha.exp().item())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=LEARNING_RATE)
        self.q1_optimizer = optim.Adam(self.q1.parameters(), lr=LEARNING_RATE)
        self.q2_optimizer = optim.Adam(self.q2.parameters(), lr=LEARNING_RATE)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=ALPHA_LR)

        self.buffer = CTDEReplayBuffer(num_agents=NUM_AGENTS)

        self.last_losses = {"q1": 0, "q2": 0, "actor": 0, "alpha": INIT_ALPHA}
        self.update_count = 0

        print("📊 CTDE SAC Agent initialized:")
        print(f"   Actor input: Local state ({STATE_DIM})")
        print(f"   Critic input: Global state ({GLOBAL_STATE_DIM}) + Joint action ({NUM_FOLLOWERS * ACTION_DIM})")

    @torch.inference_mode()
    def select_action(self, local_states: torch.Tensor, deterministic: bool = False):
        """根据本地状态选择动作（面向 follower）。

        Args:
            local_states: 本地状态。
                - 单环境：shape=(num_agents, STATE_DIM)
                - 并行环境：shape=(E, num_agents, STATE_DIM)
                其中 index=0 为 leader，follower 为 1..N。
            deterministic: 是否使用确定性动作（评估/可视化常用 True）。

        Returns:
            follower 动作张量：
            - 单环境：shape=(num_followers, ACTION_DIM)
            - 并行环境：shape=(E, num_followers, ACTION_DIM)
        """
        is_batched = local_states.dim() == 3

        if is_batched:
            batch_size = int(local_states.shape[0])
            follower_states = local_states[:, 1:, :]
            flat_states = follower_states.reshape(-1, STATE_DIM)

            if self.use_amp:
                with self._autocast():
                    action, _ = self.actor(flat_states, deterministic=deterministic)
            else:
                action, _ = self.actor(flat_states, deterministic=deterministic)

            action = action.view(batch_size, self.num_followers, ACTION_DIM)
        else:
            follower_states = local_states[1:, :]
            if self.use_amp:
                with self._autocast():
                    action, _ = self.actor(follower_states, deterministic=deterministic)
            else:
                action, _ = self.actor(follower_states, deterministic=deterministic)

        return action.float()

    def store_transitions_batch(
        self,
        local_states: torch.Tensor,
        global_states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_local_states: torch.Tensor,
        next_global_states: torch.Tensor,
        dones: torch.Tensor,
    ):
        """向 replay buffer 追加一批 transition。

        Args:
            local_states: shape=(E, num_agents, STATE_DIM)。
            global_states: shape=(E, GLOBAL_STATE_DIM)。
            actions: shape=(E, num_followers, ACTION_DIM)。
            rewards: shape=(E,)。
            next_local_states: shape=(E, num_agents, STATE_DIM)。
            next_global_states: shape=(E, GLOBAL_STATE_DIM)。
            dones: shape=(E,)；True 表示该并行环境在该步终止（含时间截断）。
        """
        self.buffer.push_batch(local_states, global_states, actions, rewards, next_local_states, next_global_states, dones)

    def update(self, batch_size: int = BATCH_SIZE, gradient_steps: int = GRADIENT_STEPS):
        """执行一次或多次 SAC 更新。

        Args:
            batch_size: 每次从 replay buffer 采样的 batch 大小。
            gradient_steps: 连续更新的次数（每次都会重新采样）。

        Returns:
            最近一次更新的损失字典（用于日志/仪表盘）。若 buffer 数据不足返回空字典。
        """
        if not self.buffer.is_ready(batch_size):
            return {}

        total_q1_loss = 0.0
        total_q2_loss = 0.0
        total_actor_loss = 0.0

        total_q1_mean = 0.0
        total_q2_mean = 0.0
        total_target_q_mean = 0.0
        total_logp_joint = 0.0
        total_entropy_joint = 0.0
        total_alpha_loss = 0.0
        policy_updates = 0

        for _ in range(int(gradient_steps)):
            self.update_count += 1

            (local_states, global_states, actions, rewards, next_local_states, next_global_states, dones) = self.buffer.sample(batch_size)

            follower_states = local_states[:, 1:, :].reshape(-1, STATE_DIM)
            follower_next_states = next_local_states[:, 1:, :].reshape(-1, STATE_DIM)
            joint_actions = actions.view(batch_size, -1)

            with torch.no_grad():
                if self.use_amp:
                    with self._autocast():
                        next_actions, next_log_probs = self.actor(follower_next_states)
                else:
                    next_actions, next_log_probs = self.actor(follower_next_states)

                next_joint_actions = next_actions.view(batch_size, -1)
                next_log_probs_joint = next_log_probs.view(batch_size, self.num_followers, 1).sum(dim=1)

                q1_next = self.q1_target(next_global_states, next_joint_actions)
                q2_next = self.q2_target(next_global_states, next_joint_actions)
                q_next = torch.min(q1_next, q2_next)

                target_q = rewards.unsqueeze(1) + GAMMA * (1 - dones.unsqueeze(1)) * (q_next - self.alpha * next_log_probs_joint)
                target_q = target_q.float()

            self.q1_optimizer.zero_grad(set_to_none=True)
            self.q2_optimizer.zero_grad(set_to_none=True)

            if self.use_amp:
                with self._autocast():
                    q1_curr = self.q1(global_states, joint_actions)
                    q2_curr = self.q2(global_states, joint_actions)
                    q1_loss = F.mse_loss(q1_curr.float(), target_q)
                    q2_loss = F.mse_loss(q2_curr.float(), target_q)
                    critic_loss = q1_loss + q2_loss

                self.scaler.scale(critic_loss).backward()

                self.scaler.unscale_(self.q1_optimizer)
                self.scaler.unscale_(self.q2_optimizer)
                torch.nn.utils.clip_grad_norm_(self.q1.parameters(), 1.0)
                torch.nn.utils.clip_grad_norm_(self.q2.parameters(), 1.0)

                self.scaler.step(self.q1_optimizer)
                self.scaler.step(self.q2_optimizer)
            else:
                q1_curr = self.q1(global_states, joint_actions)
                q2_curr = self.q2(global_states, joint_actions)
                q1_loss = F.mse_loss(q1_curr, target_q)
                q2_loss = F.mse_loss(q2_curr, target_q)
                critic_loss = q1_loss + q2_loss

                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.q1.parameters(), 1.0)
                torch.nn.utils.clip_grad_norm_(self.q2.parameters(), 1.0)
                self.q1_optimizer.step()
                self.q2_optimizer.step()

            do_policy_update = self.update_count % max(1, int(POLICY_DELAY)) == 0
            actor_loss = torch.tensor(0.0, device=DEVICE)

            if do_policy_update:
                self.actor_optimizer.zero_grad(set_to_none=True)

                if self.use_amp:
                    with self._autocast():
                        new_actions, log_probs = self.actor(follower_states)
                        new_joint_actions = new_actions.view(batch_size, -1)

                        q1_new = self.q1(global_states, new_joint_actions)
                        q2_new = self.q2(global_states, new_joint_actions)
                        q_new = torch.min(q1_new, q2_new)

                        log_probs_joint = log_probs.view(batch_size, self.num_followers, 1).sum(dim=1)
                        actor_loss = (self.alpha * log_probs_joint - q_new).mean()

                    self.scaler.scale(actor_loss).backward()
                    self.scaler.unscale_(self.actor_optimizer)
                    torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
                    self.scaler.step(self.actor_optimizer)
                else:
                    new_actions, log_probs = self.actor(follower_states)
                    new_joint_actions = new_actions.view(batch_size, -1)

                    q1_new = self.q1(global_states, new_joint_actions)
                    q2_new = self.q2(global_states, new_joint_actions)
                    q_new = torch.min(q1_new, q2_new)

                    log_probs_joint = log_probs.view(batch_size, self.num_followers, 1).sum(dim=1)
                    actor_loss = (self.alpha * log_probs_joint - q_new).mean()

                    actor_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
                    self.actor_optimizer.step()

                if self.auto_entropy:
                    self.alpha_optimizer.zero_grad(set_to_none=True)

                    log_probs_joint_detached = log_probs.view(batch_size, self.num_followers, 1).sum(dim=1).detach()
                    mean_log_prob = log_probs_joint_detached.mean()

                    alpha_loss = -(self.log_alpha * (mean_log_prob + self.target_entropy))
                    alpha_loss.backward()
                    self.alpha_optimizer.step()
                    self.alpha = float(self.log_alpha.exp().item())

            if self.use_amp:
                self.scaler.update()

            if self.update_count % max(1, int(TARGET_UPDATE_INTERVAL)) == 0:
                tau_eff = 1.0 - (1.0 - TAU) ** max(1, int(TARGET_UPDATE_INTERVAL))
                self._soft_update(self.q1, self.q1_target, tau=tau_eff)
                self._soft_update(self.q2, self.q2_target, tau=tau_eff)

            total_q1_loss += float(q1_loss.item())
            total_q2_loss += float(q2_loss.item())
            total_actor_loss += float(actor_loss.item())

            total_q1_mean += float(q1_curr.detach().mean().item())
            total_q2_mean += float(q2_curr.detach().mean().item())
            total_target_q_mean += float(target_q.detach().mean().item())

            if do_policy_update:
                policy_updates += 1
                total_logp_joint += float(log_probs_joint.detach().mean().item())
                total_entropy_joint += float((-log_probs_joint).detach().mean().item())
                if self.auto_entropy:
                    total_alpha_loss += float(alpha_loss.detach().item())

        pol_denom = max(1, int(policy_updates))
        self.last_losses = {
            "q1": total_q1_loss / float(gradient_steps),
            "q2": total_q2_loss / float(gradient_steps),
            "actor": total_actor_loss / float(gradient_steps),
            "alpha": float(self.alpha),
            "q1_mean": total_q1_mean / float(gradient_steps),
            "q2_mean": total_q2_mean / float(gradient_steps),
            "target_q_mean": total_target_q_mean / float(gradient_steps),
            "logp_joint": total_logp_joint / float(pol_denom),
            "entropy_joint": total_entropy_joint / float(pol_denom),
            "alpha_loss": total_alpha_loss / float(pol_denom) if self.auto_entropy else float("nan"),
            "policy_updates": int(policy_updates),
        }

        return self.last_losses

    @torch.no_grad()
    def _soft_update(self, source: torch.nn.Module, target: torch.nn.Module, tau: float = TAU):
        for param, target_param in zip(source.parameters(), target.parameters()):
            target_param.data.lerp_(param.data, tau)

    def save(self, path: str):
        """保存模型参数到文件。

        Args:
            path: 保存路径。
        """
        parent = os.path.dirname(str(path))
        if parent:
            os.makedirs(parent, exist_ok=True)

        torch.save(
            {
                "actor": self.actor.state_dict(),
                "q1": self.q1.state_dict(),
                "q2": self.q2.state_dict(),
                "q1_target": self.q1_target.state_dict(),
                "q2_target": self.q2_target.state_dict(),
                "log_alpha": self.log_alpha,
                "update_count": self.update_count,
            },
            path,
        )
        print(f"✅ CTDE Model saved to {path}")

    def load(self, path: str):
        """从文件加载模型参数。

        Args:
            path: checkpoint 路径。
        """
        checkpoint = torch.load(path, map_location=DEVICE)
        self.actor.load_state_dict(checkpoint["actor"])
        self.q1.load_state_dict(checkpoint["q1"])
        self.q2.load_state_dict(checkpoint["q2"])
        self.q1_target.load_state_dict(checkpoint["q1_target"])
        self.q2_target.load_state_dict(checkpoint["q2_target"])

        if "log_alpha" in checkpoint:
            ckpt_log_alpha = checkpoint["log_alpha"]
            if isinstance(ckpt_log_alpha, torch.Tensor):
                ckpt_log_alpha = ckpt_log_alpha.to(device=DEVICE, dtype=self.log_alpha.dtype)
                self.log_alpha.data.copy_(ckpt_log_alpha)
            else:
                self.log_alpha.data.fill_(float(ckpt_log_alpha))
            self.alpha = float(self.log_alpha.exp().item())

        if "update_count" in checkpoint:
            self.update_count = int(checkpoint["update_count"])

        print(f"✅ CTDE Model loaded from {path}")


class _PPOBuffer:
    """On-policy rollout buffer（按时间步存储，支持并行环境）。"""

    def __init__(self, rollout_steps: int = PPO_ROLLOUT_STEPS):
        self.rollout_steps = int(rollout_steps)
        self.clear()

    def clear(self):
        self.follower_states = []
        self.global_states = []
        self.actions = []
        self.logp_joint = []
        self.values = []
        self.rewards = []
        self.dones = []

    def add(self, follower_states, global_states, actions, logp_joint, values, rewards, dones):
        self.follower_states.append(follower_states)
        self.global_states.append(global_states)
        self.actions.append(actions)
        self.logp_joint.append(logp_joint)
        self.values.append(values)
        self.rewards.append(rewards)
        self.dones.append(dones)

    def __len__(self):
        return len(self.rewards)

    def is_full(self):
        return len(self) >= self.rollout_steps


class CTDEMAPPOAgent:
    """CTDE-MAPPO 智能体。

    - Policy：对每个 follower 使用本地状态（factorized policy，参数共享）。
    - Value：集中式 V 网络，输入 global state。

    Args:
        topology: `CommunicationTopology` 实例。
        use_amp: 是否启用 AMP（当前实现默认关闭）。

    Notes:
        MAPPO 是 on-policy 算法，需要先通过 `store_rollout_step()` 收集 rollout，
        再调用 `update()` 进行多轮 PPO 优化。
    """

    def __init__(self, topology, use_amp: bool = False):
        self.topology = topology
        self.num_followers = topology.num_followers
        self.num_agents = topology.num_agents
        self.use_amp = False

        self.actor = DecentralizedActor(hidden_dim=HIDDEN_DIM).to(DEVICE)
        self.value_net = CentralizedValue(global_state_dim=GLOBAL_STATE_DIM, hidden_dim=HIDDEN_DIM).to(DEVICE)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=PPO_LR)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=PPO_LR)

        self.buffer = _PPOBuffer(rollout_steps=PPO_ROLLOUT_STEPS)
        self.last_losses = {}

        print("📊 CTDE MAPPO Agent initialized:")
        print(f"   Actor input: Local state ({STATE_DIM})")
        print(f"   Value input: Global state ({GLOBAL_STATE_DIM})")
        print(f"   Rollout steps: {PPO_ROLLOUT_STEPS} | Epochs: {PPO_EPOCHS} | Minibatch: {PPO_MINIBATCH_SIZE}")

    @torch.inference_mode()
    def select_action(self, local_states: torch.Tensor, deterministic: bool = False):
        if local_states.dim() == 2:
            actions, _, _ = self.act(local_states.unsqueeze(0), None, deterministic=deterministic)
            return actions[0]
        actions, _, _ = self.act(local_states, None, deterministic=deterministic)
        return actions

    @torch.inference_mode()
    def act(self, local_states: torch.Tensor, global_states: torch.Tensor | None = None, deterministic: bool = False):
        """采样动作并（可选）返回 value 估计。

        Args:
            local_states: shape=(E, num_agents, STATE_DIM)。
            global_states: shape=(E, GLOBAL_STATE_DIM)；若为 None 则不计算 value。
            deterministic: 是否使用确定性动作。

        Returns:
            actions: shape=(E, num_followers, ACTION_DIM)
            logp_joint: shape=(E, 1)
            values: shape=(E, 1)；当 global_states 为 None 时填充 NaN。
        """
        assert local_states.dim() == 3, "MAPPO 训练需要 batched local_states"
        E = int(local_states.shape[0])

        follower_states = local_states[:, 1:, :].reshape(-1, STATE_DIM)
        act_flat, logp_flat = self.actor(follower_states, deterministic=deterministic)
        actions = act_flat.view(E, self.num_followers, ACTION_DIM).float()

        if logp_flat is None:
            logp_joint = torch.zeros((E, 1), device=DEVICE)
        else:
            logp_joint = logp_flat.view(E, self.num_followers, 1).sum(dim=1)

        if global_states is None:
            values = torch.full((E, 1), float("nan"), device=DEVICE)
        else:
            values = self.value_net(global_states).float()

        return actions, logp_joint, values

    def store_rollout_step(self, local_states, global_states, actions, logp_joint, values, rewards, dones):
        follower_states = local_states[:, 1:, :].detach()
        self.buffer.add(
            follower_states.float(),
            global_states.detach().float(),
            actions.detach().float(),
            logp_joint.detach().float(),
            values.detach().float(),
            rewards.detach().float(),
            dones.detach(),
        )

    def _compute_gae(self, rewards, dones, values, last_value):
        T, E = rewards.shape
        advantages = torch.zeros(T, E, device=DEVICE)
        gae = torch.zeros(E, device=DEVICE)

        for t in reversed(range(T)):
            nonterminal = (~dones[t]).float()
            next_value = last_value if t == T - 1 else values[t + 1]
            delta = rewards[t] + GAMMA * next_value * nonterminal - values[t]
            gae = delta + GAMMA * PPO_GAE_LAMBDA * nonterminal * gae
            advantages[t] = gae

        returns = advantages + values
        return advantages, returns

    def update(self, next_global_states=None, next_dones=None):
        if len(self.buffer) == 0:
            return {}

        follower_states = torch.stack(self.buffer.follower_states, dim=0)  # (T,E,F,S)
        global_states = torch.stack(self.buffer.global_states, dim=0)  # (T,E,G)
        actions = torch.stack(self.buffer.actions, dim=0)  # (T,E,F,A)
        old_logp_joint = torch.stack(self.buffer.logp_joint, dim=0)  # (T,E,1)
        values = torch.stack(self.buffer.values, dim=0)  # (T,E,1)
        rewards = torch.stack(self.buffer.rewards, dim=0)  # (T,E)
        dones = torch.stack(self.buffer.dones, dim=0)  # (T,E)

        T, E = rewards.shape

        with torch.no_grad():
            if next_global_states is None:
                last_value = torch.zeros(E, device=DEVICE)
            else:
                lv = self.value_net(next_global_states).squeeze(-1)
                if next_dones is not None:
                    lv = lv * (~next_dones).float()
                last_value = lv

            adv, ret = self._compute_gae(rewards=rewards, dones=dones, values=values.squeeze(-1), last_value=last_value)

            adv_flat = adv.reshape(-1)
            adv = (adv - adv_flat.mean()) / (adv_flat.std(unbiased=False) + 1e-8)

        N = T * E
        follower_states_env = follower_states.reshape(N, self.num_followers, STATE_DIM)
        global_states_env = global_states.reshape(N, -1)
        actions_env = actions.reshape(N, self.num_followers, ACTION_DIM)
        old_logp_env = old_logp_joint.reshape(N, 1)
        adv_env = adv.reshape(N, 1)
        ret_env = ret.reshape(N, 1)

        mb_size = int(PPO_MINIBATCH_SIZE)
        clip_eps = float(PPO_CLIP_EPS)

        pol_losses = []
        v_losses = []
        entropies = []
        kls = []
        clipfracs = []

        for _ in range(int(PPO_EPOCHS)):
            perm = torch.randperm(N, device=DEVICE)
            for start in range(0, N, mb_size):
                idx = perm[start : start + mb_size]

                mb_gs = global_states_env[idx]
                mb_fs = follower_states_env[idx]
                mb_act = actions_env[idx]
                mb_old_logp = old_logp_env[idx]
                mb_adv = adv_env[idx]
                mb_ret = ret_env[idx]

                fs_flat = mb_fs.reshape(-1, STATE_DIM)
                act_flat = mb_act.reshape(-1, ACTION_DIM)

                new_logp_flat = self.actor.evaluate_actions(fs_flat, act_flat)
                new_logp_joint = new_logp_flat.view(-1, self.num_followers, 1).sum(dim=1)

                ratio = torch.exp(new_logp_joint - mb_old_logp)
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * mb_adv
                policy_loss = -(torch.min(surr1, surr2)).mean()

                entropy_joint = (-new_logp_joint).mean()
                approx_kl = (mb_old_logp - new_logp_joint).mean()
                clipfrac = (torch.abs(ratio - 1.0) > clip_eps).float().mean()

                actor_loss = policy_loss - float(PPO_ENTROPY_COEF) * entropy_joint

                self.actor_optimizer.zero_grad(set_to_none=True)
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), float(PPO_MAX_GRAD_NORM))
                self.actor_optimizer.step()

                v_pred = self.value_net(mb_gs)
                value_loss = F.mse_loss(v_pred, mb_ret)

                self.value_optimizer.zero_grad(set_to_none=True)
                (float(PPO_VALUE_COEF) * value_loss).backward()
                torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), float(PPO_MAX_GRAD_NORM))
                self.value_optimizer.step()

                pol_losses.append(float(policy_loss.detach().item()))
                v_losses.append(float(value_loss.detach().item()))
                entropies.append(float(entropy_joint.detach().item()))
                kls.append(float(approx_kl.detach().item()))
                clipfracs.append(float(clipfrac.detach().item()))

                if float(PPO_TARGET_KL) > 0 and float(approx_kl.detach().item()) > 1.5 * float(PPO_TARGET_KL):
                    break

        self.buffer.clear()

        self.last_losses = {
            "policy": float(np.mean(pol_losses)) if pol_losses else float("nan"),
            "value": float(np.mean(v_losses)) if v_losses else float("nan"),
            "entropy_joint": float(np.mean(entropies)) if entropies else float("nan"),
            "kl": float(np.mean(kls)) if kls else float("nan"),
            "clipfrac": float(np.mean(clipfracs)) if clipfracs else float("nan"),
        }
        return self.last_losses

    def save(self, path: str):
        """保存模型参数到文件。

        Args:
            path: 保存路径。
        """
        parent = os.path.dirname(str(path))
        if parent:
            os.makedirs(parent, exist_ok=True)

        torch.save({"actor": self.actor.state_dict(), "value": self.value_net.state_dict()}, path)
        print(f"✅ CTDE MAPPO Model saved to {path}")

    def load(self, path: str):
        """从文件加载模型参数。

        Args:
            path: checkpoint 路径。
        """
        checkpoint = torch.load(path, map_location=DEVICE)
        self.actor.load_state_dict(checkpoint["actor"])
        self.value_net.load_state_dict(checkpoint["value"])
        print(f"✅ CTDE MAPPO Model loaded from {path}")
