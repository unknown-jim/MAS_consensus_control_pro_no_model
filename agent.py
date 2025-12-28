"""
SAC 智能体 - CTDE 架构版本

关键区别：
- Actor：分散式，只用本地观测
- Critic：集中式，用全局状态 + 联合动作
"""
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

from config import (
    DEVICE, STATE_DIM, HIDDEN_DIM, ACTION_DIM, NUM_AGENTS,
    LEARNING_RATE, ALPHA_LR, GAMMA, TAU, BATCH_SIZE,
    INIT_ALPHA, GRADIENT_STEPS, NUM_FOLLOWERS, GLOBAL_STATE_DIM,
    POLICY_DELAY, TARGET_UPDATE_INTERVAL,
    TARGET_ENTROPY_RATIO,

    # MAPPO/PPO
    PPO_LR, PPO_CLIP_EPS, PPO_EPOCHS, PPO_ROLLOUT_STEPS, PPO_MINIBATCH_SIZE,
    PPO_GAE_LAMBDA, PPO_VALUE_COEF, PPO_ENTROPY_COEF, PPO_MAX_GRAD_NORM, PPO_TARGET_KL,
)
from buffer import CTDEReplayBuffer
from networks import GaussianActor, SoftQNetwork, ValueNetwork


class CTDESACAgent:
    """
    CTDE SAC 智能体
    
    Centralized Training:
    - Critic 使用全局状态 + 所有智能体的联合动作
    
    Decentralized Execution:
    - Actor 只使用单个智能体的本地观测
    """
    
    def __init__(self, topology, auto_entropy=True, use_amp=True):
        self.topology = topology
        self.num_followers = topology.num_followers
        self.num_agents = topology.num_agents
        self.auto_entropy = auto_entropy
        self.use_amp = use_amp and torch.cuda.is_available()
        
        # 分散式 Actor（每个智能体共享参数）
        self.actor = GaussianActor(STATE_DIM, HIDDEN_DIM).to(DEVICE)
        
        # 集中式 Critic（使用全局状态）
        self.q1 = SoftQNetwork(GLOBAL_STATE_DIM, HIDDEN_DIM).to(DEVICE)
        self.q2 = SoftQNetwork(GLOBAL_STATE_DIM, HIDDEN_DIM).to(DEVICE)
        self.q1_target = SoftQNetwork(GLOBAL_STATE_DIM, HIDDEN_DIM).to(DEVICE)
        self.q2_target = SoftQNetwork(GLOBAL_STATE_DIM, HIDDEN_DIM).to(DEVICE)
        
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())
        
        for param in self.q1_target.parameters():
            param.requires_grad = False
        for param in self.q2_target.parameters():
            param.requires_grad = False
        
        if self.use_amp:
            # 兼容 PyTorch 2.x：优先用 torch.amp，避免 FutureWarning
            try:
                from torch.amp import GradScaler
                self.scaler = GradScaler('cuda')
                self._autocast = lambda: torch.amp.autocast('cuda')
            except Exception:
                from torch.cuda.amp import GradScaler
                self.scaler = GradScaler()
                self._autocast = lambda: torch.cuda.amp.autocast()
            print("🚀 AMP (混合精度训练) 已启用 - CTDE 架构")
        else:
            self.scaler = None
            self._autocast = None
        
        # 温度参数（每个智能体共享）
        # 对于多智能体联合动作空间，target_entropy 应考虑所有智能体
        # 允许用 TARGET_ENTROPY_RATIO 缩放熵目标：比例越小 -> 探索越弱，学习更稳
        self.target_entropy = -float(ACTION_DIM * self.num_followers) * float(TARGET_ENTROPY_RATIO)
        self.log_alpha = torch.tensor(np.log(INIT_ALPHA), requires_grad=True, device=DEVICE)
        self.alpha = self.log_alpha.exp().item()
        
        # 优化器
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=LEARNING_RATE)
        self.q1_optimizer = optim.Adam(self.q1.parameters(), lr=LEARNING_RATE)
        self.q2_optimizer = optim.Adam(self.q2.parameters(), lr=LEARNING_RATE)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=ALPHA_LR)
        
        # CTDE 缓冲区
        self.buffer = CTDEReplayBuffer(num_agents=NUM_AGENTS)
        
        self.last_losses = {'q1': 0, 'q2': 0, 'actor': 0, 'alpha': INIT_ALPHA}
        self.update_count = 0
        
        print(f"📊 CTDE Agent initialized:")
        print(f"   Actor input: Local state ({STATE_DIM})")
        print(f"   Critic input: Global state ({GLOBAL_STATE_DIM}) + Joint action ({NUM_FOLLOWERS * ACTION_DIM})")
    
    @torch.inference_mode()
    def select_action(self, local_states, deterministic=False):
        """
        分散式动作选择（只用本地观测）
        
        Args:
            local_states: (batch, num_agents, state_dim) 或 (num_agents, state_dim)
        """
        is_batched = local_states.dim() == 3
        
        if is_batched:
            batch_size = local_states.shape[0]
            # 只处理跟随者
            follower_states = local_states[:, 1:, :]
            flat_states = follower_states.reshape(-1, STATE_DIM)
            
            if self.use_amp:
                with self._autocast():
                    action, _, _ = self.actor(flat_states, deterministic=deterministic)
            else:
                action, _, _ = self.actor(flat_states, deterministic=deterministic)
            
            action = action.view(batch_size, self.num_followers, ACTION_DIM)
        else:
            follower_states = local_states[1:, :]
            
            if self.use_amp:
                with self._autocast():
                    action, _, _ = self.actor(follower_states, deterministic=deterministic)
            else:
                action, _, _ = self.actor(follower_states, deterministic=deterministic)
        
        return action.float()
    
    def store_transitions_batch(self, local_states, global_states, actions, rewards, 
                                next_local_states, next_global_states, dones):
        """批量存储（包含全局状态）"""
        self.buffer.push_batch(local_states, global_states, actions, rewards, 
                               next_local_states, next_global_states, dones)
    
    def update(self, batch_size=BATCH_SIZE, gradient_steps=GRADIENT_STEPS):
        """更新网络（CTDE 方式）"""
        if not self.buffer.is_ready(batch_size):
            return {}
        
        total_q1_loss = 0.0
        total_q2_loss = 0.0
        total_actor_loss = 0.0

        # 诊断统计（用于判断是否 Q 发散/α 过大/熵塌陷等）
        total_q1_mean = 0.0
        total_q2_mean = 0.0
        total_target_q_mean = 0.0
        total_logp_joint = 0.0
        total_entropy_joint = 0.0
        total_alpha_loss = 0.0
        policy_updates = 0
        
        for _ in range(gradient_steps):
            self.update_count += 1
            
            # 采样
            (local_states, global_states, actions, rewards, 
             next_local_states, next_global_states, dones) = self.buffer.sample(batch_size)
            
            # 准备数据
            follower_states = local_states[:, 1:, :].reshape(-1, STATE_DIM)
            follower_next_states = next_local_states[:, 1:, :].reshape(-1, STATE_DIM)
            joint_actions = actions.view(batch_size, -1)  # (batch, num_followers * action_dim)
            
            # ========== Critic 更新（使用全局状态）==========
            with torch.no_grad():
                # 使用 Actor 生成下一步动作
                if self.use_amp:
                    with self._autocast():
                        next_actions, next_log_probs, _ = self.actor(follower_next_states)
                else:
                    next_actions, next_log_probs, _ = self.actor(follower_next_states)
                
                # 重塑为联合动作
                next_joint_actions = next_actions.view(batch_size, -1)

                # 🔧 多智能体熵项：应对联合策略的 log-prob 做“求和”而不是均值
                # next_log_probs: (batch*num_followers, 1) -> (batch, num_followers, 1) -> (batch, 1)
                next_log_probs_joint = next_log_probs.view(batch_size, self.num_followers, 1).sum(dim=1)
                
                # 使用全局状态计算 Q 值
                q1_next = self.q1_target(next_global_states, next_joint_actions)
                q2_next = self.q2_target(next_global_states, next_joint_actions)
                q_next = torch.min(q1_next, q2_next)
                
                target_q = rewards.unsqueeze(1) + GAMMA * (1 - dones.unsqueeze(1)) * (q_next - self.alpha * next_log_probs_joint)
                target_q = target_q.float()
            
            # ========== Critic 更新（合并反传，减少 Python/优化器开销） ==========
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

                # 分别 unscale + clip（两套参数各自裁剪）
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

            # ========== Actor/Alpha 更新（Policy Delay：不必每个 step 都做反传） ==========
            do_policy_update = (self.update_count % max(1, POLICY_DELAY) == 0)
            actor_loss = torch.tensor(0.0, device=DEVICE)

            if do_policy_update:
                self.actor_optimizer.zero_grad(set_to_none=True)
                if self.use_amp:
                    with self._autocast():
                        new_actions, log_probs, _ = self.actor(follower_states)
                        new_joint_actions = new_actions.view(batch_size, -1)

                        # 使用全局状态评估动作
                        q1_new = self.q1(global_states, new_joint_actions)
                        q2_new = self.q2(global_states, new_joint_actions)
                        q_new = torch.min(q1_new, q2_new)

                        # 🔧 联合策略熵项：对跟随者维度求和（与 target_entropy 定义一致）
                        log_probs_joint = log_probs.view(batch_size, self.num_followers, 1).sum(dim=1)
                        actor_loss = (self.alpha * log_probs_joint - q_new).mean()

                    self.scaler.scale(actor_loss).backward()
                    self.scaler.unscale_(self.actor_optimizer)
                    torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
                    self.scaler.step(self.actor_optimizer)
                else:
                    new_actions, log_probs, _ = self.actor(follower_states)
                    new_joint_actions = new_actions.view(batch_size, -1)

                    q1_new = self.q1(global_states, new_joint_actions)
                    q2_new = self.q2(global_states, new_joint_actions)
                    q_new = torch.min(q1_new, q2_new)

                    log_probs_joint = log_probs.view(batch_size, self.num_followers, 1).sum(dim=1)
                    actor_loss = (self.alpha * log_probs_joint - q_new).mean()

                    actor_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
                    self.actor_optimizer.step()

                # ========== Alpha 更新（同样延后到 policy update） ==========
                if self.auto_entropy:
                    self.alpha_optimizer.zero_grad(set_to_none=True)

                    log_probs_joint_detached = log_probs.view(batch_size, self.num_followers, 1).sum(dim=1).detach()
                    mean_log_prob = log_probs_joint_detached.mean()

                    alpha_loss = -(self.log_alpha * (mean_log_prob + self.target_entropy))
                    alpha_loss.backward()
                    self.alpha_optimizer.step()
                    self.alpha = self.log_alpha.exp().item()
            
            # ========== AMP Scaler 更新（在所有 step 之后）==========
            if self.use_amp:
                self.scaler.update()

            # ========== Target 软更新（降低频率，减少参数拷贝开销） ==========
            if self.update_count % max(1, TARGET_UPDATE_INTERVAL) == 0:
                # 等效 tau：interval 次连续 lerp 的合成
                tau_eff = 1.0 - (1.0 - TAU) ** max(1, TARGET_UPDATE_INTERVAL)
                self._soft_update(self.q1, self.q1_target, tau=tau_eff)
                self._soft_update(self.q2, self.q2_target, tau=tau_eff)
            
            total_q1_loss += float(q1_loss.item())
            total_q2_loss += float(q2_loss.item())
            total_actor_loss += float(actor_loss.item())

            # critic 侧的数值尺度（target/Q 的均值常能快速暴露发散）
            total_q1_mean += float(q1_curr.detach().mean().item())
            total_q2_mean += float(q2_curr.detach().mean().item())
            total_target_q_mean += float(target_q.detach().mean().item())

            if do_policy_update:
                policy_updates += 1
                total_logp_joint += float(log_probs_joint.detach().mean().item())
                total_entropy_joint += float((-log_probs_joint).detach().mean().item())
                if self.auto_entropy:
                    total_alpha_loss += float(alpha_loss.detach().item())
        
        # policy 相关统计可能不是每个 gradient step 都更新
        pol_denom = max(1, int(policy_updates))

        self.last_losses = {
            'q1': total_q1_loss / float(gradient_steps),
            'q2': total_q2_loss / float(gradient_steps),
            'actor': total_actor_loss / float(gradient_steps),
            'alpha': float(self.alpha),

            # 诊断项（可用于 Dashboard/日志）
            'q1_mean': total_q1_mean / float(gradient_steps),
            'q2_mean': total_q2_mean / float(gradient_steps),
            'target_q_mean': total_target_q_mean / float(gradient_steps),
            'logp_joint': total_logp_joint / float(pol_denom),
            'entropy_joint': total_entropy_joint / float(pol_denom),
            'alpha_loss': total_alpha_loss / float(pol_denom) if self.auto_entropy else float('nan'),
            'policy_updates': int(policy_updates),
        }
        
        return self.last_losses
    
    @torch.no_grad()
    def _soft_update(self, source, target, tau: float = TAU):
        for param, target_param in zip(source.parameters(), target.parameters()):
            target_param.data.lerp_(param.data, tau)
    
    def save(self, path):
        torch.save({
            'actor': self.actor.state_dict(),
            'q1': self.q1.state_dict(),
            'q2': self.q2.state_dict(),
            'q1_target': self.q1_target.state_dict(),
            'q2_target': self.q2_target.state_dict(),
            'log_alpha': self.log_alpha,
            'update_count': self.update_count,
        }, path)
        print(f"✅ CTDE Model saved to {path}")
    
    def load(self, path):
        checkpoint = torch.load(path, map_location=DEVICE, weights_only=False)
        self.actor.load_state_dict(checkpoint['actor'])
        self.q1.load_state_dict(checkpoint['q1'])
        self.q2.load_state_dict(checkpoint['q2'])
        self.q1_target.load_state_dict(checkpoint['q1_target'])
        self.q2_target.load_state_dict(checkpoint['q2_target'])

        # 🔧 不要用“重新赋值”的方式替换 self.log_alpha；否则 alpha_optimizer 仍指向旧参数，α 更新会失效
        if 'log_alpha' in checkpoint:
            ckpt_log_alpha = checkpoint['log_alpha']
            if isinstance(ckpt_log_alpha, torch.Tensor):
                ckpt_log_alpha = ckpt_log_alpha.to(device=DEVICE, dtype=self.log_alpha.dtype)
                self.log_alpha.data.copy_(ckpt_log_alpha)
            else:
                self.log_alpha.data.fill_(float(ckpt_log_alpha))
            self.alpha = self.log_alpha.exp().item()

        if 'update_count' in checkpoint:
            self.update_count = int(checkpoint['update_count'])

        print(f"✅ CTDE Model loaded from {path}")




class _PPOBuffer:
    """On-policy rollout buffer（按时间步存储，支持并行环境）。"""

    def __init__(self, rollout_steps: int = PPO_ROLLOUT_STEPS):
        self.rollout_steps = int(rollout_steps)
        self.clear()

    def clear(self):
        self.follower_states = []     # list[(E, F, STATE_DIM)]
        self.global_states = []       # list[(E, G)]
        self.actions = []             # list[(E, F, ACTION_DIM)]
        self.logp_joint = []          # list[(E, 1)]
        self.values = []              # list[(E, 1)]
        self.rewards = []             # list[(E,)]
        self.dones = []               # list[(E,)] bool

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
    """CTDE-MAPPO：centralized value + decentralized shared policy（factorized followers）。"""

    def __init__(self, topology, use_amp: bool = False):
        self.topology = topology
        self.num_followers = topology.num_followers
        self.num_agents = topology.num_agents
        self.use_amp = False  # PPO 这里先不走 AMP，减少数值/调试复杂度

        self.actor = GaussianActor(STATE_DIM, HIDDEN_DIM).to(DEVICE)
        self.value_net = ValueNetwork(GLOBAL_STATE_DIM, HIDDEN_DIM).to(DEVICE)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=PPO_LR)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=PPO_LR)

        self.buffer = _PPOBuffer(rollout_steps=PPO_ROLLOUT_STEPS)

        self.last_losses = {}

        print("📊 CTDE MAPPO Agent initialized:")
        print(f"   Actor input: Local state ({STATE_DIM})")
        print(f"   Value input: Global state ({GLOBAL_STATE_DIM})")
        print(f"   Rollout steps: {PPO_ROLLOUT_STEPS} | Epochs: {PPO_EPOCHS} | Minibatch: {PPO_MINIBATCH_SIZE}")

    @torch.inference_mode()
    def select_action(self, local_states, deterministic=False):
        """与 SAC Agent 对齐的接口：只返回动作（用于评估/可视化）。

        支持：
        - batched: (E, A, S) -> 返回 (E, F, A)
        - single:  (A, S)    -> 返回 (F, A)
        """
        if local_states.dim() == 2:
            actions, _, _ = self.act(local_states.unsqueeze(0), None, deterministic=deterministic)
            return actions[0]
        actions, _, _ = self.act(local_states, None, deterministic=deterministic)
        return actions

    @torch.inference_mode()
    def act(self, local_states, global_states=None, deterministic=False):
        """采样动作，并返回 joint log-prob 和 value（用于 PPO rollout 收集）。

        Args:
            local_states: (E, A, STATE_DIM)
            global_states: (E, G) or None

        Returns:
            actions: (E, F, ACTION_DIM)
            logp_joint: (E, 1)
            values: (E, 1)（若 global_states=None 则为 NaN）
        """
        assert local_states.dim() == 3, "MAPPO 训练需要 batched local_states"
        E = local_states.shape[0]

        follower_states = local_states[:, 1:, :].reshape(-1, STATE_DIM)
        act_flat, logp_flat, _ = self.actor(follower_states, deterministic=deterministic)
        actions = act_flat.view(E, self.num_followers, ACTION_DIM).float()

        # factorized policy -> joint log-prob = sum(logp_i)
        # 注意：actor 在 deterministic=True 时会返回 logp=None（因为没有采样动作）。
        # 对评估/可视化而言我们只需要动作，所以这里给一个占位 0，避免 None 传播。
        if logp_flat is None:
            logp_joint = torch.zeros((E, 1), device=DEVICE)
        else:
            logp_joint = logp_flat.view(E, self.num_followers, 1).sum(dim=1)

        if global_states is None:
            values = torch.full((E, 1), float('nan'), device=DEVICE)
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
        """GAE(lambda)。

        Args:
            rewards: (T, E)
            dones: (T, E) bool
            values: (T, E)
            last_value: (E,)
        Returns:
            advantages: (T, E)
            returns: (T, E)
        """
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
        """用 buffer 中的 on-policy rollout 做 PPO 更新。"""
        if len(self.buffer) == 0:
            return {}

        # stack: list[T] -> (T, ...)
        follower_states = torch.stack(self.buffer.follower_states, dim=0)  # (T,E,F,S)
        global_states = torch.stack(self.buffer.global_states, dim=0)      # (T,E,G)
        actions = torch.stack(self.buffer.actions, dim=0)                  # (T,E,F,A)
        old_logp_joint = torch.stack(self.buffer.logp_joint, dim=0)        # (T,E,1)
        values = torch.stack(self.buffer.values, dim=0)                    # (T,E,1)
        rewards = torch.stack(self.buffer.rewards, dim=0)                  # (T,E)
        dones = torch.stack(self.buffer.dones, dim=0)                      # (T,E)

        T, E = rewards.shape

        with torch.no_grad():
            if next_global_states is None:
                last_value = torch.zeros(E, device=DEVICE)
            else:
                lv = self.value_net(next_global_states).squeeze(-1)
                if next_dones is not None:
                    lv = lv * (~next_dones).float()
                last_value = lv

            adv, ret = self._compute_gae(
                rewards=rewards,
                dones=dones,
                values=values.squeeze(-1),
                last_value=last_value,
            )

            # advantage normalization（PPO 常用）
            adv_flat = adv.reshape(-1)
            adv = (adv - adv_flat.mean()) / (adv_flat.std(unbiased=False) + 1e-8)

        # flatten time/env 维
        N = T * E
        follower_states_env = follower_states.reshape(N, self.num_followers, STATE_DIM)
        global_states_env = global_states.reshape(N, -1)
        actions_env = actions.reshape(N, self.num_followers, ACTION_DIM)
        old_logp_env = old_logp_joint.reshape(N, 1)
        adv_env = adv.reshape(N, 1)
        ret_env = ret.reshape(N, 1)

        mb_size = int(PPO_MINIBATCH_SIZE)
        clip_eps = float(PPO_CLIP_EPS)

        # logging accumulators
        pol_losses = []
        v_losses = []
        entropies = []
        kls = []
        clipfracs = []

        for _ in range(int(PPO_EPOCHS)):
            perm = torch.randperm(N, device=DEVICE)
            for start in range(0, N, mb_size):
                idx = perm[start:start + mb_size]

                mb_gs = global_states_env[idx]
                mb_fs = follower_states_env[idx]            # (B,F,S)
                mb_act = actions_env[idx]                   # (B,F,A)
                mb_old_logp = old_logp_env[idx]
                mb_adv = adv_env[idx]
                mb_ret = ret_env[idx]

                # ========== policy loss ==========
                fs_flat = mb_fs.reshape(-1, STATE_DIM)
                act_flat = mb_act.reshape(-1, ACTION_DIM)

                new_logp_flat = self.actor.evaluate_actions(fs_flat, act_flat)  # (B*F,1)
                new_logp_joint = new_logp_flat.view(-1, self.num_followers, 1).sum(dim=1)

                ratio = torch.exp(new_logp_joint - mb_old_logp)
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * mb_adv
                policy_loss = -(torch.min(surr1, surr2)).mean()

                # entropy 近似：用 -logp 作为 proxy（与 SAC 的 joint entropy 统计一致）
                entropy_joint = (-new_logp_joint).mean()

                # approx KL
                approx_kl = (mb_old_logp - new_logp_joint).mean()
                clipfrac = (torch.abs(ratio - 1.0) > clip_eps).float().mean()

                actor_loss = policy_loss - float(PPO_ENTROPY_COEF) * entropy_joint

                self.actor_optimizer.zero_grad(set_to_none=True)
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), float(PPO_MAX_GRAD_NORM))
                self.actor_optimizer.step()

                # ========== value loss ==========
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

                # optional early stop by KL
                if float(PPO_TARGET_KL) > 0 and float(approx_kl.detach().item()) > 1.5 * float(PPO_TARGET_KL):
                    break

        self.buffer.clear()

        self.last_losses = {
            'policy': float(np.mean(pol_losses)) if pol_losses else float('nan'),
            'value': float(np.mean(v_losses)) if v_losses else float('nan'),
            'entropy_joint': float(np.mean(entropies)) if entropies else float('nan'),
            'kl': float(np.mean(kls)) if kls else float('nan'),
            'clipfrac': float(np.mean(clipfracs)) if clipfracs else float('nan'),
        }
        return self.last_losses

    def save(self, path):
        torch.save({
            'actor': self.actor.state_dict(),
            'value': self.value_net.state_dict(),
        }, path)
        print(f"✅ CTDE MAPPO Model saved to {path}")

    def load(self, path):
        checkpoint = torch.load(path, map_location=DEVICE, weights_only=False)
        self.actor.load_state_dict(checkpoint['actor'])
        self.value_net.load_state_dict(checkpoint['value'])
        print(f"✅ CTDE MAPPO Model loaded from {path}")


# 保留旧名称以兼容
SACAgent = CTDESACAgent
MAPPOAgent = CTDEMAPPOAgent
