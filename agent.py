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
    INIT_ALPHA, GRADIENT_STEPS, NUM_FOLLOWERS, GLOBAL_STATE_DIM
)
from buffer import CTDEReplayBuffer
from networks import GaussianActor, SoftQNetwork


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
            from torch.cuda.amp import GradScaler
            self.scaler = GradScaler()
            print("🚀 AMP (混合精度训练) 已启用 - CTDE 架构")
        else:
            self.scaler = None
        
        # 温度参数（每个智能体共享）
        # 对于多智能体联合动作空间，target_entropy 应考虑所有智能体
        self.target_entropy = -float(ACTION_DIM * self.num_followers)
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
    
    @torch.no_grad()
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
                with torch.cuda.amp.autocast():
                    action, _, _ = self.actor(flat_states, deterministic=deterministic)
            else:
                action, _, _ = self.actor(flat_states, deterministic=deterministic)
            
            action = action.view(batch_size, self.num_followers, ACTION_DIM)
        else:
            follower_states = local_states[1:, :]
            
            if self.use_amp:
                with torch.cuda.amp.autocast():
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
        
        total_q1_loss = 0
        total_q2_loss = 0
        total_actor_loss = 0
        
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
                    with torch.cuda.amp.autocast():
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
            
            # Q1 更新
            self.q1_optimizer.zero_grad(set_to_none=True)
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    q1_curr = self.q1(global_states, joint_actions)
                    q1_loss = F.mse_loss(q1_curr.float(), target_q)
                self.scaler.scale(q1_loss).backward()
                self.scaler.unscale_(self.q1_optimizer)
                torch.nn.utils.clip_grad_norm_(self.q1.parameters(), 1.0)
                self.scaler.step(self.q1_optimizer)
            else:
                q1_curr = self.q1(global_states, joint_actions)
                q1_loss = F.mse_loss(q1_curr, target_q)
                q1_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.q1.parameters(), 1.0)
                self.q1_optimizer.step()
            
            # Q2 更新
            self.q2_optimizer.zero_grad(set_to_none=True)
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    q2_curr = self.q2(global_states, joint_actions)
                    q2_loss = F.mse_loss(q2_curr.float(), target_q)
                self.scaler.scale(q2_loss).backward()
                self.scaler.unscale_(self.q2_optimizer)
                torch.nn.utils.clip_grad_norm_(self.q2.parameters(), 1.0)
                self.scaler.step(self.q2_optimizer)
            else:
                q2_curr = self.q2(global_states, joint_actions)
                q2_loss = F.mse_loss(q2_curr, target_q)
                q2_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.q2.parameters(), 1.0)
                self.q2_optimizer.step()
            
            # ========== Actor 更新 ==========
            self.actor_optimizer.zero_grad(set_to_none=True)
            if self.use_amp:
                with torch.cuda.amp.autocast():
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
            
            # ========== Alpha 更新 ==========
            if self.auto_entropy:
                self.alpha_optimizer.zero_grad(set_to_none=True)

                # 🔧 alpha 更新也应基于“联合动作”的 log-prob（对跟随者求和后再对 batch 求均值）
                log_probs_joint_detached = log_probs.view(batch_size, self.num_followers, 1).sum(dim=1).detach()
                mean_log_prob = log_probs_joint_detached.mean()

                alpha_loss = -(self.log_alpha * (mean_log_prob + self.target_entropy))
                alpha_loss.backward()
                self.alpha_optimizer.step()
                self.alpha = self.log_alpha.exp().item()
            
            # ========== AMP Scaler 更新（在所有 step 之后）==========
            if self.use_amp:
                self.scaler.update()
            
            # ========== 软更新 ==========
            self._soft_update(self.q1, self.q1_target)
            self._soft_update(self.q2, self.q2_target)
            
            total_q1_loss += q1_loss.item()
            total_q2_loss += q2_loss.item()
            total_actor_loss += actor_loss.item()
        
        self.last_losses = {
            'q1': total_q1_loss / gradient_steps,
            'q2': total_q2_loss / gradient_steps,
            'actor': total_actor_loss / gradient_steps,
            'alpha': self.alpha
        }
        
        return self.last_losses
    
    @torch.no_grad()
    def _soft_update(self, source, target):
        for param, target_param in zip(source.parameters(), target.parameters()):
            target_param.data.lerp_(param.data, TAU)
    
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
        if 'log_alpha' in checkpoint:
            self.log_alpha = checkpoint['log_alpha']
            self.alpha = self.log_alpha.exp().item()
        print(f"✅ CTDE Model loaded from {path}")


# 保留旧名称以兼容
SACAgent = CTDESACAgent