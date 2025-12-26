"""
SAC 智能体 - 优化版 (修复 PyG 兼容性)
"""
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

from config import (
    DEVICE, STATE_DIM, HIDDEN_DIM, ACTION_DIM, NUM_AGENTS,
    LEARNING_RATE, ALPHA_LR, GAMMA, TAU, BATCH_SIZE,
    INIT_ALPHA, GRADIENT_STEPS
)
from buffer import OptimizedReplayBuffer
from networks import GaussianActor, SoftQNetwork


class SACAgent:
    """Soft Actor-Critic 智能体 (PyG 兼容版)"""
    
    def __init__(self, topology, auto_entropy=True, use_amp=True):
        self.topology = topology
        self.num_followers = topology.num_followers
        self.num_agents = topology.num_agents
        self.auto_entropy = auto_entropy
        self.use_amp = use_amp and torch.cuda.is_available()
        
        # 网络初始化
        self.actor = GaussianActor(STATE_DIM, HIDDEN_DIM).to(DEVICE)
        self.q1 = SoftQNetwork(STATE_DIM, HIDDEN_DIM).to(DEVICE)
        self.q2 = SoftQNetwork(STATE_DIM, HIDDEN_DIM).to(DEVICE)
        self.q1_target = SoftQNetwork(STATE_DIM, HIDDEN_DIM).to(DEVICE)
        self.q2_target = SoftQNetwork(STATE_DIM, HIDDEN_DIM).to(DEVICE)
        
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())
        
        # 冻结目标网络
        for param in self.q1_target.parameters():
            param.requires_grad = False
        for param in self.q2_target.parameters():
            param.requires_grad = False
        
        # 混合精度训练
        if self.use_amp:
            from torch.cuda.amp import GradScaler
            self.scaler = GradScaler()
            print("🚀 AMP (混合精度训练) 已启用")
        else:
            self.scaler = None
        
        # 熵系数
        self.target_entropy = -float(ACTION_DIM)
        self.log_alpha = torch.tensor(np.log(INIT_ALPHA), requires_grad=True, device=DEVICE)
        self.alpha = self.log_alpha.exp().item()
        
        # 优化器
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=LEARNING_RATE)
        self.q1_optimizer = optim.Adam(self.q1.parameters(), lr=LEARNING_RATE)
        self.q2_optimizer = optim.Adam(self.q2.parameters(), lr=LEARNING_RATE)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=ALPHA_LR)
        
        # 经验回放
        self.buffer = OptimizedReplayBuffer(num_agents=NUM_AGENTS)
        
        # 预计算
        self.role_ids = torch.zeros(self.num_agents, dtype=torch.long, device=DEVICE)
        self.role_ids[1:] = 1
        
        # 缓存
        self._edge_index_cache = {}
        self._role_ids_cache = {}
        
        self.last_losses = {'q1': 0, 'q2': 0, 'actor': 0, 'alpha': INIT_ALPHA}
        self.update_count = 0
    
    def _get_batch_graph_data(self, batch_size):
        """获取批量图数据 (缓存)"""
        if batch_size not in self._edge_index_cache:
            num_nodes = self.num_agents
            edge_indices = [self.topology.edge_index + i * num_nodes for i in range(batch_size)]
            self._edge_index_cache[batch_size] = torch.cat(edge_indices, dim=1)
            self._role_ids_cache[batch_size] = self.role_ids.repeat(batch_size)
        return self._edge_index_cache[batch_size], self._role_ids_cache[batch_size]
    
    @torch.no_grad()
    def select_action(self, state, deterministic=False):
        """选择动作"""
        is_batched = state.dim() == 3
        
        if is_batched:
            batch_size = state.shape[0]
            flat_state = state.view(-1, STATE_DIM)
            batch_edge_index, batch_role_ids = self._get_batch_graph_data(batch_size)
            
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    action, _, _ = self.actor(
                        flat_state, batch_edge_index, batch_role_ids, deterministic=deterministic
                    )
            else:
                action, _, _ = self.actor(
                    flat_state, batch_edge_index, batch_role_ids, deterministic=deterministic
                )
            action = action.view(batch_size, self.num_followers, ACTION_DIM)
        else:
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    action, _, _ = self.actor(
                        state, self.topology.edge_index, self.role_ids, deterministic=deterministic
                    )
            else:
                action, _, _ = self.actor(
                    state, self.topology.edge_index, self.role_ids, deterministic=deterministic
                )
        
        return action.float()
    
    def store_transitions_batch(self, states, actions, rewards, next_states, dones):
        """批量存储"""
        self.buffer.push_batch(states, actions, rewards, next_states, dones)
    
    def update(self, batch_size=BATCH_SIZE, gradient_steps=GRADIENT_STEPS):
        """更新网络"""
        if not self.buffer.is_ready(batch_size):
            return {}
        
        total_q1_loss = 0
        total_q2_loss = 0
        total_actor_loss = 0
        
        for _ in range(gradient_steps):
            self.update_count += 1
            
            states, actions, rewards, next_states, dones = self.buffer.sample(batch_size)
            
            flat_states = states.view(-1, STATE_DIM)
            flat_next_states = next_states.view(-1, STATE_DIM)
            flat_actions = actions.view(-1, ACTION_DIM)
            
            batch_edge_index, batch_role_ids = self._get_batch_graph_data(batch_size)
            
            # ========== Critic 更新 ==========
            with torch.no_grad():
                if self.use_amp:
                    with torch.cuda.amp.autocast():
                        next_actions, next_log_probs, _ = self.actor(
                            flat_next_states, batch_edge_index, batch_role_ids
                        )
                        q1_next = self.q1_target(flat_next_states, batch_edge_index, batch_role_ids, next_actions)
                        q2_next = self.q2_target(flat_next_states, batch_edge_index, batch_role_ids, next_actions)
                else:
                    next_actions, next_log_probs, _ = self.actor(
                        flat_next_states, batch_edge_index, batch_role_ids
                    )
                    q1_next = self.q1_target(flat_next_states, batch_edge_index, batch_role_ids, next_actions)
                    q2_next = self.q2_target(flat_next_states, batch_edge_index, batch_role_ids, next_actions)
                
                q_next = torch.min(q1_next, q2_next)
                q_next = q_next.view(batch_size, self.num_followers).mean(dim=1, keepdim=True)
                next_log_probs = next_log_probs.view(batch_size, self.num_followers).mean(dim=1, keepdim=True)
                
                target_q = rewards.unsqueeze(1) + GAMMA * (1 - dones.unsqueeze(1)) * (q_next - self.alpha * next_log_probs)
                target_q = target_q.float()
            
            # Q1 更新
            self.q1_optimizer.zero_grad(set_to_none=True)
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    q1_curr = self.q1(flat_states, batch_edge_index, batch_role_ids, flat_actions)
                    q1_curr = q1_curr.view(batch_size, self.num_followers).mean(dim=1, keepdim=True)
                    q1_loss = F.mse_loss(q1_curr.float(), target_q)
                self.scaler.scale(q1_loss).backward()
                self.scaler.unscale_(self.q1_optimizer)
                torch.nn.utils.clip_grad_norm_(self.q1.parameters(), 1.0)
                self.scaler.step(self.q1_optimizer)
            else:
                q1_curr = self.q1(flat_states, batch_edge_index, batch_role_ids, flat_actions)
                q1_curr = q1_curr.view(batch_size, self.num_followers).mean(dim=1, keepdim=True)
                q1_loss = F.mse_loss(q1_curr, target_q)
                q1_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.q1.parameters(), 1.0)
                self.q1_optimizer.step()
            
            # Q2 更新
            self.q2_optimizer.zero_grad(set_to_none=True)
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    q2_curr = self.q2(flat_states, batch_edge_index, batch_role_ids, flat_actions)
                    q2_curr = q2_curr.view(batch_size, self.num_followers).mean(dim=1, keepdim=True)
                    q2_loss = F.mse_loss(q2_curr.float(), target_q)
                self.scaler.scale(q2_loss).backward()
                self.scaler.unscale_(self.q2_optimizer)
                torch.nn.utils.clip_grad_norm_(self.q2.parameters(), 1.0)
                self.scaler.step(self.q2_optimizer)
            else:
                q2_curr = self.q2(flat_states, batch_edge_index, batch_role_ids, flat_actions)
                q2_curr = q2_curr.view(batch_size, self.num_followers).mean(dim=1, keepdim=True)
                q2_loss = F.mse_loss(q2_curr, target_q)
                q2_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.q2.parameters(), 1.0)
                self.q2_optimizer.step()
            
            # ========== Actor 更新 ==========
            self.actor_optimizer.zero_grad(set_to_none=True)
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    new_actions, log_probs, _ = self.actor(flat_states, batch_edge_index, batch_role_ids)
                    q1_new = self.q1(flat_states, batch_edge_index, batch_role_ids, new_actions)
                    q2_new = self.q2(flat_states, batch_edge_index, batch_role_ids, new_actions)
                    q_new = torch.min(q1_new, q2_new)
                    actor_loss = (self.alpha * log_probs - q_new).mean()
                self.scaler.scale(actor_loss).backward()
                self.scaler.unscale_(self.actor_optimizer)
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
                self.scaler.step(self.actor_optimizer)
            else:
                new_actions, log_probs, _ = self.actor(flat_states, batch_edge_index, batch_role_ids)
                q1_new = self.q1(flat_states, batch_edge_index, batch_role_ids, new_actions)
                q2_new = self.q2(flat_states, batch_edge_index, batch_role_ids, new_actions)
                q_new = torch.min(q1_new, q2_new)
                actor_loss = (self.alpha * log_probs - q_new).mean()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
                self.actor_optimizer.step()
            
            # ========== Alpha 更新 ==========
            if self.auto_entropy:
                self.alpha_optimizer.zero_grad(set_to_none=True)
                alpha_loss = -(self.log_alpha * (log_probs.detach() + self.target_entropy)).mean()
                alpha_loss.backward()
                self.alpha_optimizer.step()
                self.alpha = self.log_alpha.exp().item()
            
            # ========== 软更新 ==========
            self._soft_update(self.q1, self.q1_target)
            self._soft_update(self.q2, self.q2_target)
            
            # 更新 scaler
            if self.use_amp:
                self.scaler.update()
            
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
        print(f"✅ Model saved to {path}")
    
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
        print(f"✅ Model loaded from {path}")