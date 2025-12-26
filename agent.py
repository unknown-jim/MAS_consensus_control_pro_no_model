"""
SAC 智能体
"""
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

from config import (
    DEVICE, STATE_DIM, HIDDEN_DIM, ACTION_DIM,
    LEARNING_RATE, ALPHA_LR, GAMMA, TAU, BATCH_SIZE,
    INIT_ALPHA
)
from buffer import ReplayBuffer
from networks import GaussianActor, SoftQNetwork


class SACAgent:
    """Soft Actor-Critic 智能体"""
    
    def __init__(self, topology, auto_entropy=True):
        """
        Args:
            topology: DirectedSpanningTreeTopology 对象
            auto_entropy: 是否自动调整熵系数
        """
        self.topology = topology
        self.num_followers = topology.num_followers
        self.num_agents = topology.num_agents
        self.auto_entropy = auto_entropy
        
        # 网络初始化
        self.actor = GaussianActor(STATE_DIM, HIDDEN_DIM).to(DEVICE)
        self.q1 = SoftQNetwork(STATE_DIM, HIDDEN_DIM).to(DEVICE)
        self.q2 = SoftQNetwork(STATE_DIM, HIDDEN_DIM).to(DEVICE)
        self.q1_target = SoftQNetwork(STATE_DIM, HIDDEN_DIM).to(DEVICE)
        self.q2_target = SoftQNetwork(STATE_DIM, HIDDEN_DIM).to(DEVICE)
        
        # 复制目标网络参数
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())
        
        # 冻结目标网络
        for param in self.q1_target.parameters():
            param.requires_grad = False
        for param in self.q2_target.parameters():
            param.requires_grad = False
        
        # 熵系数 (动态计算目标熵)
        self.target_entropy = -float(ACTION_DIM)  # 动态设置，而非硬编码
        self.log_alpha = torch.tensor(np.log(INIT_ALPHA), requires_grad=True, device=DEVICE)
        self.alpha = self.log_alpha.exp().item()
        
        # 优化器
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=LEARNING_RATE)
        self.q1_optimizer = optim.Adam(self.q1.parameters(), lr=LEARNING_RATE)
        self.q2_optimizer = optim.Adam(self.q2.parameters(), lr=LEARNING_RATE)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=ALPHA_LR)
        
        # 经验回放
        self.buffer = ReplayBuffer()
        
        # 角色 ID
        self.role_ids = torch.zeros(self.num_agents, dtype=torch.long, device=DEVICE)
        self.role_ids[1:] = 1
        
        # 损失记录
        self.last_losses = {'q1': 0, 'q2': 0, 'actor': 0, 'alpha': INIT_ALPHA}
        
        # 更新计数器
        self.update_count = 0
    
    def select_action(self, state, deterministic=False):
        """选择动作
        
        Args:
            state: 状态张量
            deterministic: 是否使用确定性策略
            
        Returns:
            action: 动作张量 (num_followers, action_dim)
        """
        with torch.no_grad():
            action, _, _ = self.actor(
                state,
                self.topology.edge_index,
                self.role_ids,
                deterministic=deterministic
            )
        return action
    
    def store_transition(self, state, action, reward, next_state, done):
        """存储经验
        
        Args:
            state, action, next_state: 张量
            reward: float
            done: bool or float
        """
        self.buffer.push(state.clone(), action.clone(), reward, next_state.clone(), float(done))
    
    def update(self, batch_size=BATCH_SIZE):
        """更新网络
        
        Args:
            batch_size: 批量大小
            
        Returns:
            losses: 损失字典
        """
        if not self.buffer.is_ready(batch_size):
            return {}
        
        self.update_count += 1
        
        # 采样
        states, actions, rewards, next_states, dones = self.buffer.sample(batch_size)
        
        batch_size_actual = states.shape[0]
        
        # 展平批量数据
        flat_states = states.view(-1, STATE_DIM)
        flat_next_states = next_states.view(-1, STATE_DIM)
        flat_actions = actions.view(-1, ACTION_DIM)
        
        # 构建批量边索引和角色 ID
        batch_edge_index = self._batch_edge_index(batch_size_actual)
        batch_role_ids = self.role_ids.repeat(batch_size_actual)
        
        # ========== 更新 Critic ==========
        with torch.no_grad():
            next_actions, next_log_probs, _ = self.actor(
                flat_next_states, batch_edge_index, batch_role_ids
            )
            
            q1_next = self.q1_target(flat_next_states, batch_edge_index, batch_role_ids, next_actions)
            q2_next = self.q2_target(flat_next_states, batch_edge_index, batch_role_ids, next_actions)
            q_next = torch.min(q1_next, q2_next)
            
            # 聚合每个样本的 Q 值
            q_next = q_next.view(batch_size_actual, self.num_followers).mean(dim=1, keepdim=True)
            next_log_probs = next_log_probs.view(batch_size_actual, self.num_followers).mean(dim=1, keepdim=True)
            
            # TD 目标
            target_q = rewards.unsqueeze(1) + GAMMA * (1 - dones.unsqueeze(1)) * (q_next - self.alpha * next_log_probs)
        
        # 当前 Q 值
        q1_curr = self.q1(flat_states, batch_edge_index, batch_role_ids, flat_actions)
        q2_curr = self.q2(flat_states, batch_edge_index, batch_role_ids, flat_actions)
        
        q1_curr = q1_curr.view(batch_size_actual, self.num_followers).mean(dim=1, keepdim=True)
        q2_curr = q2_curr.view(batch_size_actual, self.num_followers).mean(dim=1, keepdim=True)
        
        # Critic 损失
        q1_loss = F.mse_loss(q1_curr, target_q)
        q2_loss = F.mse_loss(q2_curr, target_q)
        
        # 更新 Q1
        self.q1_optimizer.zero_grad()
        q1_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q1.parameters(), 1.0)
        self.q1_optimizer.step()
        
        # 更新 Q2
        self.q2_optimizer.zero_grad()
        q2_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q2.parameters(), 1.0)
        self.q2_optimizer.step()
        
        # ========== 更新 Actor ==========
        new_actions, log_probs, _ = self.actor(flat_states, batch_edge_index, batch_role_ids)
        
        q1_new = self.q1(flat_states, batch_edge_index, batch_role_ids, new_actions)
        q2_new = self.q2(flat_states, batch_edge_index, batch_role_ids, new_actions)
        q_new = torch.min(q1_new, q2_new)
        
        actor_loss = (self.alpha * log_probs - q_new).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optimizer.step()
        
        # ========== 更新 Alpha (熵系数) ==========
        if self.auto_entropy:
            alpha_loss = -(self.log_alpha * (log_probs.detach() + self.target_entropy)).mean()
            
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            
            self.alpha = self.log_alpha.exp().item()
        else:
            alpha_loss = torch.tensor(0.0)
        
        # ========== 软更新目标网络 ==========
        self._soft_update(self.q1, self.q1_target)
        self._soft_update(self.q2, self.q2_target)
        
        # 记录损失
        self.last_losses = {
            'q1': q1_loss.item(),
            'q2': q2_loss.item(),
            'actor': actor_loss.item(),
            'alpha': self.alpha
        }
        
        return self.last_losses
    
    def _batch_edge_index(self, batch_size):
        """为批量数据构建边索引
        
        Args:
            batch_size: 批量大小
            
        Returns:
            batch_edge_index: 拼接后的边索引
        """
        num_nodes = self.num_agents
        edge_indices = []
        for i in range(batch_size):
            edge_indices.append(self.topology.edge_index + i * num_nodes)
        return torch.cat(edge_indices, dim=1)
    
    def _soft_update(self, source, target):
        """软更新目标网络
        
        Args:
            source: 源网络
            target: 目标网络
        """
        for param, target_param in zip(source.parameters(), target.parameters()):
            target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)
    
    def save(self, path):
        """保存模型"""
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
        """加载模型"""
        checkpoint = torch.load(path, map_location=DEVICE, weights_only=False)
        self.actor.load_state_dict(checkpoint['actor'])
        self.q1.load_state_dict(checkpoint['q1'])
        self.q2.load_state_dict(checkpoint['q2'])
        self.q1_target.load_state_dict(checkpoint['q1_target'])
        self.q2_target.load_state_dict(checkpoint['q2_target'])
        if 'log_alpha' in checkpoint:
            self.log_alpha = checkpoint['log_alpha']
            self.alpha = self.log_alpha.exp().item()
        if 'update_count' in checkpoint:
            self.update_count = checkpoint['update_count']
        print(f"✅ Model loaded from {path}")