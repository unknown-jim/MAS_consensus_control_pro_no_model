"""
领导者-跟随者多智能体系统环境 - 完全向量化并行版
"""
import torch
import math

from config import (
    DEVICE, STATE_DIM, DT, COMM_PENALTY,
    LEADER_AMPLITUDE, LEADER_OMEGA, LEADER_PHASE,
    POS_LIMIT, VEL_LIMIT,
    REWARD_MIN, REWARD_MAX, USE_SOFT_REWARD_SCALING
)


class BatchedLeaderFollowerEnv:
    """完全向量化的批量环境"""
    
    def __init__(self, topology, num_envs=64):
        self.topology = topology
        self.num_envs = num_envs
        self.num_agents = topology.num_agents
        self.num_followers = topology.num_followers
        self.leader_id = topology.leader_id
        
        self.leader_amplitude = LEADER_AMPLITUDE
        self.leader_omega = LEADER_OMEGA
        self.leader_phase = LEADER_PHASE
        
        self.pos_limit = POS_LIMIT
        self.vel_limit = VEL_LIMIT
        self.reward_min = REWARD_MIN
        self.reward_max = REWARD_MAX
        self.use_soft_scaling = USE_SOFT_REWARD_SCALING
        
        self.role_ids = torch.zeros(self.num_agents, dtype=torch.long, device=DEVICE)
        self.role_ids[1:] = 1
        
        self._precompute_neighbor_info()
        
        # 预分配状态张量
        self.positions = torch.zeros(num_envs, self.num_agents, device=DEVICE)
        self.velocities = torch.zeros(num_envs, self.num_agents, device=DEVICE)
        self.last_broadcast_pos = torch.zeros(num_envs, self.num_agents, device=DEVICE)
        self.last_broadcast_vel = torch.zeros(num_envs, self.num_agents, device=DEVICE)
        self.t = torch.zeros(num_envs, device=DEVICE)
        
        self.reset()
    
    def _precompute_neighbor_info(self):
        """预计算邻居聚合矩阵"""
        self.adj_matrix = torch.zeros(self.num_agents, self.num_agents, device=DEVICE)
        edge_index = self.topology.edge_index
        
        for i in range(edge_index.shape[1]):
            src, dst = edge_index[0, i].item(), edge_index[1, i].item()
            self.adj_matrix[dst, src] = 1.0
        
        in_degree = self.adj_matrix.sum(dim=1, keepdim=True).clamp(min=1.0)
        self.norm_adj_matrix = self.adj_matrix / in_degree
    
    def _leader_state_batch(self, t):
        """批量计算领导者状态"""
        pos = self.leader_amplitude * torch.sin(self.leader_omega * t + self.leader_phase)
        vel = self.leader_amplitude * self.leader_omega * torch.cos(self.leader_omega * t + self.leader_phase)
        return pos, vel
    
    def reset(self, env_ids=None):
        """重置环境"""
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=DEVICE)
        
        num_reset = len(env_ids) if isinstance(env_ids, torch.Tensor) else self.num_envs
        
        self.t[env_ids] = 0.0
        
        leader_pos, leader_vel = self._leader_state_batch(self.t[env_ids])
        
        self.positions[env_ids, 0] = leader_pos
        self.velocities[env_ids, 0] = leader_vel
        self.positions[env_ids, 1:] = torch.randn(num_reset, self.num_followers, device=DEVICE) * 1.5
        self.velocities[env_ids, 1:] = torch.randn(num_reset, self.num_followers, device=DEVICE) * 0.5
        
        self.last_broadcast_pos[env_ids] = self.positions[env_ids].clone()
        self.last_broadcast_vel[env_ids] = self.velocities[env_ids].clone()
        
        return self._get_state()
    
    def _get_state(self):
        """构建观测状态"""
        state = torch.zeros(self.num_envs, self.num_agents, STATE_DIM, device=DEVICE)
        
        state[:, :, 0] = self.positions
        state[:, :, 1] = self.velocities
        state[:, :, 2] = torch.matmul(self.last_broadcast_pos, self.norm_adj_matrix.T)
        state[:, :, 3] = torch.matmul(self.last_broadcast_vel, self.norm_adj_matrix.T)
        
        return state
    
    def _scale_reward_batch(self, reward):
        """批量奖励缩放"""
        if self.use_soft_scaling:
            mid = (self.reward_max + self.reward_min) / 2
            scale = (self.reward_max - self.reward_min) / 2
            normalized = (reward - mid) / (scale + 1e-8)
            return mid + scale * torch.tanh(normalized)
        else:
            return torch.clamp(reward, self.reward_min, self.reward_max)
    
    def step(self, action):
        """批量执行一步"""
        self.t += DT
        
        leader_pos, leader_vel = self._leader_state_batch(self.t)
        self.positions[:, 0] = leader_pos
        self.velocities[:, 0] = leader_vel
        
        u = action[:, :, 0]
        threshold = action[:, :, 1]
        
        follower_pos = self.positions[:, 1:]
        follower_vel = self.velocities[:, 1:]
        
        nonlinear_term = torch.sin(follower_pos) - 0.5 * follower_vel
        acc = u + nonlinear_term
        
        new_vel = torch.clamp(follower_vel + acc * DT, -self.vel_limit, self.vel_limit)
        new_pos = torch.clamp(follower_pos + new_vel * DT, -self.pos_limit, self.pos_limit)
        
        self.positions[:, 1:] = new_pos
        self.velocities[:, 1:] = new_vel
        
        trigger_error = torch.abs(new_pos - self.last_broadcast_pos[:, 1:])
        is_triggered = trigger_error > threshold
        
        self.last_broadcast_pos[:, 1:] = torch.where(
            is_triggered, self.positions[:, 1:], self.last_broadcast_pos[:, 1:]
        )
        self.last_broadcast_vel[:, 1:] = torch.where(
            is_triggered, self.velocities[:, 1:], self.last_broadcast_vel[:, 1:]
        )
        self.last_broadcast_pos[:, 0] = self.positions[:, 0]
        self.last_broadcast_vel[:, 0] = self.velocities[:, 0]
        
        pos_error = (self.positions[:, 1:] - self.positions[:, 0:1]) ** 2
        vel_error = (self.velocities[:, 1:] - self.velocities[:, 0:1]) ** 2
        tracking_error = (pos_error + vel_error).mean(dim=1)
        
        comm_rate = is_triggered.float().mean(dim=1)
        raw_reward = -tracking_error - comm_rate * COMM_PENALTY
        rewards = self._scale_reward_batch(raw_reward)
        
        dones = torch.zeros(self.num_envs, dtype=torch.bool, device=DEVICE)
        
        infos = {
            'tracking_error': tracking_error,
            'comm_rate': comm_rate,
            'leader_pos': self.positions[:, 0],
            'leader_vel': self.velocities[:, 0],
            'avg_follower_pos': self.positions[:, 1:].mean(dim=1),
        }
        
        return self._get_state(), rewards, dones, infos


class LeaderFollowerMASEnv:
    """单环境版本 (用于评估)"""
    
    def __init__(self, topology):
        self.batched_env = BatchedLeaderFollowerEnv(topology, num_envs=1)
        self.topology = topology
        self.num_agents = topology.num_agents
        self.num_followers = topology.num_followers
        self.role_ids = self.batched_env.role_ids
    
    @property
    def positions(self):
        return self.batched_env.positions[0]
    
    @property
    def velocities(self):
        return self.batched_env.velocities[0]
    
    @property
    def t(self):
        return self.batched_env.t[0].item()
    
    def reset(self):
        state = self.batched_env.reset()
        return state[0]
    
    def step(self, action):
        action_batched = action.unsqueeze(0)
        states, rewards, dones, infos = self.batched_env.step(action_batched)
        info = {k: v[0].item() if isinstance(v, torch.Tensor) else v for k, v in infos.items()}
        return states[0], rewards[0].item(), dones[0].item(), info