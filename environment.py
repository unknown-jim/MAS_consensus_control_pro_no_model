"""
领导者-跟随者多智能体系统环境
"""
import torch
import numpy as np

from config import (
    DEVICE, STATE_DIM, DT, COMM_PENALTY,
    LEADER_AMPLITUDE, LEADER_OMEGA, LEADER_PHASE,
    POS_LIMIT, VEL_LIMIT,
    REWARD_MIN, REWARD_MAX, USE_SOFT_REWARD_SCALING
)


class LeaderFollowerMASEnv:
    r"""
    领导者-跟随者多智能体系统环境
    
    领导者动力学 (非线性正弦轨迹):
        pos_L(t) = A * sin(omega * t + phi)
        vel_L(t) = A * omega * cos(omega * t + phi)
    
    跟随者动力学 (二阶非线性):
        x_ddot_i = u_i + sin(x_i) - 0.5 * x_dot_i
    """
    
    def __init__(self, topology):
        """
        Args:
            topology: DirectedSpanningTreeTopology 对象
        """
        self.topology = topology
        self.num_agents = topology.num_agents
        self.num_followers = topology.num_followers
        self.leader_id = topology.leader_id
        
        # 领导者轨迹参数
        self.leader_amplitude = LEADER_AMPLITUDE
        self.leader_omega = LEADER_OMEGA
        self.leader_phase = LEADER_PHASE
        
        # 状态边界
        self.pos_limit = POS_LIMIT
        self.vel_limit = VEL_LIMIT
        
        # 奖励配置
        self.reward_min = REWARD_MIN
        self.reward_max = REWARD_MAX
        self.use_soft_scaling = USE_SOFT_REWARD_SCALING
        
        # 角色标识
        self.role_ids = torch.zeros(self.num_agents, dtype=torch.long, device=DEVICE)
        self.role_ids[1:] = 1  # 跟随者标记为 1
        
        self.reset()
    
    def _leader_state(self, t):
        """计算领导者在时间 t 的状态"""
        pos = self.leader_amplitude * np.sin(self.leader_omega * t + self.leader_phase)
        vel = self.leader_amplitude * self.leader_omega * np.cos(self.leader_omega * t + self.leader_phase)
        return pos, vel
    
    def reset(self):
        """重置环境"""
        self.t = 0.0
        
        # 领导者初始状态
        leader_pos, leader_vel = self._leader_state(0)
        
        # 跟随者随机初始化
        follower_pos = torch.randn(self.num_followers, device=DEVICE) * 1.5
        follower_vel = torch.randn(self.num_followers, device=DEVICE) * 0.5
        
        # 初始化状态
        self.positions = torch.zeros(self.num_agents, device=DEVICE)
        self.velocities = torch.zeros(self.num_agents, device=DEVICE)
        
        self.positions[0] = leader_pos
        self.velocities[0] = leader_vel
        self.positions[1:] = follower_pos
        self.velocities[1:] = follower_vel
        
        # 事件触发机制：上次广播的状态
        self.last_broadcast_pos = self.positions.clone()
        self.last_broadcast_vel = self.velocities.clone()
        
        return self._get_state()
    
    def _get_state(self):
        """构建观测状态"""
        state = torch.zeros(self.num_agents, STATE_DIM, device=DEVICE)
        
        # 领导者状态
        state[0, 0] = self.positions[0]
        state[0, 1] = self.velocities[0]
        
        # 跟随者状态
        for i in range(1, self.num_agents):
            state[i, 0] = self.positions[i]
            state[i, 1] = self.velocities[i]
            
            # 邻居聚合信息
            neighbors = self.topology.get_neighbors(i)
            if neighbors:
                neighbor_pos = torch.stack([self.last_broadcast_pos[n] for n in neighbors])
                neighbor_vel = torch.stack([self.last_broadcast_vel[n] for n in neighbors])
                state[i, 2] = neighbor_pos.mean()
                state[i, 3] = neighbor_vel.mean()
        
        return state
    
    def _scale_reward(self, reward):
        """奖励缩放"""
        if self.use_soft_scaling:
            # 软缩放：使用 tanh 映射到有界区间
            # 将奖励映射到 [reward_min, reward_max]
            mid = (self.reward_max + self.reward_min) / 2
            scale = (self.reward_max - self.reward_min) / 2
            normalized = (reward - mid) / (scale + 1e-8)
            return mid + scale * torch.tanh(torch.tensor(normalized)).item()
        else:
            # 硬截断
            return torch.clamp(reward, self.reward_min, self.reward_max)
    
    def step(self, action):
        """执行一步
        
        Args:
            action: shape (num_followers, 2), [控制输入 u, 触发阈值 threshold]
            
        Returns:
            state, reward, done, info
        """
        self.t += DT
        
        # 更新领导者状态
        leader_pos, leader_vel = self._leader_state(self.t)
        self.positions[0] = leader_pos
        self.velocities[0] = leader_vel
        
        # 解析动作
        u = action[:, 0]  # 控制输入
        threshold = action[:, 1]  # 事件触发阈值
        
        # 跟随者动力学
        follower_pos = self.positions[1:]
        follower_vel = self.velocities[1:]
        
        # 非线性项: sin(x) - 0.5 * x_dot
        nonlinear_term = torch.sin(follower_pos) - 0.5 * follower_vel
        acc = u + nonlinear_term
        
        # 欧拉积分
        new_vel = follower_vel + acc * DT
        new_pos = follower_pos + new_vel * DT
        
        # 边界限制
        new_pos = torch.clamp(new_pos, -self.pos_limit, self.pos_limit)
        new_vel = torch.clamp(new_vel, -self.vel_limit, self.vel_limit)
        
        self.positions[1:] = new_pos
        self.velocities[1:] = new_vel
        
        # 事件触发判断
        trigger_error = torch.abs(new_pos - self.last_broadcast_pos[1:])
        is_triggered = (trigger_error > threshold).float()
        
        # 更新广播状态
        for i in range(self.num_followers):
            if is_triggered[i] > 0.5:
                self.last_broadcast_pos[i + 1] = self.positions[i + 1]
                self.last_broadcast_vel[i + 1] = self.velocities[i + 1]
        
        # 领导者状态始终广播
        self.last_broadcast_pos[0] = self.positions[0]
        self.last_broadcast_vel[0] = self.velocities[0]
        
        # 计算奖励
        tracking_error = torch.mean(
            (self.positions[1:] - self.positions[0])**2 + 
            (self.velocities[1:] - self.velocities[0])**2
        )
        comm_rate = is_triggered.mean()
        raw_reward = -tracking_error - comm_rate * COMM_PENALTY
        reward = self._scale_reward(raw_reward)
        
        # 信息字典
        info = {
            'tracking_error': tracking_error.item(),
            'comm_rate': comm_rate.item(),
            'leader_pos': self.positions[0].item(),
            'leader_vel': self.velocities[0].item(),
            'avg_follower_pos': self.positions[1:].mean().item(),
            'raw_reward': raw_reward.item() if isinstance(raw_reward, torch.Tensor) else raw_reward
        }
        
        return self._get_state(), reward, False, info