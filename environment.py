"""
领导者-跟随者多智能体系统环境 - CTDE 版本（随机初始化）

新增功能：
- 每个 episode 随机化领导者动力学参数（振幅、角频率、相位）
- 每个 episode 随机化跟随者初始状态分布
- 每个 episode 随机化通信拓扑（保证连通性）
"""
import torch

from config import (
    DEVICE, DT,
    LOCAL_OBS_DIM, NEIGHBOR_OBS_DIM, MAX_NEIGHBORS, STATE_DIM,
    SELF_ROLE_DIM, NEIGHBOR_ROLE_DIM, NEIGHBOR_FEAT_DIM,
    COMM_PENALTY, THRESHOLD_MIN, THRESHOLD_MAX,
    TRACKING_PENALTY_SCALE, TRACKING_PENALTY_MAX, COMM_WEIGHT_DECAY,
    IMPROVEMENT_SCALE, IMPROVEMENT_CLIP,
    LEADER_AMPLITUDE, LEADER_OMEGA, LEADER_PHASE,
    POS_LIMIT, VEL_LIMIT,
    REWARD_MIN, REWARD_MAX, USE_SOFT_REWARD_SCALING,
    TH_SCALE, V_SCALE, NUM_AGENTS, GLOBAL_STATE_DIM,
    # 随机化参数
    RANDOMIZE_LEADER, RANDOMIZE_FOLLOWER, RANDOMIZE_TOPOLOGY,
    LEADER_AMPLITUDE_RANGE, LEADER_OMEGA_RANGE, LEADER_PHASE_RANGE,
    FOLLOWER_INIT_POS_STD_RANGE, FOLLOWER_INIT_VEL_STD_RANGE,
    LEADER_TRAJECTORY_TYPES
)


class BatchedModelFreeEnv:
    """
    无模型批量环境 - CTDE 版本（随机初始化 + 多轨迹类型 + 拓扑随机化）
    
    新增功能：
    - get_global_state(): 返回全局状态用于集中式 Critic 训练
    - 每个 episode 随机化领导者动力学和跟随者初始状态
    - 支持多种领导者轨迹类型：sine, cosine, mixed, chirp
    - 每个 episode 随机化通信拓扑（保证连通性）
    """
    
    def __init__(self, topology, num_envs=64):
        self.topology = topology
        self.num_envs = num_envs
        self.num_agents = topology.num_agents
        self.num_followers = topology.num_followers
        self.leader_id = topology.leader_id
        
        # 领导者动力学参数（基准值，会在 reset 时随机化）
        self.leader_amplitude_base = LEADER_AMPLITUDE
        self.leader_omega_base = LEADER_OMEGA
        self.leader_phase_base = LEADER_PHASE
        
        # 🔧 为每个环境存储独立的领导者动力学参数
        self.leader_amplitude = torch.full((num_envs,), LEADER_AMPLITUDE, device=DEVICE)
        self.leader_omega = torch.full((num_envs,), LEADER_OMEGA, device=DEVICE)
        self.leader_phase = torch.full((num_envs,), LEADER_PHASE, device=DEVICE)
        
        # 🔧 轨迹类型支持
        self.trajectory_types = LEADER_TRAJECTORY_TYPES
        self.type_to_id = {t: i for i, t in enumerate(self.trajectory_types)}
        self.id_to_type = {i: t for i, t in enumerate(self.trajectory_types)}
        self.trajectory_type_ids = torch.zeros(num_envs, dtype=torch.long, device=DEVICE)
        
        # 随机化开关
        self.randomize_leader = RANDOMIZE_LEADER
        self.randomize_follower = RANDOMIZE_FOLLOWER
        self.randomize_topology = RANDOMIZE_TOPOLOGY
        
        # 领导者随机化范围
        self.amplitude_range = LEADER_AMPLITUDE_RANGE
        self.omega_range = LEADER_OMEGA_RANGE
        self.phase_range = LEADER_PHASE_RANGE
        
        # 跟随者随机化范围
        self.pos_std_range = FOLLOWER_INIT_POS_STD_RANGE
        self.vel_std_range = FOLLOWER_INIT_VEL_STD_RANGE
        
        self.pos_limit = POS_LIMIT
        self.vel_limit = VEL_LIMIT
        self.reward_min = REWARD_MIN
        self.reward_max = REWARD_MAX
        self.use_soft_scaling = USE_SOFT_REWARD_SCALING
        
        # 奖励参数
        self.comm_penalty_base = COMM_PENALTY
        self.threshold_min = THRESHOLD_MIN
        self.threshold_max = THRESHOLD_MAX
        self.tracking_penalty_scale = TRACKING_PENALTY_SCALE
        self.tracking_penalty_max = TRACKING_PENALTY_MAX
        self.comm_weight_decay = COMM_WEIGHT_DECAY
        self.improvement_scale = IMPROVEMENT_SCALE
        self.improvement_clip = IMPROVEMENT_CLIP
        
        self.th_scale = TH_SCALE
        self.v_scale = V_SCALE
        
        # 预计算邻居索引
        self._precompute_neighbor_indices()
        
        # 预分配状态张量
        self.positions = torch.zeros(num_envs, self.num_agents, device=DEVICE)
        self.velocities = torch.zeros(num_envs, self.num_agents, device=DEVICE)
        
        # 广播状态
        self.last_broadcast_pos = torch.zeros(num_envs, self.num_agents, device=DEVICE)
        self.last_broadcast_vel = torch.zeros(num_envs, self.num_agents, device=DEVICE)
        
        self.t = torch.zeros(num_envs, device=DEVICE)
        
        # 为每个环境单独存储 prev_error
        self._prev_error = torch.zeros(num_envs, device=DEVICE)
        self._prev_error_valid = torch.zeros(num_envs, dtype=torch.bool, device=DEVICE)
        
        # 预分配状态缓存
        self._state_buffer = torch.zeros(num_envs, self.num_agents, STATE_DIM, device=DEVICE)
        
        # CTDE：预分配全局状态缓存
        self._global_state_buffer = torch.zeros(num_envs, GLOBAL_STATE_DIM, device=DEVICE)
        
        self.reset()
    
    def _precompute_neighbor_indices(self, verbose=True):
        """预计算每个智能体的邻居索引"""
        self._neighbor_indices_list = []
        self._neighbor_counts = torch.zeros(self.num_agents, dtype=torch.long, device=DEVICE)
        
        for i in range(self.num_agents):
            can_receive_mask = self.topology.adj_matrix[i, :] > 0
            indices = torch.where(can_receive_mask)[0]
            
            if len(indices) > MAX_NEIGHBORS:
                indices = indices[:MAX_NEIGHBORS]
            
            self._neighbor_indices_list.append(indices)
            self._neighbor_counts[i] = len(indices)
        
        self._padded_neighbor_indices = torch.zeros(
            self.num_agents, MAX_NEIGHBORS, dtype=torch.long, device=DEVICE
        )
        self._neighbor_valid_mask = torch.zeros(
            self.num_agents, MAX_NEIGHBORS, dtype=torch.bool, device=DEVICE
        )
        
        for i, indices in enumerate(self._neighbor_indices_list):
            num_neighbors = len(indices)
            if num_neighbors > 0:
                self._padded_neighbor_indices[i, :num_neighbors] = indices
                self._neighbor_valid_mask[i, :num_neighbors] = True
        
        self._max_actual_neighbors = int(self._neighbor_counts.max().item())
        
        # 🔧 预计算角色信息
        self._precompute_role_info()
        
        if verbose:
            print(f"📊 Precomputed neighbor indices:")
            print(f"   Max neighbors per agent: {self._max_actual_neighbors}")
            print(f"   Neighbor counts: {self._neighbor_counts.tolist()}")
            print(f"   Role encoding: Leader=0, Pinned=1, Normal=2")
    
    def _precompute_role_info(self, verbose=False):
        """
        🔧 预计算角色信息（one-hot 编码）
        
        角色定义：
        - 0: 领导者 (Leader)
        - 1: 直接与领导者通信的跟随者 (Pinned Follower)
        - 2: 普通跟随者 (Normal Follower)
        """
        # 每个智能体的角色 ID
        self._role_ids = torch.zeros(self.num_agents, dtype=torch.long, device=DEVICE)
        self._role_ids[0] = 0  # 领导者
        
        pinned_set = set(self.topology.pinned_followers)
        for i in range(1, self.num_agents):
            if i in pinned_set:
                self._role_ids[i] = 1  # Pinned follower
            else:
                self._role_ids[i] = 2  # Normal follower
        
        # 预计算 one-hot 编码 (num_agents, 3)
        self._role_onehot = torch.zeros(self.num_agents, SELF_ROLE_DIM, device=DEVICE)
        self._role_onehot.scatter_(1, self._role_ids.unsqueeze(1), 1.0)
        
        if verbose:
            print(f"   Pinned followers: {self.topology.pinned_followers}")
            print(f"   Role IDs: {self._role_ids.tolist()}")
    
    def _leader_state_batch(self, t, env_ids=None):
        """
        批量计算领导者状态（支持每个环境独立的动力学参数和轨迹类型）
        
        Args:
            t: 时间张量
            env_ids: 环境索引（None 表示所有环境）
        
        轨迹类型：
        - sine: A * sin(ω*t + φ)
        - cosine: A * cos(ω*t + φ)
        - mixed: A * (sin(ω*t + φ) + 0.3*cos(0.5*ω*t))
        - chirp: A * sin((ω + 0.1*t)*t + φ)  变频信号
        """
        if env_ids is None:
            amplitude = self.leader_amplitude
            omega = self.leader_omega
            phase = self.leader_phase
            type_ids = self.trajectory_type_ids
            num_envs = self.num_envs
        else:
            amplitude = self.leader_amplitude[env_ids]
            omega = self.leader_omega[env_ids]
            phase = self.leader_phase[env_ids]
            type_ids = self.trajectory_type_ids[env_ids]
            num_envs = len(env_ids)
        
        pos = torch.zeros(num_envs, device=DEVICE)
        vel = torch.zeros(num_envs, device=DEVICE)
        
        # Sine 轨迹
        sine_mask = type_ids == self.type_to_id.get('sine', 0)
        if sine_mask.any():
            pos[sine_mask] = amplitude[sine_mask] * torch.sin(omega[sine_mask] * t[sine_mask] + phase[sine_mask])
            vel[sine_mask] = amplitude[sine_mask] * omega[sine_mask] * torch.cos(omega[sine_mask] * t[sine_mask] + phase[sine_mask])
        
        # Cosine 轨迹
        cosine_mask = type_ids == self.type_to_id.get('cosine', 1)
        if cosine_mask.any():
            pos[cosine_mask] = amplitude[cosine_mask] * torch.cos(omega[cosine_mask] * t[cosine_mask] + phase[cosine_mask])
            vel[cosine_mask] = -amplitude[cosine_mask] * omega[cosine_mask] * torch.sin(omega[cosine_mask] * t[cosine_mask] + phase[cosine_mask])
        
        # Mixed 轨迹
        mixed_mask = type_ids == self.type_to_id.get('mixed', 2)
        if mixed_mask.any():
            t_m = t[mixed_mask]
            A_m = amplitude[mixed_mask]
            omega_m = omega[mixed_mask]
            phi_m = phase[mixed_mask]
            pos[mixed_mask] = A_m * (torch.sin(omega_m * t_m + phi_m) + 0.3 * torch.cos(0.5 * omega_m * t_m))
            vel[mixed_mask] = A_m * (omega_m * torch.cos(omega_m * t_m + phi_m) - 0.15 * omega_m * torch.sin(0.5 * omega_m * t_m))
        
        # Chirp 轨迹（变频信号）
        chirp_mask = type_ids == self.type_to_id.get('chirp', 3)
        if chirp_mask.any():
            t_c = t[chirp_mask]
            A_c = amplitude[chirp_mask]
            omega_c = omega[chirp_mask]
            phi_c = phase[chirp_mask]
            chirp_rate = 0.1
            inst_phase = (omega_c + chirp_rate * t_c) * t_c + phi_c
            inst_freq = omega_c + 2 * chirp_rate * t_c
            pos[chirp_mask] = A_c * torch.sin(inst_phase)
            vel[chirp_mask] = A_c * inst_freq * torch.cos(inst_phase)
        
        return pos, vel
    
    def _randomize_leader_dynamics(self, env_ids):
        """
        🔧 随机化领导者动力学参数和轨迹类型
        
        Args:
            env_ids: 需要随机化的环境索引
        """
        num_envs = len(env_ids)
        
        # 随机振幅
        self.leader_amplitude[env_ids] = torch.empty(num_envs, device=DEVICE).uniform_(
            self.amplitude_range[0], self.amplitude_range[1]
        )
        
        # 随机角频率
        self.leader_omega[env_ids] = torch.empty(num_envs, device=DEVICE).uniform_(
            self.omega_range[0], self.omega_range[1]
        )
        
        # 随机相位
        self.leader_phase[env_ids] = torch.empty(num_envs, device=DEVICE).uniform_(
            self.phase_range[0], self.phase_range[1]
        )
        
        # 🔧 随机轨迹类型
        import numpy as np
        random_types = np.random.choice(len(self.trajectory_types), size=num_envs)
        self.trajectory_type_ids[env_ids] = torch.tensor(random_types, device=DEVICE)
    
    def _randomize_follower_init(self, env_ids, leader_pos, leader_vel):
        """
        🔧 随机化跟随者初始状态
        
        Args:
            env_ids: 需要随机化的环境索引
            leader_pos: 领导者初始位置
            leader_vel: 领导者初始速度
        """
        num_envs = len(env_ids)
        
        # 为每个环境随机生成位置和速度的标准差
        pos_std = torch.empty(num_envs, 1, device=DEVICE).uniform_(
            self.pos_std_range[0], self.pos_std_range[1]
        )
        vel_std = torch.empty(num_envs, 1, device=DEVICE).uniform_(
            self.vel_std_range[0], self.vel_std_range[1]
        )
        
        # 生成随机偏移
        pos_offset = torch.randn(num_envs, self.num_followers, device=DEVICE) * pos_std
        vel_offset = torch.randn(num_envs, self.num_followers, device=DEVICE) * vel_std
        
        # 设置跟随者初始状态
        self.positions[env_ids, 1:] = leader_pos.unsqueeze(1) + pos_offset
        self.velocities[env_ids, 1:] = leader_vel.unsqueeze(1) + vel_offset
    
    def get_global_state(self):
        """
        CTDE：获取全局状态（用于集中式 Critic）
        
        Returns:
            global_state: (num_envs, global_state_dim)
        """
        self._global_state_buffer[:, 0::2] = self.positions / self.pos_limit
        self._global_state_buffer[:, 1::2] = self.velocities / self.vel_limit
        return self._global_state_buffer.clone()
    
    def reset(self, env_ids=None):
        """
        重置环境（支持随机初始化）
        
        Args:
            env_ids: 要重置的环境索引（None 表示所有环境）
        """
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=DEVICE)
        
        num_reset = len(env_ids) if isinstance(env_ids, torch.Tensor) else self.num_envs
        
        # 重置时间
        self.t[env_ids] = 0.0
        
        # 🔧 随机化拓扑结构（仅在重置所有环境时）
        if self.randomize_topology and (env_ids is None or len(env_ids) == self.num_envs):
            self._randomize_topology()
        
        # 🔧 随机化领导者动力学参数
        if self.randomize_leader:
            self._randomize_leader_dynamics(env_ids)
        
        # 计算领导者初始状态
        leader_pos, leader_vel = self._leader_state_batch(self.t[env_ids], env_ids)
        
        self.positions[env_ids, 0] = leader_pos
        self.velocities[env_ids, 0] = leader_vel
        
        # 🔧 随机化跟随者初始状态
        if self.randomize_follower:
            self._randomize_follower_init(env_ids, leader_pos, leader_vel)
        else:
            # 使用固定的初始分布
            init_pos_std = 0.5
            init_vel_std = 0.2
            self.positions[env_ids, 1:] = leader_pos.unsqueeze(1) + torch.randn(
                num_reset, self.num_followers, device=DEVICE
            ) * init_pos_std
            self.velocities[env_ids, 1:] = leader_vel.unsqueeze(1) + torch.randn(
                num_reset, self.num_followers, device=DEVICE
            ) * init_vel_std
        
        # 限制在边界内
        self.positions[env_ids] = torch.clamp(self.positions[env_ids], -self.pos_limit, self.pos_limit)
        self.velocities[env_ids] = torch.clamp(self.velocities[env_ids], -self.vel_limit, self.vel_limit)
        
        # 重置广播状态
        self.last_broadcast_pos[env_ids] = self.positions[env_ids].clone()
        self.last_broadcast_vel[env_ids] = self.velocities[env_ids].clone()
        
        # 只重置指定环境的 prev_error
        self._prev_error[env_ids] = 0.0
        self._prev_error_valid[env_ids] = False
        
        return self._get_state_optimized()
    
    def _randomize_topology(self):
        """
        🔧 随机化拓扑结构并更新相关缓存
        """
        # 调用拓扑的随机化方法
        self.topology.randomize()
        
        # 重新计算邻居索引和角色信息（静默模式）
        self._precompute_neighbor_indices(verbose=False)
    
    def _get_state_optimized(self):
        """
        获取本地状态（用于分散式 Actor）
        
        状态结构（每个智能体）:
        - [0:2] 自身位置、速度（归一化）
        - [2:5] 自身角色 one-hot [leader, pinned, normal]
        - [5:5+MAX_NEIGHBORS*5] 邻居数据，每个邻居 5 维:
            - [0:2] 邻居位置、速度（归一化）
            - [2:5] 邻居角色 one-hot [leader, pinned, normal]
        """
        self._state_buffer.zero_()
        
        # 本地观测：自身位置、速度
        self._state_buffer[:, :, 0] = self.positions / self.pos_limit
        self._state_buffer[:, :, 1] = self.velocities / self.vel_limit
        
        # 自身角色 one-hot（对所有环境广播）
        # _role_onehot: (num_agents, 3) -> 扩展到 (num_envs, num_agents, 3)
        self._state_buffer[:, :, LOCAL_OBS_DIM:LOCAL_OBS_DIM + SELF_ROLE_DIM] = self._role_onehot.unsqueeze(0)
        
        # 邻居数据起始位置
        neighbor_start = LOCAL_OBS_DIM + SELF_ROLE_DIM  # 5
        
        # 邻居广播状态（归一化）
        broadcast_pos_norm = self.last_broadcast_pos / self.pos_limit
        broadcast_vel_norm = self.last_broadcast_vel / self.vel_limit
        
        for i in range(self.num_agents):
            num_neighbors = self._neighbor_counts[i].item()
            
            if num_neighbors == 0:
                continue
            
            neighbor_indices = self._neighbor_indices_list[i]
            
            # 邻居位置、速度
            neighbor_pos = broadcast_pos_norm[:, neighbor_indices]
            neighbor_vel = broadcast_vel_norm[:, neighbor_indices]
            
            # 邻居角色 one-hot
            neighbor_roles = self._role_onehot[neighbor_indices]  # (num_neighbors, 3)
            
            for j in range(num_neighbors):
                base_idx = neighbor_start + j * NEIGHBOR_FEAT_DIM
                # 位置、速度
                self._state_buffer[:, i, base_idx] = neighbor_pos[:, j]
                self._state_buffer[:, i, base_idx + 1] = neighbor_vel[:, j]
                # 角色 one-hot
                self._state_buffer[:, i, base_idx + 2:base_idx + 2 + NEIGHBOR_ROLE_DIM] = neighbor_roles[j]
        
        return self._state_buffer.clone()
    
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
        """执行一步"""
        self.t += DT
        
        # 🔧 更新领导者（使用每个环境独立的动力学参数）
        leader_pos, leader_vel = self._leader_state_batch(self.t)
        self.positions[:, 0] = leader_pos
        self.velocities[:, 0] = leader_vel
        self.last_broadcast_pos[:, 0] = leader_pos
        self.last_broadcast_vel[:, 0] = leader_vel
        
        # 解析动作
        # 🔧 Actor 输出的第 0 维已经按 V_SCALE 缩放过，这里不再二次缩放
        delta_v = action[:, :, 0]
        raw_threshold = action[:, :, 1]
        
        # 阈值映射
        normalized_threshold = raw_threshold / self.th_scale
        normalized_threshold = normalized_threshold.clamp(0.0, 1.0)
        threshold = self.threshold_min + (self.threshold_max - self.threshold_min) * normalized_threshold
        threshold = threshold.clamp(min=max(0.001, self.threshold_min), 
                                     max=min(self.threshold_max, 1.0))
        
        # 无模型动力学
        follower_vel = self.velocities[:, 1:]
        follower_pos = self.positions[:, 1:]
        
        new_vel = follower_vel + delta_v
        new_vel = torch.clamp(new_vel, -self.vel_limit, self.vel_limit)
        
        new_pos = follower_pos + new_vel * DT
        new_pos = torch.clamp(new_pos, -self.pos_limit, self.pos_limit)
        
        self.positions[:, 1:] = new_pos
        self.velocities[:, 1:] = new_vel
        
        # 事件触发通信
        trigger_error = torch.abs(new_pos - self.last_broadcast_pos[:, 1:])
        is_triggered = trigger_error > threshold
        
        self.last_broadcast_pos[:, 1:] = torch.where(
            is_triggered, self.positions[:, 1:], self.last_broadcast_pos[:, 1:]
        )
        self.last_broadcast_vel[:, 1:] = torch.where(
            is_triggered, self.velocities[:, 1:], self.last_broadcast_vel[:, 1:]
        )
        
        # ==================== 计算奖励 ====================
        pos_error = torch.abs(self.positions[:, 1:] - self.positions[:, 0:1])
        vel_error = torch.abs(self.velocities[:, 1:] - self.velocities[:, 0:1])
        tracking_error = pos_error.mean(dim=1) + 0.5 * vel_error.mean(dim=1)
        
        # 1. 跟踪惩罚
        tracking_penalty = -torch.tanh(tracking_error * self.tracking_penalty_scale) * self.tracking_penalty_max
        
        # 2. 改进奖励
        improvement_bonus = torch.zeros_like(tracking_error)
        valid_mask = self._prev_error_valid
        if valid_mask.any():
            improvement = self._prev_error - tracking_error
            improvement_bonus = torch.where(
                valid_mask,
                torch.clamp(improvement * self.improvement_scale, 
                           -self.improvement_clip, self.improvement_clip),
                torch.zeros_like(improvement)
            )
        
        # 更新 prev_error
        self._prev_error = tracking_error.detach().clone()
        self._prev_error_valid[:] = True
        
        # 3. 通信惩罚
        comm_weight = torch.exp(-tracking_error * self.comm_weight_decay)
        comm_rate = is_triggered.float().mean(dim=1)
        comm_penalty = -comm_rate * self.comm_penalty_base * comm_weight
        
        # 总奖励
        raw_reward = tracking_penalty + improvement_bonus + comm_penalty
        rewards = self._scale_reward_batch(raw_reward)
        
        dones = torch.zeros(self.num_envs, dtype=torch.bool, device=DEVICE)
        
        infos = {
            'tracking_error': tracking_error,
            'comm_rate': comm_rate,
            'comm_weight': comm_weight,
            'leader_pos': self.positions[:, 0],
            'leader_vel': self.velocities[:, 0],
            'avg_follower_pos': self.positions[:, 1:].mean(dim=1),
            'threshold_mean': threshold.mean(),
            'tracking_penalty': tracking_penalty.mean(),
            'improvement_bonus': improvement_bonus.mean(),
            'comm_penalty': comm_penalty.mean(),
            # 🔧 新增：领导者动力学参数信息
            'leader_amplitude_mean': self.leader_amplitude.mean(),
            'leader_omega_mean': self.leader_omega.mean(),
        }
        
        return self._get_state_optimized(), rewards, dones, infos


class ModelFreeEnv:
    """单环境版本"""
    
    def __init__(self, topology):
        self.batched_env = BatchedModelFreeEnv(topology, num_envs=1)
        self.topology = topology
        self.num_agents = topology.num_agents
        self.num_followers = topology.num_followers
    
    @property
    def positions(self):
        return self.batched_env.positions[0]
    
    @property
    def velocities(self):
        return self.batched_env.velocities[0]
    
    @property
    def t(self):
        return self.batched_env.t[0].item()
    
    @property
    def leader_amplitude(self):
        return self.batched_env.leader_amplitude[0].item()
    
    @property
    def leader_omega(self):
        return self.batched_env.leader_omega[0].item()
    
    def get_global_state(self):
        """获取全局状态"""
        return self.batched_env.get_global_state()[0]
    
    def reset(self):
        state = self.batched_env.reset()
        return state[0]
    
    def step(self, action):
        action_batched = action.unsqueeze(0)
        states, rewards, dones, infos = self.batched_env.step(action_batched)
        info = {k: (v[0].item() if isinstance(v, torch.Tensor) and v.dim() > 0 else
                    v.item() if isinstance(v, torch.Tensor) else v)
                for k, v in infos.items()}
        return states[0], rewards[0].item(), dones[0].item(), info


# 兼容旧接口
BatchedLeaderFollowerEnv = BatchedModelFreeEnv
LeaderFollowerMASEnv = ModelFreeEnv