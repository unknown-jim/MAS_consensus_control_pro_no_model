"""领导者-跟随者多智能体系统环境（CTDE 版本）。

特性：
- 每个 episode 可随机化：领导者动力学、跟随者初始分布、通信拓扑
- CTDE：提供 `get_global_state()` 供 centralized critic/value 使用
"""

from __future__ import annotations

import torch

from .config import (
    ACTION_DIM,
    AGE_MAX_STEPS,
    COMM_PENALTY,
    COMM_WEIGHT_DECAY,
    COOLDOWN_STEPS,
    DEVICE,
    DT,
    ETC_VEL_COEF,
    FOLLOWER_INIT_POS_STD,
    FOLLOWER_INIT_POS_STD_RANGE,
    FOLLOWER_INIT_VEL_STD,
    FOLLOWER_INIT_VEL_STD_RANGE,
    GLOBAL_STATE_DIM,
    GLOBAL_STATE_INCLUDE_BROADCAST,
    GLOBAL_STATE_INCLUDE_LEADER_PARAMS,
    GLOBAL_STATE_INCLUDE_TIME,
    GLOBAL_STATE_INCLUDE_TRAJ_TYPE,
    IMPROVEMENT_CLIP,
    IMPROVEMENT_SCALE,
    INFO_GAIN_SCALE,
    LEADER_AMPLITUDE,
    LEADER_AMPLITUDE_RANGE,
    LEADER_OMEGA,
    LEADER_OMEGA_RANGE,
    LEADER_PHASE,
    LEADER_PHASE_RANGE,
    LEADER_TRAJECTORY_TYPES,
    LOCAL_OBS_DIM,
    MAX_NEIGHBORS,
    MAX_STEPS,
    NEIGHBOR_FEAT_DIM,
    NUM_AGENTS,
    POS_LIMIT,
    RANDOMIZE_FOLLOWER,
    RANDOMIZE_LEADER,
    RANDOMIZE_TOPOLOGY,
    REWARD_MAX,
    REWARD_MIN,
    SELF_ROLE_DIM,
    STATE_DIM,
    THRESHOLD_MAX,
    THRESHOLD_MIN,
    TRACKING_PENALTY_MAX,
    TRACKING_PENALTY_SCALE,
    USE_SOFT_REWARD_SCALING,
    VEL_LIMIT,
    V_SCALE,
)


class BatchedModelFreeEnv:
    """无模型并行环境（CTDE 版本）。

    环境内部维护 `num_envs` 个并行 episode，用向量化方式推进动力学与通信触发逻辑。

    Args:
        topology: `CommunicationTopology` 实例。
        num_envs: 并行环境数量（E）。

    Notes:
        - `reset()` 返回本地状态（给 actor 用）。
        - `get_global_state()` 返回全局状态（给 centralized critic/value 用）。
    """

    def __init__(self, topology, num_envs: int = 64):
        self.topology = topology
        self.num_envs = int(num_envs)
        self.num_agents = topology.num_agents
        self.num_followers = topology.num_followers
        self.leader_id = topology.leader_id

        # 领导者动力学参数（基准值，会在 reset 时随机化）
        self.leader_amplitude_base = LEADER_AMPLITUDE
        self.leader_omega_base = LEADER_OMEGA
        self.leader_phase_base = LEADER_PHASE

        self.leader_amplitude = torch.full((self.num_envs,), LEADER_AMPLITUDE, device=DEVICE)
        self.leader_omega = torch.full((self.num_envs,), LEADER_OMEGA, device=DEVICE)
        self.leader_phase = torch.full((self.num_envs,), LEADER_PHASE, device=DEVICE)

        self.trajectory_types = LEADER_TRAJECTORY_TYPES
        self.type_to_id = {t: i for i, t in enumerate(self.trajectory_types)}
        self.id_to_type = {i: t for i, t in enumerate(self.trajectory_types)}
        self.trajectory_type_ids = torch.zeros(self.num_envs, dtype=torch.long, device=DEVICE)

        self.randomize_leader = RANDOMIZE_LEADER
        self.randomize_follower = RANDOMIZE_FOLLOWER
        self.randomize_topology = RANDOMIZE_TOPOLOGY

        self.amplitude_range = LEADER_AMPLITUDE_RANGE
        self.omega_range = LEADER_OMEGA_RANGE
        self.phase_range = LEADER_PHASE_RANGE

        self.pos_std_range = FOLLOWER_INIT_POS_STD_RANGE
        self.vel_std_range = FOLLOWER_INIT_VEL_STD_RANGE

        self.pos_limit = float(POS_LIMIT)
        self.vel_limit = float(VEL_LIMIT)
        self.reward_min = float(REWARD_MIN)
        self.reward_max = float(REWARD_MAX)
        self.use_soft_scaling = bool(USE_SOFT_REWARD_SCALING)

        self.comm_penalty_base = float(COMM_PENALTY)
        self.tracking_penalty_scale = float(TRACKING_PENALTY_SCALE)
        self.tracking_penalty_max = float(TRACKING_PENALTY_MAX)
        self.comm_weight_decay = float(COMM_WEIGHT_DECAY)
        self.improvement_scale = float(IMPROVEMENT_SCALE)
        self.improvement_clip = float(IMPROVEMENT_CLIP)
        self.info_gain_scale = float(INFO_GAIN_SCALE)

        self.v_scale = float(V_SCALE)

        self._precompute_neighbor_indices()

        self.positions = torch.zeros(self.num_envs, self.num_agents, device=DEVICE)
        self.velocities = torch.zeros(self.num_envs, self.num_agents, device=DEVICE)

        self.last_broadcast_pos = torch.zeros(self.num_envs, self.num_agents, device=DEVICE)
        self.last_broadcast_vel = torch.zeros(self.num_envs, self.num_agents, device=DEVICE)

        # leader gossip
        self.leader_seq = torch.zeros(self.num_envs, dtype=torch.long, device=DEVICE)

        self.leader_est_pos = torch.zeros(self.num_envs, self.num_agents, device=DEVICE)
        self.leader_est_vel = torch.zeros(self.num_envs, self.num_agents, device=DEVICE)
        self.leader_est_seq = torch.full((self.num_envs, self.num_agents), -1, dtype=torch.long, device=DEVICE)

        self.last_broadcast_leader_pos = torch.zeros(self.num_envs, self.num_agents, device=DEVICE)
        self.last_broadcast_leader_vel = torch.zeros(self.num_envs, self.num_agents, device=DEVICE)
        self.last_broadcast_leader_seq = torch.full((self.num_envs, self.num_agents), -1, dtype=torch.long, device=DEVICE)

        self.t = torch.zeros(self.num_envs, device=DEVICE)

        self._prev_error = torch.zeros(self.num_envs, device=DEVICE)
        self._prev_error_valid = torch.zeros(self.num_envs, dtype=torch.bool, device=DEVICE)

        # ETC 相关状态缓存
        self._cooldown_counter = torch.zeros(self.num_envs, self.num_followers, dtype=torch.long, device=DEVICE)
        # 方案(1)：距上次 leader 信息更新的步数（用于"多久没收到更新"保底触发）
        self._since_leader_update = torch.zeros(self.num_envs, self.num_followers, dtype=torch.long, device=DEVICE)

        # ETC 超参缓存（避免每步从 config 读取）
        self._etc_vel_coef = float(ETC_VEL_COEF)
        self._age_max_steps = int(AGE_MAX_STEPS)
        self._cooldown_steps = int(COOLDOWN_STEPS)
        self._threshold_min = float(THRESHOLD_MIN)
        self._threshold_max = float(THRESHOLD_MAX)

        self._state_buffer = torch.zeros(self.num_envs, self.num_agents, STATE_DIM, device=DEVICE)

        # 评估场景支持任意 `num_agents`：全局状态维度与 buffer 按 topology 动态分配
        self.global_state_dim = int(self._compute_global_state_dim(self.num_agents))
        self._global_state_buffer = torch.zeros(self.num_envs, self.global_state_dim, device=DEVICE)

        self.reset()

    def _precompute_neighbor_indices(self, verbose: bool = True):
        """预计算每个智能体的候选入邻居索引。

        注意：
        - `MAX_NEIGHBORS` 现在语义为 **Actor 输入的 Top-K 槽位数**（固定维度，不随智能体数量增长）。
        - 这里会保留“全量候选邻居”（入邻居）用于：
            1) leader gossip 吸收更新（看见所有可接收的邻居广播）
            2) 构造观测时按 `leader_age` 选择 Top-K
        """

        self._neighbor_indices_list = []
        self._neighbor_counts = torch.zeros(self.num_agents, dtype=torch.long, device=DEVICE)

        for i in range(self.num_agents):
            can_receive_mask = self.topology.adj_matrix[i, :] > 0
            indices = torch.where(can_receive_mask)[0]

            self._neighbor_indices_list.append(indices)
            self._neighbor_counts[i] = len(indices)

        # 候选邻居槽位数（用于内部 gather/max）；至少要 >= MAX_NEIGHBORS，避免 topk/view 出错
        max_candidates = int(self._neighbor_counts.max().item())
        max_candidates = max(int(MAX_NEIGHBORS), max_candidates)

        self._padded_neighbor_indices = torch.zeros(self.num_agents, max_candidates, dtype=torch.long, device=DEVICE)
        self._neighbor_valid_mask = torch.zeros(self.num_agents, max_candidates, dtype=torch.bool, device=DEVICE)

        for i, indices in enumerate(self._neighbor_indices_list):
            num_neighbors = int(len(indices))
            if num_neighbors > 0:
                self._padded_neighbor_indices[i, :num_neighbors] = indices
                self._neighbor_valid_mask[i, :num_neighbors] = True

        self._max_actual_neighbors = int(self._neighbor_counts.max().item())
        self._max_candidate_neighbors = int(max_candidates)

        # 方案(2)：预计算每个智能体的出度（是否有下游节点能接收自己的广播）
        # out_degree[i] = 有多少 j 满足 adj_matrix[j, i] > 0（即 j 能接收 i 的广播）
        self._out_degree = (self.topology.adj_matrix > 0).sum(dim=0).to(device=DEVICE)  # shape=(A,)
        # 叶子节点 mask（出度为 0 的 follower）：这些节点禁用 age 保底触发
        self._is_leaf_follower = self._out_degree[1:] == 0  # shape=(F,)

        self._precompute_role_info()

        if verbose:
            print("📊 Precomputed neighbor indices:")
            print(f"   Top-K slots (actor): {int(MAX_NEIGHBORS)}")
            print(f"   Candidate neighbors (max in-degree): {self._max_actual_neighbors}")
            print(f"   Neighbor counts: {self._neighbor_counts.tolist()}")
            print("   Role encoding: Leader=0, Pinned=1, Normal=2")

    def _precompute_role_info(self, verbose: bool = False):
        self._role_ids = torch.zeros(self.num_agents, dtype=torch.long, device=DEVICE)
        self._role_ids[0] = 0

        pinned_set = set(self.topology.pinned_followers)
        for i in range(1, self.num_agents):
            self._role_ids[i] = 1 if i in pinned_set else 2

        self._role_onehot = torch.zeros(self.num_agents, SELF_ROLE_DIM, device=DEVICE)
        self._role_onehot.scatter_(1, self._role_ids.unsqueeze(1), 1.0)

        if verbose:
            print(f"   Pinned followers: {self.topology.pinned_followers}")
            print(f"   Role IDs: {self._role_ids.tolist()}")

    def _leader_state_batch(self, t: torch.Tensor, env_ids=None):
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
            num_envs = int(env_ids.numel()) if isinstance(env_ids, torch.Tensor) else int(len(env_ids))

        pos = torch.zeros(num_envs, device=DEVICE)
        vel = torch.zeros(num_envs, device=DEVICE)

        sine_mask = type_ids == self.type_to_id.get("sine", 0)
        if sine_mask.any():
            pos[sine_mask] = amplitude[sine_mask] * torch.sin(omega[sine_mask] * t[sine_mask] + phase[sine_mask])
            vel[sine_mask] = amplitude[sine_mask] * omega[sine_mask] * torch.cos(
                omega[sine_mask] * t[sine_mask] + phase[sine_mask]
            )

        cosine_mask = type_ids == self.type_to_id.get("cosine", 1)
        if cosine_mask.any():
            pos[cosine_mask] = amplitude[cosine_mask] * torch.cos(omega[cosine_mask] * t[cosine_mask] + phase[cosine_mask])
            vel[cosine_mask] = -amplitude[cosine_mask] * omega[cosine_mask] * torch.sin(
                omega[cosine_mask] * t[cosine_mask] + phase[cosine_mask]
            )

        mixed_mask = type_ids == self.type_to_id.get("mixed", 2)
        if mixed_mask.any():
            t_m = t[mixed_mask]
            A_m = amplitude[mixed_mask]
            omega_m = omega[mixed_mask]
            phi_m = phase[mixed_mask]
            pos[mixed_mask] = A_m * (torch.sin(omega_m * t_m + phi_m) + 0.3 * torch.cos(0.5 * omega_m * t_m))
            vel[mixed_mask] = A_m * (
                omega_m * torch.cos(omega_m * t_m + phi_m) - 0.15 * omega_m * torch.sin(0.5 * omega_m * t_m)
            )

        chirp_mask = type_ids == self.type_to_id.get("chirp", 3)
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
        num_envs = int(env_ids.numel()) if isinstance(env_ids, torch.Tensor) else len(env_ids)

        self.leader_amplitude[env_ids] = torch.empty(num_envs, device=DEVICE).uniform_(
            self.amplitude_range[0], self.amplitude_range[1]
        )
        self.leader_omega[env_ids] = torch.empty(num_envs, device=DEVICE).uniform_(self.omega_range[0], self.omega_range[1])
        self.leader_phase[env_ids] = torch.empty(num_envs, device=DEVICE).uniform_(self.phase_range[0], self.phase_range[1])

        self.trajectory_type_ids[env_ids] = torch.randint(
            low=0,
            high=len(self.trajectory_types),
            size=(num_envs,),
            device=DEVICE,
            dtype=torch.long,
        )

    def _randomize_follower_init(self, env_ids, leader_pos, leader_vel):
        num_envs = int(env_ids.numel()) if isinstance(env_ids, torch.Tensor) else int(len(env_ids))

        pos_std = torch.empty(num_envs, 1, device=DEVICE).uniform_(self.pos_std_range[0], self.pos_std_range[1])
        vel_std = torch.empty(num_envs, 1, device=DEVICE).uniform_(self.vel_std_range[0], self.vel_std_range[1])

        pos_offset = torch.randn(num_envs, self.num_followers, device=DEVICE) * pos_std
        vel_offset = torch.randn(num_envs, self.num_followers, device=DEVICE) * vel_std

        self.positions[env_ids, 1:] = leader_pos.unsqueeze(1) + pos_offset
        self.velocities[env_ids, 1:] = leader_vel.unsqueeze(1) + vel_offset

    @staticmethod
    def _compute_global_state_dim(num_agents: int) -> int:
        """根据 `get_global_state()` 的拼接顺序动态计算全局状态维度。

        目的：让评估时可以把 `num_agents` 设置得很大，而不会因为 `GLOBAL_STATE_DIM`
        是训练时固定常量导致 buffer 越界。
        """
        n = int(num_agents)
        dim = 2 * n
        if bool(GLOBAL_STATE_INCLUDE_BROADCAST):
            dim += 2 * n
        if bool(GLOBAL_STATE_INCLUDE_LEADER_PARAMS):
            dim += 3
        if bool(GLOBAL_STATE_INCLUDE_TRAJ_TYPE):
            dim += len(LEADER_TRAJECTORY_TYPES)
        if bool(GLOBAL_STATE_INCLUDE_TIME):
            dim += 1
        return int(dim)

    def get_global_state(self):
        """构造 CTDE 的全局状态。

        Returns:
            全局状态张量，shape=(E, self.global_state_dim)。内容包含：
            - 所有 agent 的 (pos, vel)（归一化）
            - （可选）最近一次广播的 (pos, vel)
            - （可选）leader 动力学参数（幅值/角频率/相位）
            - （可选）轨迹类型 one-hot
            - （可选）时间归一化
        """
        offset = 0

        self._global_state_buffer[:, offset:offset + self.num_agents] = self.positions / self.pos_limit
        offset += self.num_agents
        self._global_state_buffer[:, offset:offset + self.num_agents] = self.velocities / self.vel_limit
        offset += self.num_agents

        if GLOBAL_STATE_INCLUDE_BROADCAST:
            self._global_state_buffer[:, offset:offset + self.num_agents] = self.last_broadcast_pos / self.pos_limit
            offset += self.num_agents
            self._global_state_buffer[:, offset:offset + self.num_agents] = self.last_broadcast_vel / self.vel_limit
            offset += self.num_agents

        if GLOBAL_STATE_INCLUDE_LEADER_PARAMS:
            self._global_state_buffer[:, offset] = self.leader_amplitude / (self.amplitude_range[1] + 1e-8)
            offset += 1
            self._global_state_buffer[:, offset] = self.leader_omega / (self.omega_range[1] + 1e-8)
            offset += 1
            self._global_state_buffer[:, offset] = self.leader_phase / (2.0 * 3.14159265)
            offset += 1

        if GLOBAL_STATE_INCLUDE_TRAJ_TYPE:
            k = len(self.trajectory_types)
            onehot = torch.zeros(self.num_envs, k, device=DEVICE, dtype=self._global_state_buffer.dtype)
            onehot.scatter_(1, self.trajectory_type_ids.unsqueeze(1), 1.0)
            self._global_state_buffer[:, offset:offset + k] = onehot
            offset += k

        if GLOBAL_STATE_INCLUDE_TIME:
            denom = float(MAX_STEPS) * float(DT) + 1e-8
            self._global_state_buffer[:, offset] = (self.t / denom).clamp(0.0, 1.0)
            offset += 1

        return self._global_state_buffer.clone()

    def reset(self, env_ids=None):
        """重置环境并返回本地状态。

        Args:
            env_ids: 需要重置的环境 id（Tensor 或 list）。若为 None 则重置全部环境。

        Returns:
            本地状态张量，shape=(E, num_agents, STATE_DIM)。
        """
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=DEVICE)

        if isinstance(env_ids, torch.Tensor):
            num_reset = int(env_ids.numel())
        else:
            num_reset = int(len(env_ids))

        self.t[env_ids] = 0.0

        # 仅当重置全部环境时随机化拓扑
        if self.randomize_topology and num_reset == self.num_envs:
            self._randomize_topology()

        if self.randomize_leader:
            self._randomize_leader_dynamics(env_ids)

        leader_pos, leader_vel = self._leader_state_batch(self.t[env_ids], env_ids)

        self.positions[env_ids, 0] = leader_pos
        self.velocities[env_ids, 0] = leader_vel

        if self.randomize_follower:
            self._randomize_follower_init(env_ids, leader_pos, leader_vel)
        else:
            init_pos_std = float(FOLLOWER_INIT_POS_STD)
            init_vel_std = float(FOLLOWER_INIT_VEL_STD)
            self.positions[env_ids, 1:] = leader_pos.unsqueeze(1) + torch.randn(num_reset, self.num_followers, device=DEVICE) * init_pos_std
            self.velocities[env_ids, 1:] = leader_vel.unsqueeze(1) + torch.randn(num_reset, self.num_followers, device=DEVICE) * init_vel_std

        self.positions[env_ids] = torch.clamp(self.positions[env_ids], -self.pos_limit, self.pos_limit)
        self.velocities[env_ids] = torch.clamp(self.velocities[env_ids], -self.vel_limit, self.vel_limit)

        self.last_broadcast_pos[env_ids] = self.positions[env_ids].clone()
        self.last_broadcast_vel[env_ids] = self.velocities[env_ids].clone()

        self.leader_seq[env_ids] = 0

        can_receive_leader = self.topology.adj_matrix[:, self.leader_id] > 0
        if isinstance(can_receive_leader, torch.Tensor):
            can_receive_leader = can_receive_leader.to(device=DEVICE)
        can_receive_leader[self.leader_id] = True

        self.leader_est_pos[env_ids].zero_()
        self.leader_est_vel[env_ids].zero_()
        self.leader_est_seq[env_ids].fill_(-1)

        mask = can_receive_leader.unsqueeze(0).expand(num_reset, -1)
        leader_pos_full = leader_pos.unsqueeze(1).expand(num_reset, self.num_agents)
        leader_vel_full = leader_vel.unsqueeze(1).expand(num_reset, self.num_agents)

        self.leader_est_pos[env_ids] = torch.where(mask, leader_pos_full, self.leader_est_pos[env_ids])
        self.leader_est_vel[env_ids] = torch.where(mask, leader_vel_full, self.leader_est_vel[env_ids])
        self.leader_est_seq[env_ids] = torch.where(mask, torch.zeros_like(self.leader_est_seq[env_ids]), self.leader_est_seq[env_ids])

        self.last_broadcast_leader_pos[env_ids] = self.leader_est_pos[env_ids]
        self.last_broadcast_leader_vel[env_ids] = self.leader_est_vel[env_ids]
        self.last_broadcast_leader_seq[env_ids] = self.leader_est_seq[env_ids]

        self._prev_error[env_ids] = 0.0
        self._prev_error_valid[env_ids] = False

        # 重置 ETC cooldown 计数器
        self._cooldown_counter[env_ids] = 0
        # 重置 since_leader_update 计数器
        self._since_leader_update[env_ids] = 0

        return self._get_state_optimized()

    def _randomize_topology(self):
        self.topology.randomize()
        self._precompute_neighbor_indices(verbose=False)

    def _get_state_optimized(self):
        self._state_buffer.zero_()

        self._state_buffer[:, :, 0] = self.positions / self.pos_limit
        self._state_buffer[:, :, 1] = self.velocities / self.vel_limit

        denom_steps = float(MAX_STEPS) if float(MAX_STEPS) > 0 else 1.0

        leader_pos_norm = self.leader_est_pos / self.pos_limit
        leader_vel_norm = self.leader_est_vel / self.vel_limit

        seq = self.leader_est_seq
        seq_norm = torch.clamp(seq.float() / denom_steps, 0.0, 1.0)

        age = torch.where(
            seq >= 0,
            (self.leader_seq.unsqueeze(1) - seq).clamp(min=0).float(),
            torch.full_like(seq_norm, denom_steps),
        )
        age_norm = torch.clamp(age / denom_steps, 0.0, 1.0)

        self._state_buffer[:, :, 2] = leader_pos_norm
        self._state_buffer[:, :, 3] = leader_vel_norm
        self._state_buffer[:, :, 4] = seq_norm
        self._state_buffer[:, :, 5] = age_norm

        self._state_buffer[:, :, LOCAL_OBS_DIM:LOCAL_OBS_DIM + SELF_ROLE_DIM] = self._role_onehot.unsqueeze(0)

        neighbor_start = LOCAL_OBS_DIM + SELF_ROLE_DIM

        broadcast_pos_norm = self.last_broadcast_pos / self.pos_limit
        broadcast_vel_norm = self.last_broadcast_vel / self.vel_limit

        b_leader_pos_norm = self.last_broadcast_leader_pos / self.pos_limit
        b_leader_vel_norm = self.last_broadcast_leader_vel / self.vel_limit

        b_seq = self.last_broadcast_leader_seq
        b_seq_safe = torch.where(b_seq >= 0, b_seq.float(), torch.zeros_like(b_seq.float()))
        b_seq_norm = torch.clamp(b_seq_safe / denom_steps, 0.0, 1.0)

        b_age = torch.where(
            b_seq >= 0,
            (self.leader_seq.unsqueeze(1) - b_seq).clamp(min=0).float(),
            torch.full_like(b_seq_norm, denom_steps),
        )
        b_age_norm = torch.clamp(b_age / denom_steps, 0.0, 1.0)

        # 候选邻居集合（全量入邻居），以及有效 mask
        padded_idx = self._padded_neighbor_indices
        valid_mask = self._neighbor_valid_mask

        idx = padded_idx.unsqueeze(0).expand(self.num_envs, -1, -1)   # (E, A, Kcand)
        valid = valid_mask.unsqueeze(0).expand(self.num_envs, -1, -1) # (E, A, Kcand)

        candidate_k = int(padded_idx.shape[1])
        idx_flat = idx.reshape(self.num_envs, -1)  # (E, A*kcand)

        # 先抽取“候选邻居”的特征（不直接写入观测；后面按 age 选 Top-K）
        cand_pos = broadcast_pos_norm.gather(1, idx_flat).view(self.num_envs, self.num_agents, candidate_k)
        cand_vel = broadcast_vel_norm.gather(1, idx_flat).view(self.num_envs, self.num_agents, candidate_k)

        cand_leader_pos = b_leader_pos_norm.gather(1, idx_flat).view(self.num_envs, self.num_agents, candidate_k)
        cand_leader_vel = b_leader_vel_norm.gather(1, idx_flat).view(self.num_envs, self.num_agents, candidate_k)
        cand_leader_seq = b_seq_norm.gather(1, idx_flat).view(self.num_envs, self.num_agents, candidate_k)
        cand_leader_age = b_age_norm.gather(1, idx_flat).view(self.num_envs, self.num_agents, candidate_k)

        # 按 “leader_age 越小越新鲜” 选 Top-K（K = MAX_NEIGHBORS），无效槽位置为极大避免被选中
        age_for_sort = cand_leader_age.clone()
        age_for_sort = torch.where(valid, age_for_sort, torch.full_like(age_for_sort, 1e9))

        # torch.topk 支持 largest=False 取最小的 K 个
        _, topk_idx = torch.topk(age_for_sort, k=int(MAX_NEIGHBORS), dim=2, largest=False)

        # 把 Top-K 特征按新鲜度顺序 gather 出来
        neighbor_pos = cand_pos.gather(2, topk_idx)
        neighbor_vel = cand_vel.gather(2, topk_idx)

        neighbor_leader_pos = cand_leader_pos.gather(2, topk_idx)
        neighbor_leader_vel = cand_leader_vel.gather(2, topk_idx)
        neighbor_leader_seq = cand_leader_seq.gather(2, topk_idx)
        neighbor_leader_age = cand_leader_age.gather(2, topk_idx)

        # 把对应的有效位也 gather 过来，确保“没有足够邻居”时其余槽位为 0
        selected_valid = valid.gather(2, topk_idx)
        valid_f = selected_valid.to(dtype=neighbor_pos.dtype)

        neighbor_pos = neighbor_pos * valid_f
        neighbor_vel = neighbor_vel * valid_f
        neighbor_leader_pos = neighbor_leader_pos * valid_f
        neighbor_leader_vel = neighbor_leader_vel * valid_f
        neighbor_leader_seq = neighbor_leader_seq * valid_f
        neighbor_leader_age = neighbor_leader_age * valid_f

        neighbor_feat = self._state_buffer[:, :, neighbor_start:].view(
            self.num_envs, self.num_agents, int(MAX_NEIGHBORS), int(NEIGHBOR_FEAT_DIM)
        )
        neighbor_feat[:, :, :, 0] = neighbor_pos
        neighbor_feat[:, :, :, 1] = neighbor_vel
        neighbor_feat[:, :, :, 2] = neighbor_leader_pos
        neighbor_feat[:, :, :, 3] = neighbor_leader_vel
        neighbor_feat[:, :, :, 4] = neighbor_leader_seq
        neighbor_feat[:, :, :, 5] = neighbor_leader_age

        return self._state_buffer.clone()

    def _scale_reward_batch(self, reward: torch.Tensor):
        if self.use_soft_scaling:
            mid = (self.reward_max + self.reward_min) / 2
            scale = (self.reward_max - self.reward_min) / 2
            normalized = (reward - mid) / (scale + 1e-8)
            return mid + scale * torch.tanh(normalized)
        return torch.clamp(reward, self.reward_min, self.reward_max)

    def step(self, action: torch.Tensor):
        """推进一步环境动力学与通信触发。

        Args:
            action: follower 动作，shape=(E, num_followers, ACTION_DIM)。
                - action[..., 0]：速度增量（delta_v）
                - action[..., 1]：阈值归一化值（theta_norm），范围 [0, 1]
                  环境会用反向映射：theta = TH_MAX - theta_norm * (TH_MAX - TH_MIN)
                  这样 theta_norm 越大，theta 越小，通信越频繁

        Returns:
            states: 下一步本地状态，shape=(E, num_agents, STATE_DIM)
            rewards: shape=(E,)
            dones: shape=(E,)（当前实现不提前终止，主要由训练端做时间截断）
            infos: 诊断信息字典（tracking_error/comm_rate 等）
        """
        self.t += DT

        leader_pos, leader_vel = self._leader_state_batch(self.t)
        self.positions[:, 0] = leader_pos
        self.velocities[:, 0] = leader_vel

        self.last_broadcast_pos[:, 0] = leader_pos
        self.last_broadcast_vel[:, 0] = leader_vel

        self.leader_seq += 1

        self.leader_est_pos[:, 0] = leader_pos
        self.leader_est_vel[:, 0] = leader_vel
        self.leader_est_seq[:, 0] = self.leader_seq

        self.last_broadcast_leader_pos[:, 0] = leader_pos
        self.last_broadcast_leader_vel[:, 0] = leader_vel
        self.last_broadcast_leader_seq[:, 0] = self.leader_seq

        delta_v = action[:, :, 0]
        theta_norm = action[:, :, 1].clamp(0.0, 1.0)  # 阈值归一化值

        # 反向映射：theta_norm 越大 → theta 越小 → 通信越频繁
        # 这与原 Beta 分布"初始高通信"的初始化趋势一致
        theta = self._threshold_max - theta_norm * (self._threshold_max - self._threshold_min)

        follower_vel = self.velocities[:, 1:]
        follower_pos = self.positions[:, 1:]

        new_vel = torch.clamp(follower_vel + delta_v, -self.vel_limit, self.vel_limit)
        new_pos = torch.clamp(follower_pos + new_vel * DT, -self.pos_limit, self.pos_limit)

        self.positions[:, 1:] = new_pos
        self.velocities[:, 1:] = new_vel

        # ============================================================
        # 确定性事件触发（ETC）逻辑
        # ============================================================
        # 1. 计算触发增量 delta = |x - x_b| + coef * DT * |v - v_b|
        pos_diff = torch.abs(new_pos - self.last_broadcast_pos[:, 1:])
        vel_diff = torch.abs(new_vel - self.last_broadcast_vel[:, 1:])
        delta = pos_diff + self._etc_vel_coef * DT * vel_diff

        # 2. 阈值触发条件
        threshold_triggered = delta > theta

        # 3. 新鲜度保底触发（方案1改进）：用"多久没收到更新"而非"绝对滞后"
        #    _since_leader_update 记录距上次 leader_est_seq 被更新的步数
        #    当 since_leader_update > AGE_MAX_STEPS 时触发
        age_triggered_raw = self._since_leader_update > self._age_max_steps

        # 方案(2)：叶子节点（出度为 0）禁用 age 保底触发，因为它们发送的广播没有下游接收者
        # _is_leaf_follower: shape=(F,)，需要 broadcast 到 (E, F)
        is_leaf = self._is_leaf_follower.unsqueeze(0).expand(self.num_envs, -1)
        age_triggered = age_triggered_raw & (~is_leaf)

        # 4. Cooldown 去抖：触发后 COOLDOWN_STEPS 步内不再触发
        cooldown_ok = self._cooldown_counter == 0

        # 5. 综合触发条件：(阈值触发 OR 新鲜度保底) AND cooldown 允许
        is_triggered = (threshold_triggered | age_triggered) & cooldown_ok

        # 6. 更新 cooldown 计数器
        # 触发的 follower 重置为 COOLDOWN_STEPS；未触发的递减（但不低于 0）
        self._cooldown_counter = torch.where(
            is_triggered,
            torch.full_like(self._cooldown_counter, self._cooldown_steps),
            (self._cooldown_counter - 1).clamp(min=0),
        )

        self.last_broadcast_pos[:, 1:] = torch.where(is_triggered, self.positions[:, 1:], self.last_broadcast_pos[:, 1:])
        self.last_broadcast_vel[:, 1:] = torch.where(is_triggered, self.velocities[:, 1:], self.last_broadcast_vel[:, 1:])

        trigger_full = torch.zeros(self.num_envs, self.num_agents, dtype=torch.bool, device=DEVICE)
        trigger_full[:, 1:] = is_triggered

        self.last_broadcast_leader_pos = torch.where(trigger_full, self.leader_est_pos, self.last_broadcast_leader_pos)
        self.last_broadcast_leader_vel = torch.where(trigger_full, self.leader_est_vel, self.last_broadcast_leader_vel)
        self.last_broadcast_leader_seq = torch.where(trigger_full, self.leader_est_seq, self.last_broadcast_leader_seq)

        padded_idx = self._padded_neighbor_indices
        valid_mask = self._neighbor_valid_mask

        idx = padded_idx.unsqueeze(0).expand(self.num_envs, -1, -1)
        valid = valid_mask.unsqueeze(0).expand(self.num_envs, -1, -1)

        candidate_k = int(padded_idx.shape[1])
        idx_flat = idx.reshape(self.num_envs, -1)

        n_seq = self.last_broadcast_leader_seq.gather(1, idx_flat).view(self.num_envs, self.num_agents, candidate_k)
        n_seq = torch.where(valid, n_seq, torch.full_like(n_seq, -1))
        seq_max, argmax = n_seq.max(dim=2)

        update_mask = (seq_max >= 0) & (seq_max > self.leader_est_seq)

        n_pos = self.last_broadcast_leader_pos.gather(1, idx_flat).view(self.num_envs, self.num_agents, candidate_k)
        n_vel = self.last_broadcast_leader_vel.gather(1, idx_flat).view(self.num_envs, self.num_agents, candidate_k)

        best_pos = n_pos.gather(2, argmax.unsqueeze(-1)).squeeze(-1)
        best_vel = n_vel.gather(2, argmax.unsqueeze(-1)).squeeze(-1)

        # 计算信息更新前的 leader 估计误差（用于奖励）
        old_est_error = (
            torch.abs(self.leader_est_pos[:, 1:] - self.positions[:, 0:1]).mean(dim=1)
            + 0.01 * torch.abs(self.leader_est_vel[:, 1:] - self.velocities[:, 0:1]).mean(dim=1)
        )

        self.leader_est_pos = torch.where(update_mask, best_pos, self.leader_est_pos)
        self.leader_est_vel = torch.where(update_mask, best_vel, self.leader_est_vel)
        self.leader_est_seq = torch.where(update_mask, seq_max, self.leader_est_seq)

        # 方案(1)：更新 _since_leader_update 计数器
        # follower 部分的 update_mask: shape=(E, A)，取 [:, 1:] 得到 (E, F)
        follower_updated = update_mask[:, 1:]  # (E, F)
        # 被更新的 follower 清零，否则 +1
        self._since_leader_update = torch.where(
            follower_updated,
            torch.zeros_like(self._since_leader_update),
            self._since_leader_update + 1,
        )

        # 计算信息更新后的 leader 估计误差
        new_est_error = (
            torch.abs(self.leader_est_pos[:, 1:] - self.positions[:, 0:1]).mean(dim=1)
            + 0.01 * torch.abs(self.leader_est_vel[:, 1:] - self.velocities[:, 0:1]).mean(dim=1)
        )

        # 信息增益奖励：通信使得 leader 估计更准确
        info_gain = (old_est_error - new_est_error).clamp(min=0.0)
        info_gain_bonus = info_gain * self.info_gain_scale

        pos_error = torch.abs(self.positions[:, 1:] - self.positions[:, 0:1])
        vel_error = torch.abs(self.velocities[:, 1:] - self.velocities[:, 0:1])
        tracking_error = pos_error.mean(dim=1) + 0.01 * vel_error.mean(dim=1)

        pos_error_norm = pos_error.mean(dim=1) / self.pos_limit
        vel_error_norm = vel_error.mean(dim=1) / self.vel_limit
        tracking_error_norm = pos_error_norm + 0.01 * vel_error_norm
        tracking_penalty = -self.tracking_penalty_max * torch.log1p(tracking_error_norm * self.tracking_penalty_scale)

        improvement_bonus = torch.zeros_like(tracking_error)
        valid_prev = self._prev_error_valid
        if valid_prev.any():
            improvement = self._prev_error - tracking_error
            improvement_bonus = torch.where(
                valid_prev,
                torch.clamp(improvement * self.improvement_scale, -self.improvement_clip, self.improvement_clip),
                torch.zeros_like(improvement),
            )

        self._prev_error.copy_(tracking_error.detach())
        self._prev_error_valid[:] = True

        comm_weight = torch.exp(-tracking_error * self.comm_weight_decay)
        comm_rate = is_triggered.float().mean(dim=1)
        comm_penalty = -comm_rate * self.comm_penalty_base * comm_weight

        raw_reward = tracking_penalty + improvement_bonus + comm_penalty + info_gain_bonus
        rewards = self._scale_reward_batch(raw_reward)

        dones = torch.zeros(self.num_envs, dtype=torch.bool, device=DEVICE)

        infos = {
            "tracking_error": tracking_error,
            "comm_rate": comm_rate,
            "comm_weight": comm_weight,
            "leader_pos": self.positions[:, 0],
            "leader_vel": self.velocities[:, 0],
            "avg_follower_pos": self.positions[:, 1:].mean(dim=1),
            "theta_mean": theta.mean(),  # 改为阈值均值（原 comm_prob_mean）
            "tracking_penalty": tracking_penalty.mean(),
            "improvement_bonus": improvement_bonus.mean(),
            "comm_penalty": comm_penalty.mean(),
            "info_gain_bonus": info_gain_bonus.mean(),
            "leader_amplitude_mean": self.leader_amplitude.mean(),
            "leader_omega_mean": self.leader_omega.mean(),
        }

        return self._get_state_optimized(), rewards, dones, infos


class ModelFreeEnv:
    """单环境包装器。

    该类将 `BatchedModelFreeEnv(num_envs=1)` 封装为更直观的单环境接口：
    - `reset()` -> shape=(num_agents, STATE_DIM)
    - `step()` -> reward/done 返回标量
    """

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
        return self.batched_env.get_global_state()[0]

    def reset(self):
        state = self.batched_env.reset()
        return state[0]

    def step(self, action: torch.Tensor):
        action_batched = action.unsqueeze(0)
        states, rewards, dones, infos = self.batched_env.step(action_batched)
        info = {
            k: (v[0].item() if isinstance(v, torch.Tensor) and v.dim() > 0 else v.item() if isinstance(v, torch.Tensor) else v)
            for k, v in infos.items()
        }
        return states[0], rewards[0].item(), dones[0].item(), info
