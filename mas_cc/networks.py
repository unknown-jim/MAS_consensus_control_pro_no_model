"""神经网络模型（CTDE 版本）。

CTDE = Centralized Training Decentralized Execution
- Actor: 分散式，只使用本地观测（flat state）
- Critic/Value: 集中式，使用全局状态 +（可选）联合动作
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from .config import (
    ACTION_DIM,
    DROPOUT,
    GLOBAL_STATE_DIM,
    GLOBAL_STATE_INCLUDE_BROADCAST,
    GLOBAL_STATE_INCLUDE_LEADER_PARAMS,
    GLOBAL_STATE_INCLUDE_TIME,
    GLOBAL_STATE_INCLUDE_TRAJ_TYPE,
    HIDDEN_DIM,
    LEADER_TRAJECTORY_TYPES,
    LOCAL_OBS_DIM,
    LOG_STD_MAX,
    LOG_STD_MIN,
    MAX_NEIGHBORS,
    NEIGHBOR_FEAT_DIM,
    NEIGHBOR_OBS_DIM,
    NEIGHBOR_ROLE_DIM,
    NUM_AGENTS,
    NUM_ATTENTION_HEADS,
    NUM_FOLLOWERS,
    NUM_TRANSFORMER_LAYERS,
    SELF_ROLE_DIM,
    STATE_DIM,
    TH_SCALE,
    V_SCALE,
)


def _parse_flat_state(state: torch.Tensor):
    """把 flat state 拆成结构化输入。

    Args:
        state: shape=(B, STATE_DIM) 的 flat 状态。

    Returns:
        local_obs: shape=(B, LOCAL_OBS_DIM)
        self_role: shape=(B, SELF_ROLE_DIM)
        neighbor_data: shape=(B, MAX_NEIGHBORS, NEIGHBOR_FEAT_DIM)
        neighbor_mask: shape=(B, MAX_NEIGHBORS)；True 表示该邻居槽位为空（需要 mask）。

    Raises:
        ValueError: 当输入维度不匹配时抛出。
    """

    if state.dim() != 2:
        raise ValueError(f"Expected state dim=2, got shape={tuple(state.shape)}")
    if state.shape[1] != int(STATE_DIM):
        raise ValueError(f"Expected state_dim={STATE_DIM}, got {state.shape[1]}")

    local_obs = state[:, :LOCAL_OBS_DIM]
    self_role = state[:, LOCAL_OBS_DIM:LOCAL_OBS_DIM + SELF_ROLE_DIM]
    neighbor_start = LOCAL_OBS_DIM + SELF_ROLE_DIM

    neighbor_data = state[:, neighbor_start:].view(-1, MAX_NEIGHBORS, NEIGHBOR_FEAT_DIM)
    neighbor_mask = (neighbor_data.abs().sum(dim=-1) == 0)
    return local_obs, self_role, neighbor_data, neighbor_mask


def _atanh(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    x = x.clamp(min=-1.0 + eps, max=1.0 - eps)
    return 0.5 * (torch.log1p(x) - torch.log1p(-x))


def _logit(p: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    p = p.clamp(min=eps, max=1.0 - eps)
    return torch.log(p) - torch.log1p(-p)


# ============================================================
# 注意力模块
# ============================================================


class LightweightAttention(nn.Module):
    """轻量级自注意力模块。

    用于对邻居序列特征做编码；支持传入 mask 把空邻居槽位从 attention 中屏蔽掉。

    Args:
        dim: 输入/输出特征维度。
        num_heads: multi-head 数量。
        dropout: dropout 概率。
    """

    def __init__(self, dim: int, num_heads: int = 2, dropout: float = 0.05):
        super().__init__()
        self.num_heads = int(num_heads)
        self.head_dim = dim // self.num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
            nn.Dropout(dropout),
        )

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None):
        batch_size, seq_len, dim = x.shape

        residual = x
        x = self.norm1(x)

        qkv = self.qkv(x).reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if mask is not None:
            mask_expanded = mask.unsqueeze(1).unsqueeze(2)
            attn = attn.masked_fill(mask_expanded, float("-inf"))

        attn = F.softmax(attn, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(batch_size, seq_len, dim)
        out = self.proj(out)

        x = residual + out
        x = x + self.ffn(self.norm2(x))
        return x


class LightweightAttentionEncoder(nn.Module):
    """轻量版注意力编码器"""

    def __init__(self, neighbor_dim: int, hidden_dim: int, num_heads: int, num_layers: int, dropout: float):
        super().__init__()

        self.neighbor_embed = nn.Sequential(
            nn.Linear(neighbor_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )

        self.pos_embed = nn.Parameter(torch.randn(1, MAX_NEIGHBORS, hidden_dim) * 0.02)

        self.attention_layers = nn.ModuleList(
            [LightweightAttention(hidden_dim, num_heads, dropout) for _ in range(int(num_layers))]
        )

        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, neighbor_data: torch.Tensor, neighbor_mask: torch.Tensor | None = None):
        h = self.neighbor_embed(neighbor_data)
        h = h + self.pos_embed

        for layer in self.attention_layers:
            h = layer(h, neighbor_mask)

        if neighbor_mask is not None:
            valid_mask = ~neighbor_mask
            valid_count = valid_mask.sum(dim=1, keepdim=True).clamp(min=1).unsqueeze(-1)
            h = (h * valid_mask.unsqueeze(-1)).sum(dim=1) / valid_count.squeeze(-1)
        else:
            h = h.mean(dim=1)

        return self.output_proj(h)


# ============================================================
# Actor（flat state -> action, log_prob）
# ============================================================


class DecentralizedActor(nn.Module):
    """分散式 Actor（用于执行阶段）。

    输入：单个智能体的 flat state（shape=(B, STATE_DIM)）
    输出：动作（shape=(B, ACTION_DIM)）以及 log_prob（shape=(B, 1)；deterministic 时为 None）。

    同时提供 `evaluate_actions(state, action)`：给定动作求 log_prob（PPO/MAPPO 需要）。
    """

    def __init__(
        self,
        local_dim: int = LOCAL_OBS_DIM,
        role_dim: int = SELF_ROLE_DIM,
        neighbor_dim: int = NEIGHBOR_OBS_DIM,
        neighbor_role_dim: int = NEIGHBOR_ROLE_DIM,
        hidden_dim: int = HIDDEN_DIM,
    ):
        super().__init__()

        self.local_dim = int(local_dim)
        self.role_dim = int(role_dim)
        self.neighbor_obs_dim = int(neighbor_dim)
        self.neighbor_role_dim = int(neighbor_role_dim)
        self.neighbor_feat_dim = self.neighbor_obs_dim + self.neighbor_role_dim

        self.local_encoder = nn.Sequential(
            nn.Linear(self.local_dim + self.role_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )

        self.neighbor_encoder = LightweightAttentionEncoder(
            self.neighbor_feat_dim, hidden_dim, NUM_ATTENTION_HEADS, NUM_TRANSFORMER_LAYERS, DROPOUT
        )

        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )

        self.shared = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
        )

        self.v_mean = nn.Linear(hidden_dim // 2, 1)
        self.v_log_std = nn.Linear(hidden_dim // 2, 1)

        self.th_mean = nn.Linear(hidden_dim // 2, 1)
        self.th_log_std = nn.Linear(hidden_dim // 2, 1)

        self.v_scale = float(V_SCALE)
        self.th_scale = float(TH_SCALE)
        self._eps = 1e-6

        self.register_buffer("_log_v_scale", torch.log(torch.tensor(self.v_scale)), persistent=False)
        self.register_buffer("_log_th_scale", torch.log(torch.tensor(self.th_scale)), persistent=False)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def compute_action_params(self, state: torch.Tensor):
        local_obs, self_role, neighbor_data, neighbor_mask = _parse_flat_state(state)

        local_with_role = torch.cat([local_obs, self_role], dim=-1)
        local_feat = self.local_encoder(local_with_role)

        neighbor_feat = self.neighbor_encoder(neighbor_data, neighbor_mask)
        combined = torch.cat([local_feat, neighbor_feat], dim=-1)
        hidden = self.fusion(combined)

        shared_feat = self.shared(hidden)

        v_mean = self.v_mean(shared_feat)
        v_log_std = torch.clamp(self.v_log_std(shared_feat), LOG_STD_MIN, LOG_STD_MAX)

        th_mean = self.th_mean(shared_feat)
        th_log_std = torch.clamp(self.th_log_std(shared_feat), LOG_STD_MIN, LOG_STD_MAX)

        return v_mean, v_log_std, th_mean, th_log_std

    def forward(self, state: torch.Tensor, deterministic: bool = False):
        v_mean, v_log_std, th_mean, th_log_std = self.compute_action_params(state)

        v_std = torch.exp(v_log_std)
        th_std = torch.exp(th_log_std)

        if deterministic:
            v = torch.tanh(v_mean) * self.v_scale
            th = torch.sigmoid(th_mean) * self.th_scale
            log_prob = None
        else:
            v_dist = Normal(v_mean, v_std)
            th_dist = Normal(th_mean, th_std)

            v_sample = v_dist.rsample()
            th_sample = th_dist.rsample()

            v_tanh = torch.tanh(v_sample)
            v = v_tanh * self.v_scale

            th_sigmoid = torch.sigmoid(th_sample)
            th = th_sigmoid * self.th_scale

            log_prob_v = v_dist.log_prob(v_sample) - torch.log(
                torch.clamp(1.0 - v_tanh.pow(2), min=self._eps, max=1.0)
            ) - self._log_v_scale.to(v_sample.device)

            log_prob_th = th_dist.log_prob(th_sample) - torch.log(
                torch.clamp(th_sigmoid * (1.0 - th_sigmoid), min=self._eps, max=0.25)
            ) - self._log_th_scale.to(th_sample.device)

            log_prob = (log_prob_v + log_prob_th).sum(dim=-1, keepdim=True)

        action = torch.cat([v, th], dim=-1)
        return action, log_prob

    def evaluate_actions(self, state: torch.Tensor, action: torch.Tensor):
        """给定动作计算 log_prob（用于 PPO/MAPPO）。

        Args:
            state: shape=(B, STATE_DIM)。
            action: shape=(B, ACTION_DIM)，为 post-squash 空间动作（与环境交互的动作）。

        Returns:
            log_prob: shape=(B, 1)
        """

        v_mean, v_log_std, th_mean, th_log_std = self.compute_action_params(state)
        v_std = torch.exp(v_log_std)
        th_std = torch.exp(th_log_std)

        v = action[:, 0:1]
        th = action[:, 1:2]

        v_tanh = (v / self.v_scale).clamp(-1.0 + self._eps, 1.0 - self._eps)
        th_sigmoid = (th / self.th_scale).clamp(self._eps, 1.0 - self._eps)

        v_pre = _atanh(v_tanh, eps=self._eps)
        th_pre = _logit(th_sigmoid, eps=self._eps)

        v_dist = Normal(v_mean, v_std)
        th_dist = Normal(th_mean, th_std)

        log_prob_v = v_dist.log_prob(v_pre) - torch.log(
            torch.clamp(1.0 - v_tanh.pow(2), min=self._eps, max=1.0)
        ) - self._log_v_scale.to(v_pre.device)

        log_prob_th = th_dist.log_prob(th_pre) - torch.log(
            torch.clamp(th_sigmoid * (1.0 - th_sigmoid), min=self._eps, max=0.25)
        ) - self._log_th_scale.to(th_pre.device)

        log_prob = (log_prob_v + log_prob_th).sum(dim=-1, keepdim=True)
        return log_prob


# ============================================================
# Critic / Value（Set-Invariant Encoder，适配规模扩展）
# ============================================================


def _split_global_state(global_state: torch.Tensor):
    """把 flat `global_state` 拆成：agent 级特征序列 + 全局上下文。

    该拆分必须与 `environment.py::get_global_state()` 的拼接顺序严格一致。

    Returns:
        agent_feats: (B, NUM_AGENTS, agent_feat_dim)
        global_ctx: (B, ctx_dim)；可能为 0 维
    """

    if global_state.dim() != 2:
        raise ValueError(f"Expected global_state dim=2, got shape={tuple(global_state.shape)}")

    B = int(global_state.shape[0])
    N = int(NUM_AGENTS)

    offset = 0
    pos = global_state[:, offset:offset + N]
    offset += N
    vel = global_state[:, offset:offset + N]
    offset += N

    components = [pos, vel]

    if bool(GLOBAL_STATE_INCLUDE_BROADCAST):
        b_pos = global_state[:, offset:offset + N]
        offset += N
        b_vel = global_state[:, offset:offset + N]
        offset += N
        components.extend([b_pos, b_vel])

    agent_feats = torch.stack(components, dim=-1)  # (B, N, 2/4)

    # 剩余部分作为全局上下文（leader params / traj onehot / time）
    global_ctx = global_state[:, offset:]
    if global_ctx.numel() == 0:
        global_ctx = global_state.new_zeros((B, 0))

    return agent_feats, global_ctx


class DeepSetsEncoder(nn.Module):
    """DeepSets 风格集合编码器：对可变数量实体做置换不变的 pooling。"""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int | None = None, dropout: float = 0.0):
        super().__init__()
        out_dim = int(out_dim) if out_dim is not None else int(hidden_dim)

        self.phi = nn.Sequential(
            nn.Linear(int(in_dim), hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )

        self.rho = nn.Sequential(
            nn.Linear(hidden_dim, out_dim),
            nn.GELU(),
            nn.LayerNorm(out_dim),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None):
        """Args:
        x: (B, M, D)
        mask: (B, M)；True 表示该实体无效（会被排除）
        """
        h = self.phi(x)

        if mask is not None:
            valid = (~mask).to(dtype=h.dtype).unsqueeze(-1)
            h = h * valid
            denom = valid.sum(dim=1).clamp(min=1.0)
            pooled = h.sum(dim=1) / denom
        else:
            pooled = h.mean(dim=1)

        return self.rho(pooled)


class CentralizedCritic(nn.Module):
    """集中式 Q 网络（用于 CTDE-SAC / 可扩展到 DDPG）。

    与旧实现相比：
    - 不再直接对 `(global_state, joint_action)` 做大向量拼接 MLP
    - 而是：
        1) 将 `global_state` 拆成 agent 级特征序列，DeepSets pooling 得到固定维度表示
        2) 将 `joint_action` 拆成 follower 动作序列，DeepSets pooling 得到固定维度表示
        3) 拼接（state_set, action_set, global_ctx）后输出 Q

    这样输入维度对 `NUM_AGENTS/NUM_FOLLOWERS` 不敏感，适合规模扩展。
    """

    def __init__(
        self,
        global_state_dim: int = GLOBAL_STATE_DIM,
        num_followers: int = NUM_FOLLOWERS,
        action_dim: int = ACTION_DIM,
        hidden_dim: int = HIDDEN_DIM,
    ):
        super().__init__()

        self.global_state_dim = int(global_state_dim)
        self.num_followers = int(num_followers)
        self.action_dim = int(action_dim)

        # agent 特征维度与 get_global_state() 保持一致：pos/vel (+ broadcast pos/vel)
        self.agent_feat_dim = 2 + (2 if bool(GLOBAL_STATE_INCLUDE_BROADCAST) else 0)

        # 全局上下文维度：leader params / traj type / time
        ctx_dim = 0
        if bool(GLOBAL_STATE_INCLUDE_LEADER_PARAMS):
            ctx_dim += 3
        if bool(GLOBAL_STATE_INCLUDE_TRAJ_TYPE):
            ctx_dim += len(LEADER_TRAJECTORY_TYPES)
        if bool(GLOBAL_STATE_INCLUDE_TIME):
            ctx_dim += 1
        self.ctx_dim = int(ctx_dim)

        self.state_set_encoder = DeepSetsEncoder(self.agent_feat_dim, hidden_dim, out_dim=hidden_dim, dropout=DROPOUT)
        self.action_set_encoder = DeepSetsEncoder(self.action_dim, hidden_dim, out_dim=hidden_dim, dropout=DROPOUT)

        self.ctx_encoder = (
            nn.Sequential(
                nn.Linear(self.ctx_dim, hidden_dim),
                nn.GELU(),
                nn.LayerNorm(hidden_dim),
            )
            if self.ctx_dim > 0
            else nn.Identity()
        )

        self.q_net = nn.Sequential(
            nn.Linear(hidden_dim * (3 if self.ctx_dim > 0 else 2), hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, global_state: torch.Tensor, joint_action: torch.Tensor):
        # 兼容 `@torch.inference_mode()` 产生的 inference tensor（避免反向保存时报错）
        if global_state.is_inference():
            global_state = global_state.clone()
        if joint_action.is_inference():
            joint_action = joint_action.clone()

        if global_state.shape[1] != int(self.global_state_dim):
            raise ValueError(f"Expected global_state_dim={self.global_state_dim}, got {global_state.shape[1]}")

        if joint_action.dim() != 2:
            raise ValueError(f"Expected joint_action dim=2, got shape={tuple(joint_action.shape)}")
        expected_joint = int(self.num_followers * self.action_dim)
        if joint_action.shape[1] != expected_joint:
            raise ValueError(f"Expected joint_action_dim={expected_joint}, got {joint_action.shape[1]}")

        agent_feats, global_ctx = _split_global_state(global_state)
        z_s = self.state_set_encoder(agent_feats)

        act_seq = joint_action.view(-1, self.num_followers, self.action_dim)
        z_a = self.action_set_encoder(act_seq)

        if self.ctx_dim > 0:
            z_c = self.ctx_encoder(global_ctx)
            z = torch.cat([z_s, z_a, z_c], dim=-1)
        else:
            z = torch.cat([z_s, z_a], dim=-1)

        return self.q_net(z)


class CentralizedValue(nn.Module):
    """集中式 Value 网络（CTDE-MAPPO 用）。

    同样使用集合编码器把可变规模的 agent 状态压成固定维度表示。
    """

    def __init__(self, global_state_dim: int = GLOBAL_STATE_DIM, hidden_dim: int = HIDDEN_DIM):
        super().__init__()

        self.global_state_dim = int(global_state_dim)
        self.agent_feat_dim = 2 + (2 if bool(GLOBAL_STATE_INCLUDE_BROADCAST) else 0)

        ctx_dim = 0
        if bool(GLOBAL_STATE_INCLUDE_LEADER_PARAMS):
            ctx_dim += 3
        if bool(GLOBAL_STATE_INCLUDE_TRAJ_TYPE):
            ctx_dim += len(LEADER_TRAJECTORY_TYPES)
        if bool(GLOBAL_STATE_INCLUDE_TIME):
            ctx_dim += 1
        self.ctx_dim = int(ctx_dim)

        self.state_set_encoder = DeepSetsEncoder(self.agent_feat_dim, hidden_dim, out_dim=hidden_dim, dropout=DROPOUT)

        self.ctx_encoder = (
            nn.Sequential(
                nn.Linear(self.ctx_dim, hidden_dim),
                nn.GELU(),
                nn.LayerNorm(hidden_dim),
            )
            if self.ctx_dim > 0
            else nn.Identity()
        )

        self.v_net = nn.Sequential(
            nn.Linear(hidden_dim * (2 if self.ctx_dim > 0 else 1), hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, global_state: torch.Tensor):
        if global_state.is_inference():
            global_state = global_state.clone()

        if global_state.shape[1] != int(self.global_state_dim):
            raise ValueError(f"Expected global_state_dim={self.global_state_dim}, got {global_state.shape[1]}")

        agent_feats, global_ctx = _split_global_state(global_state)
        z_s = self.state_set_encoder(agent_feats)

        if self.ctx_dim > 0:
            z_c = self.ctx_encoder(global_ctx)
            z = torch.cat([z_s, z_c], dim=-1)
        else:
            z = z_s

        return self.v_net(z)
