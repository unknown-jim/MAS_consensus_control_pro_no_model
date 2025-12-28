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
    HIDDEN_DIM,
    LOCAL_OBS_DIM,
    LOG_STD_MAX,
    LOG_STD_MIN,
    MAX_NEIGHBORS,
    NEIGHBOR_FEAT_DIM,
    NEIGHBOR_OBS_DIM,
    NEIGHBOR_ROLE_DIM,
    NUM_ATTENTION_HEADS,
    NUM_FOLLOWERS,
    NUM_TRANSFORMER_LAYERS,
    SELF_ROLE_DIM,
    STATE_DIM,
    TH_SCALE,
    V_SCALE,
)


def _parse_flat_state(state: torch.Tensor):
    """把 (batch, STATE_DIM) 的 flat state 拆成结构化输入。"""

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
    """轻量级注意力模块"""

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
        """给定 action（post-squash）计算 log_prob。"""

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
# Critic / Value
# ============================================================


class CentralizedCritic(nn.Module):
    """集中式 Critic（用于 CTDE-SAC）。"""

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
        self.joint_action_dim = self.num_followers * self.action_dim

        self.state_encoder = nn.Sequential(
            nn.Linear(self.global_state_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )

        self.action_encoder = nn.Sequential(
            nn.Linear(self.joint_action_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )

        self.q_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
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
        state_feat = self.state_encoder(global_state)
        action_feat = self.action_encoder(joint_action)
        combined = torch.cat([state_feat, action_feat], dim=-1)
        return self.q_net(combined)


class CentralizedValue(nn.Module):
    """集中式 Value 网络（CTDE-MAPPO 用）。"""

    def __init__(self, global_state_dim: int = GLOBAL_STATE_DIM, hidden_dim: int = HIDDEN_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(int(global_state_dim), hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, global_state: torch.Tensor):
        return self.net(global_state)
