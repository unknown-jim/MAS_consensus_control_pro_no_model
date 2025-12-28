"""
神经网络模型 - CTDE 架构版本

CTDE = Centralized Training Decentralized Execution
- Actor: 分散式，只使用本地观测
- Critic: 集中式，使用全局状态 + 所有智能体动作
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import math

from config import (
    STATE_DIM, HIDDEN_DIM, ACTION_DIM,
    LOCAL_OBS_DIM, NEIGHBOR_OBS_DIM, MAX_NEIGHBORS,
    SELF_ROLE_DIM, NEIGHBOR_ROLE_DIM, NEIGHBOR_FEAT_DIM,
    LOG_STD_MIN, LOG_STD_MAX, V_SCALE, TH_SCALE,
    NUM_ATTENTION_HEADS, NUM_TRANSFORMER_LAYERS, DROPOUT,
    LIGHTWEIGHT_MODE, NUM_FOLLOWERS, GLOBAL_STATE_DIM
)


# ============================================================
# 注意力模块（与原版相同）
# ============================================================

class LightweightAttention(nn.Module):
    """轻量级注意力模块"""
    
    def __init__(self, dim, num_heads=2, dropout=0.05):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
            nn.Dropout(dropout)
        )
        
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
    
    def forward(self, x, mask=None):
        batch_size, seq_len, dim = x.shape
        
        residual = x
        x = self.norm1(x)
        
        qkv = self.qkv(x).reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            mask_expanded = mask.unsqueeze(1).unsqueeze(2)
            attn = attn.masked_fill(mask_expanded, float('-inf'))
        
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
    
    def __init__(self, neighbor_dim, hidden_dim, num_heads, num_layers, dropout):
        super().__init__()
        
        self.neighbor_embed = nn.Sequential(
            nn.Linear(neighbor_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim)
        )
        
        self.pos_embed = nn.Parameter(torch.randn(1, MAX_NEIGHBORS, hidden_dim) * 0.02)
        
        self.attention_layers = nn.ModuleList([
            LightweightAttention(hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
    
    def forward(self, neighbor_data, neighbor_mask=None):
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
# Actor 网络（分散式执行 - 只用本地观测）
# ============================================================

class DecentralizedActor(nn.Module):
    """分散式 Actor（用于执行阶段）。

    输入：单个智能体的本地观测
    - 自身观测 (LOCAL_OBS_DIM) + 自身角色 (SELF_ROLE_DIM)
    - 邻居数据: MAX_NEIGHBORS × (NEIGHBOR_OBS_DIM + NEIGHBOR_ROLE_DIM)
      （当前实现里 NEIGHBOR_ROLE_DIM=0，邻居广播包包含邻居自身状态 + leader 估计 seq/age）

    输出：单个智能体的动作（速度改变量 + 通信阈值）

    说明：为了支持 MAPPO/PPO，我们额外提供“给定动作求 log_prob”的能力。
    """
    
    def __init__(self, local_dim=LOCAL_OBS_DIM, role_dim=SELF_ROLE_DIM,
                 neighbor_dim=NEIGHBOR_OBS_DIM, neighbor_role_dim=NEIGHBOR_ROLE_DIM,
                 hidden_dim=HIDDEN_DIM):
        super().__init__()
        
        self.local_dim = local_dim
        self.role_dim = role_dim
        self.neighbor_obs_dim = neighbor_dim
        self.neighbor_role_dim = neighbor_role_dim
        self.neighbor_feat_dim = neighbor_dim + neighbor_role_dim  # = NEIGHBOR_FEAT_DIM
        
        # 本地状态编码（位置+速度+角色）
        self.local_encoder = nn.Sequential(
            nn.Linear(local_dim + role_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim)
        )
        
        # 邻居信息编码（位置+速度+角色）
        self.neighbor_encoder = LightweightAttentionEncoder(
            self.neighbor_feat_dim, hidden_dim, NUM_ATTENTION_HEADS, 
            NUM_TRANSFORMER_LAYERS, DROPOUT
        )
        
        # 融合层
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim)
        )
        
        # 策略头
        self.shared = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU()
        )
        
        # 速度改变量头
        self.v_mean = nn.Linear(hidden_dim // 2, 1)
        self.v_log_std = nn.Linear(hidden_dim // 2, 1)
        
        # 阈值头
        self.th_mean = nn.Linear(hidden_dim // 2, 1)
        self.th_log_std = nn.Linear(hidden_dim // 2, 1)
        
        self.v_scale = V_SCALE
        self.th_scale = TH_SCALE
        self._eps = 1e-6
        
        self._log_v_scale = torch.log(torch.tensor(self.v_scale))
        self._log_th_scale = torch.log(torch.tensor(self.th_scale))
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def compute_action_params(self, local_obs, self_role, neighbor_data, neighbor_mask=None):
        """返回动作分布参数（用于 PPO/MAPPO 评估给定动作的 log_prob）。

        Returns:
            v_mean, v_log_std, th_mean, th_log_std: shape (batch, 1)
        """
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

    def forward(self, local_obs, self_role, neighbor_data, neighbor_mask=None, deterministic=False):
        """采样动作并返回 log_prob（SAC/MAPPO 共用）。"""
        v_mean, v_log_std, th_mean, th_log_std = self.compute_action_params(
            local_obs, self_role, neighbor_data, neighbor_mask
        )

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
            ) - self._log_v_scale.to(v.device)

            log_prob_th = th_dist.log_prob(th_sample) - torch.log(
                torch.clamp(th_sigmoid * (1.0 - th_sigmoid), min=self._eps, max=0.25)
            ) - self._log_th_scale.to(th.device)

            log_prob = (log_prob_v + log_prob_th).sum(dim=-1, keepdim=True)

        action = torch.cat([v, th], dim=-1)
        return action, log_prob


# ============================================================
# Critic 网络（集中式训练 - 使用全局信息）
# ============================================================

class CentralizedCritic(nn.Module):
    """
    集中式 Critic（用于训练阶段）
    
    输入：
    - 全局状态：所有智能体的真实位置和速度
    - 联合动作：所有跟随者的动作
    
    输出：Q 值
    """
    
    def __init__(self, global_state_dim=GLOBAL_STATE_DIM, 
                 num_followers=NUM_FOLLOWERS,
                 action_dim=ACTION_DIM, 
                 hidden_dim=HIDDEN_DIM):
        super().__init__()
        
        self.global_state_dim = global_state_dim
        self.num_followers = num_followers
        self.action_dim = action_dim
        self.joint_action_dim = num_followers * action_dim
        
        # 全局状态编码器
        self.state_encoder = nn.Sequential(
            nn.Linear(global_state_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim)
        )
        
        # 联合动作编码器
        self.action_encoder = nn.Sequential(
            nn.Linear(self.joint_action_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim)
        )
        
        # Q 网络
        self.q_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, global_state, joint_action):
        """
        Args:
            global_state: (batch, global_state_dim) - 全局状态
            joint_action: (batch, num_followers * action_dim) - 联合动作
        
        Returns:
            q_value: (batch, 1)
        """
        state_feat = self.state_encoder(global_state)
        action_feat = self.action_encoder(joint_action)
        combined = torch.cat([state_feat, action_feat], dim=-1)
        return self.q_net(combined)


class CentralizedValue(nn.Module):
    """集中式 Value 网络（CTDE-MAPPO 用）。

    输入：global_state
    输出：V(global_state)
    """

    def __init__(self, global_state_dim=GLOBAL_STATE_DIM, hidden_dim=HIDDEN_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(global_state_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, global_state):
        return self.net(global_state)


# ============================================================
# 兼容旧接口的包装器
# ============================================================

class GaussianActor(nn.Module):
    """兼容旧接口的 Actor 包装器（分散式）。

    额外支持：给定动作计算 log_prob（PPO/MAPPO 需要）。
    """

    def __init__(self, state_dim=STATE_DIM, hidden_dim=HIDDEN_DIM, num_heads=4):
        super().__init__()
        self.actor = DecentralizedActor(
            LOCAL_OBS_DIM, SELF_ROLE_DIM,
            NEIGHBOR_OBS_DIM, NEIGHBOR_ROLE_DIM,
            hidden_dim
        )
        self.local_dim = LOCAL_OBS_DIM
        self.role_dim = SELF_ROLE_DIM
        self.neighbor_feat_dim = NEIGHBOR_FEAT_DIM
        self.max_neighbors = MAX_NEIGHBORS
        self._eps = 1e-6

    @staticmethod
    def _atanh(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        x = x.clamp(min=-1.0 + eps, max=1.0 - eps)
        return 0.5 * (torch.log1p(x) - torch.log1p(-x))

    @staticmethod
    def _logit(p: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        p = p.clamp(min=eps, max=1.0 - eps)
        return torch.log(p) - torch.log1p(-p)

    def _parse_state(self, state: torch.Tensor):
        local_obs = state[:, :self.local_dim]
        self_role = state[:, self.local_dim:self.local_dim + self.role_dim]
        neighbor_start = self.local_dim + self.role_dim
        neighbor_data = state[:, neighbor_start:].view(-1, self.max_neighbors, self.neighbor_feat_dim)
        neighbor_mask = (neighbor_data.abs().sum(dim=-1) == 0)
        return local_obs, self_role, neighbor_data, neighbor_mask

    def forward(self, state, edge_index=None, role_ids=None, deterministic=False):
        """采样动作并返回 log_prob（与旧接口兼容）。"""
        local_obs, self_role, neighbor_data, neighbor_mask = self._parse_state(state)
        action, log_prob = self.actor(local_obs, self_role, neighbor_data, neighbor_mask, deterministic)
        return action, log_prob, None

    def evaluate_actions(self, state: torch.Tensor, action: torch.Tensor):
        """给定动作计算 log_prob。

        Args:
            state: (batch, state_dim)
            action: (batch, ACTION_DIM)

        Returns:
            log_prob: (batch, 1)
        """
        local_obs, self_role, neighbor_data, neighbor_mask = self._parse_state(state)

        v_mean, v_log_std, th_mean, th_log_std = self.actor.compute_action_params(
            local_obs, self_role, neighbor_data, neighbor_mask
        )
        v_std = torch.exp(v_log_std)
        th_std = torch.exp(th_log_std)

        # 反解 pre-squash 变量
        v = action[:, 0:1]
        th = action[:, 1:2]

        v_tanh = (v / float(self.actor.v_scale)).clamp(-1.0 + self._eps, 1.0 - self._eps)
        th_sigmoid = (th / float(self.actor.th_scale)).clamp(self._eps, 1.0 - self._eps)

        v_pre = self._atanh(v_tanh, eps=self._eps)
        th_pre = self._logit(th_sigmoid, eps=self._eps)

        v_dist = Normal(v_mean, v_std)
        th_dist = Normal(th_mean, th_std)

        log_prob_v = v_dist.log_prob(v_pre) - torch.log(
            torch.clamp(1.0 - v_tanh.pow(2), min=self._eps, max=1.0)
        ) - self.actor._log_v_scale.to(v.device)

        log_prob_th = th_dist.log_prob(th_pre) - torch.log(
            torch.clamp(th_sigmoid * (1.0 - th_sigmoid), min=self._eps, max=0.25)
        ) - self.actor._log_th_scale.to(th.device)

        log_prob = (log_prob_v + log_prob_th).sum(dim=-1, keepdim=True)
        return log_prob


class SoftQNetwork(nn.Module):
    """兼容旧接口的 Critic 包装器（集中式）"""

    def __init__(self, state_dim=STATE_DIM, hidden_dim=HIDDEN_DIM,
                 action_dim=ACTION_DIM, num_heads=4):
        super().__init__()
        # 🔧 使用传入的 state_dim（即 global_state_dim），避免“config 变化但这里没跟上”
        self.critic = CentralizedCritic(state_dim, NUM_FOLLOWERS, action_dim, hidden_dim)

    def forward(self, global_state, joint_action):
        return self.critic(global_state, joint_action)


class ValueNetwork(nn.Module):
    """集中式 Value 网络包装器（CTDE-MAPPO 用）。"""

    def __init__(self, state_dim=GLOBAL_STATE_DIM, hidden_dim=HIDDEN_DIM):
        super().__init__()
        self.value = CentralizedValue(state_dim, hidden_dim)

    def forward(self, global_state):
        return self.value(global_state)
