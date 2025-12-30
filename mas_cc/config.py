"""配置文件（CTDE 版本）。

输出目录约定（统一管理训练产物）：
- 根目录：OUTPUT_ROOT（默认 results/）
- 按算法/日期分层：results/<algo>/YYYYMMDD/HHMMSS/
- 模型：.../models/
- 图片：.../figs/

可通过环境变量覆盖：
- RUN_DIR：直接指定本次运行输出目录（最高优先级）
- OUTPUT_ROOT：指定根目录（默认 results）

本项目不再包含任何“旧版本/旧接口”的兼容分支：
- 设备选择仅采用标准的 CUDA 可用性判定
- 训练/可视化依赖按“强依赖”处理（缺失直接报错）
"""

from __future__ import annotations

import os
import random
from datetime import datetime

import numpy as np
import torch

# ==================== 设备配置 ====================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==================== 随机种子 ====================
SEED = 42
TOPOLOGY_SEED = 42

# ==================== 输出目录（训练/评估产物统一落盘）====================
OUTPUT_ROOT = os.getenv("OUTPUT_ROOT", "results")

_RUN_DATE = datetime.now().strftime("%Y%m%d")
_RUN_TIME = datetime.now().strftime("%H%M%S")

RUN_DIR = os.getenv("RUN_DIR", "").strip()
MODELS_DIR = ""
FIGS_DIR = ""


def ensure_dir(path: str) -> str:
    """确保目录存在。

    Args:
        path: 目录路径。

    Returns:
        原样返回 `path`，便于在配置中链式使用。
    """
    os.makedirs(path, exist_ok=True)
    return path


def ensure_parent_dir(file_path: str) -> str:
    """确保文件的父目录存在。

    Args:
        file_path: 文件路径。

    Returns:
        原样返回 `file_path`。
    """
    parent = os.path.dirname(file_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    return file_path


# ==================== 模式选择 ====================
LIGHTWEIGHT_MODE = True

# ==================== 算法选择 ====================
# 可选："MASAC"（CTDE-SAC） / "MAPPO"（CTDE-MAPPO）
ALGO = "MAPPO"

# ==================== 输出目录（按算法隔离）====================
_ALGO_TAG = str(ALGO).lower().strip() if str(ALGO).strip() else "unknown"
_RUN_DIR_DEFAULT = os.path.join(OUTPUT_ROOT, _ALGO_TAG, _RUN_DATE, _RUN_TIME)
if not RUN_DIR:
    RUN_DIR = _RUN_DIR_DEFAULT
MODELS_DIR = os.path.join(RUN_DIR, "models")
FIGS_DIR = os.path.join(RUN_DIR, "figs")

# ==================== 网络拓扑 ====================
NUM_FOLLOWERS = 20
NUM_PINNED = 5
NUM_AGENTS = NUM_FOLLOWERS + 1

# ==================== 无模型状态空间 ====================
SELF_STATE_DIM = 2
SELF_LEADER_DIM = 4
LOCAL_OBS_DIM = SELF_STATE_DIM + SELF_LEADER_DIM  # 6

SELF_ROLE_DIM = 3

NEIGHBOR_STATE_DIM = 2
NEIGHBOR_LEADER_DIM = 4
NEIGHBOR_OBS_DIM = NEIGHBOR_STATE_DIM + NEIGHBOR_LEADER_DIM  # 6
NEIGHBOR_ROLE_DIM = 0

# Actor 观测里的 Top-K 邻居槽位数（固定维度，不随智能体数量增长）
#
# 可通过环境变量覆盖（在导入本模块前设置）：
# - MAX_NEIGHBORS=12 python train.py
# - TOP_K=12 python train.py  （兼容别名）
_MAX_NEIGHBORS_DEFAULT = 4
_max_neighbors_env = os.getenv("MAX_NEIGHBORS", "").strip() or os.getenv("TOP_K", "").strip()
MAX_NEIGHBORS = int(_max_neighbors_env) if _max_neighbors_env else int(_MAX_NEIGHBORS_DEFAULT)
if MAX_NEIGHBORS <= 0:
    raise ValueError(f"MAX_NEIGHBORS must be positive, got {MAX_NEIGHBORS}")

NEIGHBOR_FEAT_DIM = NEIGHBOR_OBS_DIM + NEIGHBOR_ROLE_DIM  # 6

STATE_DIM = LOCAL_OBS_DIM + SELF_ROLE_DIM + MAX_NEIGHBORS * NEIGHBOR_FEAT_DIM

# ==================== CTDE 全局状态维度 ====================
GLOBAL_STATE_INCLUDE_BROADCAST = True
GLOBAL_STATE_INCLUDE_LEADER_PARAMS = True
GLOBAL_STATE_INCLUDE_TRAJ_TYPE = True
GLOBAL_STATE_INCLUDE_TIME = True

GLOBAL_STATE_DIM_BASE = NUM_AGENTS * 2
GLOBAL_STATE_DIM = GLOBAL_STATE_DIM_BASE

# ==================== 动作空间 ====================
ACTION_DIM = 2

# ==================== 环境参数 ====================
DT = 0.05
MAX_STEPS = 300

# ==================== 领导者动力学参数（基准值）====================
LEADER_AMPLITUDE = 2.0
LEADER_OMEGA = 0.5
LEADER_PHASE = 0.0

# ==================== 随机初始化参数 ====================
RANDOMIZE_LEADER = True
RANDOMIZE_FOLLOWER = True
RANDOMIZE_TOPOLOGY = True

NUM_PINNED_RANGE = (2, 5)
EXTRA_EDGE_PROB = 0.15

LEADER_AMPLITUDE_RANGE = (1.0, 3.0)
LEADER_OMEGA_RANGE = (0.3, 0.8)
LEADER_PHASE_RANGE = (0.0, 2 * 3.14159)

LEADER_TRAJECTORY_TYPES = ["sine", "cosine"]

# finalize global state dim
if GLOBAL_STATE_INCLUDE_BROADCAST:
    GLOBAL_STATE_DIM += NUM_AGENTS * 2
if GLOBAL_STATE_INCLUDE_LEADER_PARAMS:
    GLOBAL_STATE_DIM += 3
if GLOBAL_STATE_INCLUDE_TRAJ_TYPE:
    GLOBAL_STATE_DIM += len(LEADER_TRAJECTORY_TYPES)
if GLOBAL_STATE_INCLUDE_TIME:
    GLOBAL_STATE_DIM += 1

FOLLOWER_INIT_POS_STD_RANGE = (0.4, 1.2)
FOLLOWER_INIT_VEL_STD_RANGE = (0.15, 0.5)

FOLLOWER_INIT_POS_STD = 0.5
FOLLOWER_INIT_VEL_STD = 0.2

POS_LIMIT = 10.0
VEL_LIMIT = 10.0

COMM_RANGE = 5.0

# ==================== 通信参数 ====================
# 事件触发阈值范围：当 |pos - last_broadcast_pos| > threshold 时触发通信
# 注意：每步最大位置变化 = VEL_LIMIT × DT = 0.5
# 阈值应该 < 0.5 才能有效触发通信
COMM_PENALTY = 0.15  # 通信惩罚（增大以鼓励节省通信）
THRESHOLD_MIN = 0.05  # 最小阈值（高通信率）- 提高下限，避免过于敏感
THRESHOLD_MAX = 0.5   # 最大阈值（低通信率）- 提高上限，允许更低通信率

# ==================== 确定性事件触发（ETC）参数 ====================
# 触发条件：delta = |x - x_b| + ETC_VEL_COEF * DT * |v - v_b| > theta
ETC_VEL_COEF = 0.5  # 速度增量映射系数（降低，减少速度对 delta 的贡献）

# 新鲜度保底触发：当 leader_age > AGE_MAX_STEPS 时强制触发（即便 delta <= theta）
# 这对"低误差"目标非常关键：防止 gossip 链路不畅时 leader 估计过旧
AGE_MAX_STEPS = 10  # 放宽保底触发（原5太紧，限制了策略探索空间）

# 去抖/最小触发间隔：触发后 COOLDOWN_STEPS 步内不再触发
# 防止阈值边界来回跨越导致频繁开关
COOLDOWN_STEPS = 1  # 降低 cooldown（原2太紧，锁死通信率上限在33%）

# ==================== 奖励参数 ====================
TRACKING_PENALTY_SCALE = 2.0
TRACKING_PENALTY_MAX = 1.0
COMM_WEIGHT_DECAY = 0.8  # 降低衰减（原1.5太大），让通信惩罚在低误差时更显著
IMPROVEMENT_SCALE = 1.5
IMPROVEMENT_CLIP = 0.3
INFO_GAIN_SCALE = 0.3  # 信息增益奖励系数（降低，避免过度鼓励通信）

REWARD_MIN = -2.0
REWARD_MAX = 2.0
USE_SOFT_REWARD_SCALING = True

# ==================== Dashboard 显示阈值 ====================
DASH_ERROR_GOOD_FRAC = 0.05
DASH_ERROR_POOR_FRAC = 0.20
DASH_COMM_GOOD_THRESHOLD = 0.30
DASH_COMM_POOR_THRESHOLD = 0.70

# ==================== 网络参数 ====================
if LIGHTWEIGHT_MODE:
    HIDDEN_DIM = 192
    NUM_ATTENTION_HEADS = 4
    NUM_TRANSFORMER_LAYERS = 2
    DROPOUT = 0.05

    BATCH_SIZE = 256
    NUM_EPISODES = 1200
    NUM_PARALLEL_ENVS = 48

    UPDATE_FREQUENCY = 2
    GRADIENT_STEPS = 1
else:
    HIDDEN_DIM = 512
    NUM_ATTENTION_HEADS = 8
    NUM_TRANSFORMER_LAYERS = 3
    DROPOUT = 0.1

    BATCH_SIZE = 1024
    NUM_EPISODES = 2000
    NUM_PARALLEL_ENVS = 16

    UPDATE_FREQUENCY = 4
    GRADIENT_STEPS = 4

# ==================== SAC 参数 ====================
LEARNING_RATE = 3e-4
ACTOR_LR = 3e-4
CRITIC_LR = 3e-4
ALPHA_LR = 3e-4

# ==================== PPO/MAPPO 参数 ====================
PPO_LR = 1e-4  # 降低学习率（原 3e-4），提高稳定性
PPO_CLIP_EPS = 0.15  # 减小裁剪范围（原 0.2），限制策略更新幅度
PPO_EPOCHS = 4
PPO_ROLLOUT_STEPS = 128
PPO_MINIBATCH_SIZE = 1024
PPO_GAE_LAMBDA = 0.95
PPO_VALUE_COEF = 0.5
PPO_ENTROPY_COEF = 0.05  # 大幅增加熵系数，防止 Beta 分布熵崩溃
PPO_MAX_GRAD_NORM = 0.5  # 减小梯度裁剪（原 1.0），提高稳定性
PPO_TARGET_KL = 0.015  # 减小目标 KL（原 0.02），更保守的更新

GAMMA = 0.99
TAU = 0.005

INIT_ALPHA = 0.2
AUTO_ALPHA = True
TARGET_ENTROPY_RATIO = 0.4

BUFFER_SIZE = 1_000_000

# ==================== Replay Buffer 存储设置 ====================
REPLAY_BUFFER_DEVICE = torch.device("cpu") if DEVICE.type == "cuda" else DEVICE
REPLAY_BUFFER_DTYPE = torch.float16 if DEVICE.type == "cuda" else torch.float32
REPLAY_BUFFER_PIN_MEMORY = DEVICE.type == "cuda"

# ==================== 动作缩放 ====================
LOG_STD_MIN = -20
LOG_STD_MAX = 2
V_SCALE = 1.0
TH_SCALE = 1.0

# ==================== 训练参数 ====================
VIS_INTERVAL = 10
USE_AMP = True
WARMUP_STEPS = 3000

POLICY_DELAY = 2
TARGET_UPDATE_INTERVAL = 2

_SAVE_TAG = "mappo" if str(ALGO).upper().strip() == "MAPPO" else "masac"

SAVE_MODEL_PATH = os.path.join(
    MODELS_DIR,
    (f"best_model_ctde_14f_{_SAVE_TAG}_light.pt" if LIGHTWEIGHT_MODE else f"best_model_ctde_14f_{_SAVE_TAG}.pt"),
)

EVAL_NUM_TESTS = 3
EVAL_SAVE_PATH = os.path.join(FIGS_DIR, f"final_evaluation_ctde_{_SAVE_TAG}.png")

GENERALIZATION_TEST_STEPS = MAX_STEPS * 2
GENERALIZATION_SAVE_PATH = os.path.join(FIGS_DIR, f"generalization_test_ctde_{_SAVE_TAG}.png")

GENERALIZATION_INCLUDE_OOD = True
GENERALIZATION_OOD_AMPLITUDE = LEADER_AMPLITUDE
GENERALIZATION_OOD_OMEGA = LEADER_OMEGA_RANGE[1] * 1.25


def set_seed(seed: int = SEED) -> None:
    """设置随机种子。

    Args:
        seed: 随机种子。

    Notes:
        若当前设备为 CUDA，会额外设置 `torch.cuda.manual_seed(_all)`。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if DEVICE.type == "cuda":
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def print_config() -> None:
    """打印当前关键配置（用于实验可复现与排查）。"""
    mode_str = "Lightweight" if LIGHTWEIGHT_MODE else "Full"
    print("=" * 70)
    print(f"🔧 Configuration - CTDE Architecture ({mode_str} Mode) - Large Scale")
    print(f"  Algorithm: {ALGO}")
    print("=" * 70)
    print(f"  Device: {DEVICE}")
    print(f"  Seed: {SEED}")
    print(f"  🌐 Large-Scale MAS: {NUM_AGENTS} Agents (1 Leader + {NUM_FOLLOWERS} Followers)")
    print(f"  Episodes: {NUM_EPISODES}, Max Steps: {MAX_STEPS}")
    print(f"  📡 CTDE Settings:")
    print(f"     Local Obs Dim: {LOCAL_OBS_DIM} (self pos/vel + leader_est pos/vel + leader_seq/age)")
    print(f"     Self Role Dim: {SELF_ROLE_DIM} (one-hot: leader/pinned/normal)")
    print(f"     Neighbor Feat Dim: {NEIGHBOR_FEAT_DIM} (neighbor pos/vel + carried leader_est + leader_seq/age)")
    print(f"     Global State Dim: {GLOBAL_STATE_DIM}")
    print(f"       - include_broadcast: {GLOBAL_STATE_INCLUDE_BROADCAST}")
    print(f"       - include_leader_params: {GLOBAL_STATE_INCLUDE_LEADER_PARAMS}")
    print(f"       - include_traj_type: {GLOBAL_STATE_INCLUDE_TRAJ_TYPE}")
    print(f"       - include_time: {GLOBAL_STATE_INCLUDE_TIME}")
    print(f"     Max Neighbors: {MAX_NEIGHBORS}")
    print(f"     Actor Input: Local State ({STATE_DIM} = {LOCAL_OBS_DIM} + {SELF_ROLE_DIM} + {MAX_NEIGHBORS}×{NEIGHBOR_FEAT_DIM})")
    print(f"     Critic Input: Global State ({GLOBAL_STATE_DIM}) + Joint Action ({NUM_FOLLOWERS * ACTION_DIM})")
    print(f"  🎭 Role Encoding:")
    print(f"     [1,0,0] = Leader")
    print(f"     [0,1,0] = Pinned Follower (direct leader connection)")
    print(f"     [0,0,1] = Normal Follower")
    print(f"  🎲 Randomization Settings:")
    print(f"     Randomize Leader: {RANDOMIZE_LEADER}")
    if RANDOMIZE_LEADER:
        print(f"       Amplitude: {LEADER_AMPLITUDE_RANGE}")
        print(f"       Omega: {LEADER_OMEGA_RANGE}")
        print(f"       Phase: {LEADER_PHASE_RANGE}")
    print(f"     Randomize Follower: {RANDOMIZE_FOLLOWER}")
    if RANDOMIZE_FOLLOWER:
        print(f"       Pos Std: {FOLLOWER_INIT_POS_STD_RANGE}")
        print(f"       Vel Std: {FOLLOWER_INIT_VEL_STD_RANGE}")
    print(f"     Randomize Topology: {RANDOMIZE_TOPOLOGY}")
    if RANDOMIZE_TOPOLOGY:
        print(f"       Pinned Range: {NUM_PINNED_RANGE}")
        print(f"       Extra Edge Prob: {EXTRA_EDGE_PROB}")
    print(f"  🧠 Network Settings ({mode_str} - Scaled for {NUM_FOLLOWERS} followers):")
    print(f"     Hidden Dim: {HIDDEN_DIM}")
    print(f"     Attention Heads: {NUM_ATTENTION_HEADS}")
    print(f"     Transformer Layers: {NUM_TRANSFORMER_LAYERS}")
    print(f"     Dropout: {DROPOUT}")
    print(f"  ⚡ Training Settings:")
    print(f"     Batch Size: {BATCH_SIZE}")
    print(f"     Parallel Envs: {NUM_PARALLEL_ENVS}")
    print(f"     Update Frequency: {UPDATE_FREQUENCY}")
    print(f"     Gradient Steps: {GRADIENT_STEPS}")
    print(f"     Policy Delay: {POLICY_DELAY}")
    print(f"     Target Update Interval: {TARGET_UPDATE_INTERVAL}")
    print(f"     Buffer Size: {BUFFER_SIZE:,}")
    print(f"     Replay Buffer Device: {REPLAY_BUFFER_DEVICE}")
    print(f"     Replay Buffer DType: {REPLAY_BUFFER_DTYPE}")
    print(f"     Replay Buffer Pin Memory: {REPLAY_BUFFER_PIN_MEMORY}")
    print(f"  📡 Communication Settings:")
    print(f"     Base Comm Penalty: {COMM_PENALTY}")
    print(f"     Comm Weight Decay: {COMM_WEIGHT_DECAY}")
    print(f"     Threshold Range: [{THRESHOLD_MIN}, {THRESHOLD_MAX}]")
    print(f"  💾 Output Paths:")
    print(f"     RUN_DIR: {RUN_DIR}")
    print(f"     MODELS_DIR: {MODELS_DIR}")
    print(f"     FIGS_DIR: {FIGS_DIR}")
    print(f"     SAVE_MODEL_PATH: {SAVE_MODEL_PATH}")
    print(f"     EVAL_SAVE_PATH: {EVAL_SAVE_PATH}")
    print(f"  🎯 Reward Settings (Soft Comm Reduction):")
    print(f"     Tracking Penalty: -{TRACKING_PENALTY_MAX}*log1p(err_norm*{TRACKING_PENALTY_SCALE})")
    print(f"       where err_norm = mean(|pos_f-leader|)/{POS_LIMIT} + 0.01*mean(|vel_f-leader|)/{VEL_LIMIT}")
    print(f"     Comm Penalty: {COMM_PENALTY}")
    print(f"     Comm Weight Decay: {COMM_WEIGHT_DECAY}")
    print(f"     Improvement: scale={IMPROVEMENT_SCALE}, clip=±{IMPROVEMENT_CLIP}")
    print(f"  🔧 Action Scales: V={V_SCALE}, TH={TH_SCALE}")
    print(f"  🔥 Warmup Steps: {WARMUP_STEPS}")
    if RANDOMIZE_LEADER:
        print(f"  🎭 Leader Trajectory Types: {LEADER_TRAJECTORY_TYPES}")
    print("=" * 70)
