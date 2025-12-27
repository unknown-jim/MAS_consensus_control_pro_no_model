"""
配置文件 - CTDE 架构版本（随机初始化）
"""
import torch
import random
import numpy as np

# ==================== 设备配置 ====================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED = 42
TOPOLOGY_SEED = 42

# ==================== 模式选择 ====================
LIGHTWEIGHT_MODE = True

# ==================== 网络拓扑 ====================
NUM_FOLLOWERS = 30              # 增加到 30 个跟随者
NUM_PINNED = 5                  # 增加 pinned followers 数量
NUM_AGENTS = NUM_FOLLOWERS + 1  # 31 个智能体

# ==================== 无模型状态空间 ====================
LOCAL_OBS_DIM = 2       # 自身位置、速度
SELF_ROLE_DIM = 3       # 自身角色 one-hot: [leader, pinned, normal]
NEIGHBOR_OBS_DIM = 2    # 邻居位置、速度
NEIGHBOR_ROLE_DIM = 3   # 邻居角色 one-hot: [leader, pinned, normal]
MAX_NEIGHBORS = 10      # 增加最大邻居数（大规模系统需要更多连接）

# 每个邻居的完整特征维度
NEIGHBOR_FEAT_DIM = NEIGHBOR_OBS_DIM + NEIGHBOR_ROLE_DIM  # 5

# 总状态维度 = 自身状态 + 自身角色 + 邻居(状态+角色)
STATE_DIM = LOCAL_OBS_DIM + SELF_ROLE_DIM + MAX_NEIGHBORS * NEIGHBOR_FEAT_DIM  # 2 + 3 + 10*5 = 55

# ==================== CTDE 全局状态维度 ====================
GLOBAL_STATE_DIM = NUM_AGENTS * 2

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
RANDOMIZE_LEADER = True          # 是否随机化领导者动力学
RANDOMIZE_FOLLOWER = True        # 是否随机化跟随者初始状态
RANDOMIZE_TOPOLOGY = True        # 是否每 episode 随机化拓扑

# 拓扑随机化参数
NUM_PINNED_RANGE = (3, 8)        # Pinned followers 数量范围（大规模系统需要更多）
EXTRA_EDGE_PROB = 0.15           # 降低额外边概率（避免过度连接）

# 领导者动力学随机化范围
LEADER_AMPLITUDE_RANGE = (1.0, 3.0)    # 振幅范围
LEADER_OMEGA_RANGE = (0.3, 0.8)        # 角频率范围
LEADER_PHASE_RANGE = (0.0, 2 * 3.14159) # 相位范围 [0, 2π]

# 领导者轨迹类型（提升泛化能力）
LEADER_TRAJECTORY_TYPES = ['sine', 'cosine', 'mixed', 'chirp']

# 跟随者初始状态随机化范围
FOLLOWER_INIT_POS_STD_RANGE = (1.0, 5.0)  # 位置标准差范围
FOLLOWER_INIT_VEL_STD_RANGE = (0.3, 1.5)  # 速度标准差范围

POS_LIMIT = 10.0
VEL_LIMIT = 10.0

COMM_RANGE = 5.0

# ==================== 通信参数（固定值）====================
COMM_PENALTY = 0.15  # 增大通信惩罚，强化减少通信的激励
THRESHOLD_MIN = 0.1  # 提高下限：阻止"总是通信"策略
THRESHOLD_MAX = 1.0

# ==================== 奖励参数（自适应权重）====================
TRACKING_PENALTY_SCALE = 2.0
TRACKING_PENALTY_MAX = 1.0
COMM_WEIGHT_DECAY = 0.8           # 降低：让通信惩罚在更大误差范围内生效
IMPROVEMENT_SCALE = 1.5           # 降低：减少对快速改进的过度激励
IMPROVEMENT_CLIP = 0.3            # 降低：限制改进奖励幅度

# 奖励缩放
REWARD_MIN = -2.0
REWARD_MAX = 2.0
USE_SOFT_REWARD_SCALING = True

# ==================== 网络参数（根据模式调整）====================
# 大规模系统（30 followers）需要更大的网络容量
if LIGHTWEIGHT_MODE:
    HIDDEN_DIM = 256              # 增大隐藏层（30 智能体需要更多容量）
    NUM_ATTENTION_HEADS = 4       # 增加注意力头
    NUM_TRANSFORMER_LAYERS = 2    # 增加层数
    DROPOUT = 0.1
    BATCH_SIZE = 512              # 增大批量（更多智能体产生更多数据）
    NUM_EPISODES = 1500           # 增加训练轮数
    NUM_PARALLEL_ENVS = 32        # 减少并行环境（显存考虑）
    UPDATE_FREQUENCY = 4          # 更频繁更新
    GRADIENT_STEPS = 2            # 增加梯度步数
    print("🚀 Lightweight Mode Enabled (CTDE) - 30 Followers")
else:
    HIDDEN_DIM = 512              # 大规模系统使用更大网络
    NUM_ATTENTION_HEADS = 8
    NUM_TRANSFORMER_LAYERS = 3
    DROPOUT = 0.1
    BATCH_SIZE = 1024
    NUM_EPISODES = 2000
    NUM_PARALLEL_ENVS = 16
    UPDATE_FREQUENCY = 4
    GRADIENT_STEPS = 4
    print("🔬 Full Mode Enabled (CTDE) - 30 Followers")

# ==================== SAC 参数 ====================
LEARNING_RATE = 3e-4
ACTOR_LR = 3e-4
CRITIC_LR = 3e-4
ALPHA_LR = 3e-4

GAMMA = 0.99
TAU = 0.005

INIT_ALPHA = 0.2
AUTO_ALPHA = True
TARGET_ENTROPY_RATIO = 0.5

BUFFER_SIZE = 1000000            # 增大缓冲区（30 智能体产生更多经验）

# ==================== 网络参数 ====================
LOG_STD_MIN = -20
LOG_STD_MAX = 2
V_SCALE = 1.0   # 平衡：隐含加速度 20 m/s²，既能跟上又不会瞬跳
TH_SCALE = 1.0

# ==================== 训练参数 ====================
VIS_INTERVAL = 20                # 增大可视化间隔（训练轮数更多）
USE_AMP = True
WARMUP_STEPS = 10000             # 增加预热步数（大规模系统需要更多预热）

SAVE_MODEL_PATH = 'best_model_ctde_30f_light.pt' if LIGHTWEIGHT_MODE else 'best_model_ctde_30f.pt'


def set_seed(seed=SEED):
    """设置全局随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def print_config():
    """打印配置信息"""
    mode_str = "Lightweight" if LIGHTWEIGHT_MODE else "Full"
    print("=" * 70)
    print(f"🔧 Configuration - CTDE Architecture ({mode_str} Mode) - Large Scale")
    print("=" * 70)
    print(f"  Device: {DEVICE}")
    print(f"  Seed: {SEED}")
    print(f"  🌐 Large-Scale MAS: {NUM_AGENTS} Agents (1 Leader + {NUM_FOLLOWERS} Followers)")
    print(f"  Episodes: {NUM_EPISODES}, Max Steps: {MAX_STEPS}")
    print(f"  📡 CTDE Settings:")
    print(f"     Local Obs Dim: {LOCAL_OBS_DIM} (pos, vel)")
    print(f"     Self Role Dim: {SELF_ROLE_DIM} (one-hot: leader/pinned/normal)")
    print(f"     Neighbor Feat Dim: {NEIGHBOR_FEAT_DIM} (obs + role)")
    print(f"     Global State Dim: {GLOBAL_STATE_DIM}")
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
    print(f"     Buffer Size: {BUFFER_SIZE:,}")
    print(f"  📡 Communication Settings:")
    print(f"     Base Comm Penalty: {COMM_PENALTY}")
    print(f"     Comm Weight Decay: {COMM_WEIGHT_DECAY}")
    print(f"     Threshold Range: [{THRESHOLD_MIN}, {THRESHOLD_MAX}]")
    print(f"  🎯 Reward Settings (Soft Comm Reduction):")
    print(f"     Tracking Penalty: tanh(err*{TRACKING_PENALTY_SCALE})*{TRACKING_PENALTY_MAX}")
    print(f"     Comm Penalty: {COMM_PENALTY}")
    print(f"     Comm Weight Decay: {COMM_WEIGHT_DECAY}")
    print(f"     Improvement: scale={IMPROVEMENT_SCALE}, clip=±{IMPROVEMENT_CLIP}")
    print(f"  🔧 Action Scales: V={V_SCALE}, TH={TH_SCALE}")
    print(f"  🔥 Warmup Steps: {WARMUP_STEPS}")
    if RANDOMIZE_LEADER:
        print(f"  🎭 Leader Trajectory Types: {LEADER_TRAJECTORY_TYPES}")
    print("=" * 70)