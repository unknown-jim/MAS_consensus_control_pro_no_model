"""
配置文件 - CTDE 架构版本（随机初始化）
"""
import torch
import random
import numpy as np

# ==================== 设备配置 ====================
# 注意：部分新 GPU（如 RTX 5090 / sm_120）需要更高版本的 PyTorch + CUDA（例如 cu128+）
# 若检测到当前 PyTorch 不支持本机 GPU 架构，则自动回退到 CPU，避免运行时报
# "CUDA error: no kernel image is available for execution on the device"。

def _select_device():
    if not torch.cuda.is_available():
        return torch.device('cpu')

    # 通过 PyTorch 编译时支持的 arch 列表判断兼容性
    try:
        cap = torch.cuda.get_device_capability(0)  # e.g. (12, 0)
        arch_list = torch.cuda.get_arch_list() if hasattr(torch.cuda, 'get_arch_list') else []

        supported_caps = set()
        for a in arch_list:
            if isinstance(a, str) and a.startswith('sm_'):
                n = int(a[3:])  # '90' -> 90
                supported_caps.add((n // 10, n % 10))

        # 如果能拿到 supported_caps 且不包含当前 cap，则认为不兼容
        if supported_caps and cap not in supported_caps:
            gpu_name = torch.cuda.get_device_name(0)
            print(
                f"⚠️ Detected GPU {gpu_name} with compute capability sm_{cap[0]}{cap[1]} is not compatible with "
                f"current PyTorch build (supports: {sorted(supported_caps)}). Falling back to CPU.\n"
                f"   Fix: install a PyTorch build that includes sm_{cap[0]}{cap[1]} (often requires CUDA 12.8+ / cu128+)."
            )
            return torch.device('cpu')

    except Exception:
        # 若探测失败，保守起见仍尝试用 CUDA（多数情况下可用）
        return torch.device('cuda')

    return torch.device('cuda')


DEVICE = _select_device()
SEED = 42
TOPOLOGY_SEED = 42

# ==================== 模式选择 ====================
LIGHTWEIGHT_MODE = True

# ==================== 算法选择 ====================
# 可选："MASAC"（当前 CTDE-SAC 实现） / "MAPPO"（新增 CTDE-MAPPO 实现）
ALGO = "MASAC"

# ==================== 网络拓扑 ====================
# 总智能体数 = 1 Leader + NUM_FOLLOWERS Followers
# 你要求改为 15 个智能体 => NUM_FOLLOWERS = 14
NUM_FOLLOWERS = 14              # 14 个跟随者
NUM_PINNED = 3                  # pinned followers 数量（会被 NUM_PINNED_RANGE 约束）
NUM_AGENTS = NUM_FOLLOWERS + 1  # 15 个智能体

# ==================== 无模型状态空间 ====================
# 观测设计（Decentralized Actor 输入）：
# - 自身：位置/速度 + 自己当前掌握的 leader 估计（用于无记忆策略）
# - 自身角色：one-hot [leader, pinned, normal]
# - 邻居：邻居的广播状态(位置/速度) + 邻居随包携带的 leader 估计（含 seq/age）
#
# leader 估计字段（同时出现在自身与邻居广播包里）：
# - leader_pos_est, leader_vel_est
# - leader_seq_norm, leader_age_norm
SELF_STATE_DIM = 2      # 自身位置、速度
SELF_LEADER_DIM = 4     # leader 估计：pos/vel + seq_norm/age_norm
LOCAL_OBS_DIM = SELF_STATE_DIM + SELF_LEADER_DIM  # 6

SELF_ROLE_DIM = 3       # 自身角色 one-hot: [leader, pinned, normal]

# 邻居特征（广播包可见）
NEIGHBOR_STATE_DIM = 2  # 邻居广播的自身位置、速度
NEIGHBOR_LEADER_DIM = 4 # 邻居携带的 leader 估计：pos/vel + seq_norm/age_norm

# 注意：为了兼容现有网络实现，这里的 NEIGHBOR_OBS_DIM 直接表示“邻居广播包里的完整可见特征”（不含 role）。
NEIGHBOR_OBS_DIM = NEIGHBOR_STATE_DIM + NEIGHBOR_LEADER_DIM  # 6
NEIGHBOR_ROLE_DIM = 0   # ✅ 移除：邻居角色不再作为输入（保留 self_role 即可）

# 15 智能体规模下，拓扑中实际最大入邻居数通常远小于 10；
# 适当降低 MAX_NEIGHBORS 可以显著减小 STATE_DIM，提升样本效率与训练速度。
MAX_NEIGHBORS = 6       # 最大邻居数（覆盖当前拓扑统计的上界即可）

# 每个邻居的完整特征维度
NEIGHBOR_FEAT_DIM = NEIGHBOR_OBS_DIM + NEIGHBOR_ROLE_DIM  # 6

# 总状态维度 = 本地观测 + 自身角色 + 邻居特征
STATE_DIM = LOCAL_OBS_DIM + SELF_ROLE_DIM + MAX_NEIGHBORS * NEIGHBOR_FEAT_DIM

# ==================== CTDE 全局状态维度 ====================
# 说明：为了让 centralized critic 的输入更接近 Markov 状态，我们允许把“通信广播记忆/领导者参数/轨迹类型/时间”
# 也并入 global state。否则在 RANDOMIZE_* 打开时，critic 只能学到“混合任务的平均值”，会显著增大方差。
GLOBAL_STATE_INCLUDE_BROADCAST = True        # 加入 last_broadcast_pos/vel
GLOBAL_STATE_INCLUDE_LEADER_PARAMS = True    # 加入 leader 的 amplitude/omega/phase
GLOBAL_STATE_INCLUDE_TRAJ_TYPE = True        # 加入 trajectory type one-hot
GLOBAL_STATE_INCLUDE_TIME = True             # 加入 t 的归一化值（0~1）

GLOBAL_STATE_DIM_BASE = NUM_AGENTS * 2       # positions/velocities
GLOBAL_STATE_DIM = GLOBAL_STATE_DIM_BASE     # 会在 LEADER_TRAJECTORY_TYPES 定义后“最终确定”

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
# 对于 14 个跟随者，pinned 范围适当收敛，避免拓扑变化过大导致训练高方差
NUM_PINNED_RANGE = (2, 5)        # Pinned followers 数量范围
EXTRA_EDGE_PROB = 0.15           # 额外边概率（避免过度连接）

# 领导者动力学随机化范围
LEADER_AMPLITUDE_RANGE = (1.0, 3.0)    # 振幅范围
LEADER_OMEGA_RANGE = (0.3, 0.8)        # 角频率范围
LEADER_PHASE_RANGE = (0.0, 2 * 3.14159) # 相位范围 [0, 2π]

# 领导者轨迹类型（提升泛化能力）
# 轨迹类型越多/越非平稳（如 chirp），越容易导致 early-stage 学习变慢。
# 先用更简单的分布快速学到“跟随”，再按需扩展类型。
LEADER_TRAJECTORY_TYPES = ['sine', 'cosine']

# ===== finalize CTDE global state dim (needs LEADER_TRAJECTORY_TYPES) =====
GLOBAL_STATE_DIM = GLOBAL_STATE_DIM_BASE
if GLOBAL_STATE_INCLUDE_BROADCAST:
    GLOBAL_STATE_DIM += NUM_AGENTS * 2
if GLOBAL_STATE_INCLUDE_LEADER_PARAMS:
    GLOBAL_STATE_DIM += 3
if GLOBAL_STATE_INCLUDE_TRAJ_TYPE:
    GLOBAL_STATE_DIM += len(LEADER_TRAJECTORY_TYPES)
if GLOBAL_STATE_INCLUDE_TIME:
    GLOBAL_STATE_DIM += 1

# 跟随者初始状态随机化范围
# 初始随机化过强会让前期大量 episode 都处在“很难恢复”的区域，学习信号被噪声淹没。
FOLLOWER_INIT_POS_STD_RANGE = (0.4, 1.2)  # 位置标准差范围
FOLLOWER_INIT_VEL_STD_RANGE = (0.15, 0.5)  # 速度标准差范围

# 当 RANDOMIZE_FOLLOWER=False 时使用的固定标准差（与环境实现保持一致）
FOLLOWER_INIT_POS_STD = 0.5
FOLLOWER_INIT_VEL_STD = 0.2

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

# ==================== Dashboard 显示阈值（颜色/参考线，建议跟随 config）====================
# 误差阈值使用“相对上限的比例”来定义：
# - good: 位置/速度平均误差约为上限的 5%
# - poor: 位置/速度平均误差约为上限的 20%
DASH_ERROR_GOOD_FRAC = 0.05
DASH_ERROR_POOR_FRAC = 0.20

# 通信率阈值（比例值）
DASH_COMM_GOOD_THRESHOLD = 0.30
DASH_COMM_POOR_THRESHOLD = 0.70

# ==================== 网络参数（根据模式调整）====================
# 大规模系统（30 followers）需要更大的网络容量
if LIGHTWEIGHT_MODE:
    # 14 followers 规模下可以适当减小网络容量，提高更新密度来换取更快的 early-stage 提升
    HIDDEN_DIM = 192
    NUM_ATTENTION_HEADS = 4
    NUM_TRANSFORMER_LAYERS = 2
    DROPOUT = 0.05

    # 训练参数：提高单位环境步的优化次数（当前任务更需要“前期快”）
    BATCH_SIZE = 256
    NUM_EPISODES = 1200
    NUM_PARALLEL_ENVS = 48

    # 相比之前的 UPDATE_FREQUENCY=8，这里提高更新频率，让曲线更快抬头
    UPDATE_FREQUENCY = 2
    GRADIENT_STEPS = 1

    print("🚀 Lightweight Mode Enabled (CTDE) - 14 Followers")
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
    print("🔬 Full Mode Enabled (CTDE) - 14 Followers")

# ==================== SAC 参数 ====================
LEARNING_RATE = 3e-4
ACTOR_LR = 3e-4
CRITIC_LR = 3e-4
ALPHA_LR = 3e-4

# ==================== PPO/MAPPO 参数 ====================
# 说明：本实现为 CTDE-MAPPO（centralized value + decentralized shared policy）。
PPO_LR = 3e-4
PPO_CLIP_EPS = 0.2
PPO_EPOCHS = 4
PPO_ROLLOUT_STEPS = 128          # 每次收集多少步 on-policy 数据再更新
PPO_MINIBATCH_SIZE = 1024        # 以 (rollout_steps * num_envs) 为基准切分
PPO_GAE_LAMBDA = 0.95
PPO_VALUE_COEF = 0.5
PPO_ENTROPY_COEF = 0.01
PPO_MAX_GRAD_NORM = 1.0
PPO_TARGET_KL = 0.02             # 可用于早停（0 表示不启用）

GAMMA = 0.99
TAU = 0.005

INIT_ALPHA = 0.2
AUTO_ALPHA = True
TARGET_ENTROPY_RATIO = 0.4

BUFFER_SIZE = 1000000            # 增大缓冲区（30 智能体产生更多经验）

# ==================== Replay Buffer 存储设置 ====================
# 说明：local state 维度很大（num_agents×STATE_DIM），若把整块 buffer 放 GPU 会非常占显存。
# 默认策略：若训练在 GPU 上，则把 replay buffer 存在 CPU（可选 pinned memory），采样时再搬到 GPU。
REPLAY_BUFFER_DEVICE = torch.device('cpu') if DEVICE.type == 'cuda' else DEVICE
REPLAY_BUFFER_DTYPE = torch.float16 if DEVICE.type == 'cuda' else torch.float32
REPLAY_BUFFER_PIN_MEMORY = (DEVICE.type == 'cuda')

# ==================== 网络参数 ====================
LOG_STD_MIN = -20
LOG_STD_MAX = 2
V_SCALE = 1.0   # 平衡：隐含加速度 20 m/s²，既能跟上又不会瞬跳
TH_SCALE = 1.0

# ==================== 训练参数 ====================
VIS_INTERVAL = 5                # 增大可视化间隔（训练轮数更多）
USE_AMP = True
WARMUP_STEPS = 3000              # 15 智能体规模下适当降低预热，加快进入学习阶段

# ==================== 更新策略加速开关 ====================
# Policy Delay: Actor/α 不必每个 critic step 都更新（可显著减少反传开销）
POLICY_DELAY = 2                 # 每 2 个 critic 更新，做 1 次 actor+alpha 更新
# Target 更新间隔：减少 target 网络软更新的频率（用等效 tau 保持效果接近）
TARGET_UPDATE_INTERVAL = 2       # 每 2 个 critic 更新做一次 target soft-update

_SAVE_TAG = 'mappo' if str(ALGO).upper().strip() == 'MAPPO' else 'masac'
SAVE_MODEL_PATH = (
    f'best_model_ctde_14f_{_SAVE_TAG}_light.pt'
    if LIGHTWEIGHT_MODE
    else f'best_model_ctde_14f_{_SAVE_TAG}.pt'
)

# ==================== Notebook / Evaluation（建议统一从 config 读取，避免 hardcode）====================
EVAL_NUM_TESTS = 3
EVAL_SAVE_PATH = f'final_evaluation_ctde_{_SAVE_TAG}.png'

# 泛化测试：单环境不使用 done（环境内部 dones 恒为 False），这里用步数控制时长
GENERALIZATION_TEST_STEPS = MAX_STEPS * 2
GENERALIZATION_SAVE_PATH = f'generalization_test_ctde_{_SAVE_TAG}.png'

# 泛化 OOD（Out-of-distribution）测试：用于查看是否能外推到训练范围外
GENERALIZATION_INCLUDE_OOD = True
GENERALIZATION_OOD_AMPLITUDE = LEADER_AMPLITUDE
GENERALIZATION_OOD_OMEGA = LEADER_OMEGA_RANGE[1] * 1.25


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
    print(f"  🎯 Reward Settings (Soft Comm Reduction):")
    print(f"     Tracking Penalty: -{TRACKING_PENALTY_MAX}*log1p(err_norm*{TRACKING_PENALTY_SCALE})")
    print(f"       where err_norm = mean(|pos_f-leader|)/{POS_LIMIT} + 0.5*mean(|vel_f-leader|)/{VEL_LIMIT}")
    print(f"     Comm Penalty: {COMM_PENALTY}")
    print(f"     Comm Weight Decay: {COMM_WEIGHT_DECAY}")
    print(f"     Improvement: scale={IMPROVEMENT_SCALE}, clip=±{IMPROVEMENT_CLIP}")
    print(f"  🔧 Action Scales: V={V_SCALE}, TH={TH_SCALE}")
    print(f"  🔥 Warmup Steps: {WARMUP_STEPS}")
    if RANDOMIZE_LEADER:
        print(f"  🎭 Leader Trajectory Types: {LEADER_TRAJECTORY_TYPES}")
    print("=" * 70)