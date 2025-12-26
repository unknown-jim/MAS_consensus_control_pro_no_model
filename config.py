"""
配置文件 - 速度优化版
"""
import torch
import random
import numpy as np

# ==========================================
# 设备配置
# ==========================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 随机种子
# ==========================================
SEED = 42

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# ==========================================
# 系统配置
# ==========================================
NUM_FOLLOWERS = 9
NUM_AGENTS = NUM_FOLLOWERS + 1
LEADER_ID = 0
STATE_DIM = 4
HIDDEN_DIM = 128
ACTION_DIM = 2

# ==========================================
# 环境配置
# ==========================================
DT = 0.05
MAX_STEPS = 300
COMM_PENALTY = 0.03

LEADER_AMPLITUDE = 2.0
LEADER_OMEGA = 0.5
LEADER_PHASE = 0.0

POS_LIMIT = 10.0
VEL_LIMIT = 5.0

REWARD_MIN = -20.0
REWARD_MAX = 5.0
USE_SOFT_REWARD_SCALING = True

# ==========================================
# SAC 超参数
# ==========================================
BUFFER_SIZE = 500000
BATCH_SIZE = 2048           # 增大批量以提高GPU利用率
GAMMA = 0.99
TAU = 0.005
LEARNING_RATE = 3e-4
ALPHA_LR = 3e-4
LOG_STD_MIN = -20
LOG_STD_MAX = 2
INIT_ALPHA = 0.2

# ==========================================
# 训练配置 (速度优化) ⬇️ 关键修改
# ==========================================
NUM_EPISODES = 400
VIS_INTERVAL = 20           # 减少可视化频率
SAVE_MODEL_PATH = 'best_leader_follower_model.pth'

NUM_PARALLEL_ENVS = 64      # 32 -> 64 ⬆️
UPDATE_FREQUENCY = 32       # 8 -> 32  ⬆️ (关键！减少更新次数)
GRADIENT_STEPS = 1          # 4 -> 1   ⬇️ (关键！每次只更新1步)

# 混合精度
USE_AMP = True

# ==========================================
# 拓扑配置
# ==========================================
NUM_PINNED = 3
TOPOLOGY_SEED = 42


def print_config():
    # 计算每 episode 的更新次数
    updates_per_ep = MAX_STEPS // UPDATE_FREQUENCY
    total_gradient_steps = updates_per_ep * GRADIENT_STEPS
    
    print("=" * 60)
    print("🔧 Configuration (Speed Optimized)")
    print("=" * 60)
    print(f"  Device: {DEVICE}")
    print(f"  Parallel Envs: {NUM_PARALLEL_ENVS}")
    print(f"  Batch Size: {BATCH_SIZE}")
    print(f"  Update Frequency: every {UPDATE_FREQUENCY} steps")
    print(f"  Gradient Steps: {GRADIENT_STEPS}")
    print(f"  Updates per Episode: {updates_per_ep}")
    print(f"  Total Gradient Steps per Episode: {total_gradient_steps}")
    print(f"  AMP Training: {USE_AMP}")
    print("=" * 60)