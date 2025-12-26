"""
配置文件 - 所有超参数和全局配置
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
    """设置全局随机种子"""
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

# 领导者轨迹参数
LEADER_AMPLITUDE = 2.0
LEADER_OMEGA = 0.5
LEADER_PHASE = 0.0

# 状态边界
POS_LIMIT = 10.0
VEL_LIMIT = 5.0

# 奖励配置
REWARD_MIN = -20.0
REWARD_MAX = 5.0
USE_SOFT_REWARD_SCALING = True  # 使用软缩放而非硬截断

# ==========================================
# SAC 超参数
# ==========================================
BUFFER_SIZE = 200000
BATCH_SIZE = 256
GAMMA = 0.99
TAU = 0.005
LEARNING_RATE = 3e-4
ALPHA_LR = 3e-4
LOG_STD_MIN = -20
LOG_STD_MAX = 2
INIT_ALPHA = 0.2

# ==========================================
# 训练配置
# ==========================================
NUM_EPISODES = 400
VIS_INTERVAL = 5
SAVE_MODEL_PATH = 'best_leader_follower_model.pth'

# ==========================================
# 拓扑配置
# ==========================================
NUM_PINNED = 3
TOPOLOGY_SEED = 42


def print_config():
    """打印配置信息"""
    print("=" * 60)
    print("🔧 Configuration")
    print("=" * 60)
    print(f"  Device: {DEVICE}")
    print(f"  Random Seed: {SEED}")
    print(f"  Followers: {NUM_FOLLOWERS}, Pinned: {NUM_PINNED}")
    print(f"  State Dim: {STATE_DIM}, Hidden Dim: {HIDDEN_DIM}")
    print(f"  Max Steps: {MAX_STEPS}, Episodes: {NUM_EPISODES}")
    print(f"  Batch Size: {BATCH_SIZE}, Buffer Size: {BUFFER_SIZE}")
    print(f"  Position Limit: ±{POS_LIMIT}, Velocity Limit: ±{VEL_LIMIT}")
    print("=" * 60)