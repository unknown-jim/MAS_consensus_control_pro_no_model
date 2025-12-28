"""环境快速自检脚本。

用于在开始训练前做最小 sanity check：
- 零控制/固定阈值下是否稳定
- 粗略搜索一个可用的阈值范围
- 轨迹跟踪相关性（leader 与 follower 平均位置）

注：本脚本里的阈值使用“环境实际阈值”（与 `positions` 同量纲），会转换成策略动作的 raw 值。
"""
import torch

from mas_cc.config import NUM_FOLLOWERS, NUM_PINNED, MAX_STEPS, DEVICE, THRESHOLD_MIN, THRESHOLD_MAX, TH_SCALE
from mas_cc.environment import ModelFreeEnv
from mas_cc.topology import CommunicationTopology

def _raw_threshold_from_env_threshold(threshold_env: float) -> float:
    """把“环境实际阈值”反算为策略动作的 raw 阈值。

    环境内部的映射关系为：
        threshold_env = THRESHOLD_MIN + (THRESHOLD_MAX - THRESHOLD_MIN) * clamp(raw/TH_SCALE, 0, 1)

    Args:
        threshold_env: 环境实际阈值（位于 [THRESHOLD_MIN, THRESHOLD_MAX]）。

    Returns:
        raw 阈值（对应 actor 动作第 2 维的 raw 值）。
    """
    x = (threshold_env - THRESHOLD_MIN) / (THRESHOLD_MAX - THRESHOLD_MIN + 1e-12)
    x = float(min(1.0, max(0.0, x)))
    return x * TH_SCALE


def test_zero_control():
    """测试零调整控制（仅固定阈值）下系统是否稳定。

    Returns:
        若满足稳定判据返回 True，否则 False。
    """
    topology = CommunicationTopology(NUM_FOLLOWERS, num_pinned=NUM_PINNED)
    env = ModelFreeEnv(topology)
    
    state = env.reset()
    errors = []
    
    for step in range(MAX_STEPS):
        action = torch.zeros(env.num_followers, 2, device=DEVICE)
        # 这里的 0.15 指“环境实际阈值”（单位同 positions），需要反算成 raw action
        action[:, 1] = _raw_threshold_from_env_threshold(0.15)
        state, reward, done, info = env.step(action)
        errors.append(info['tracking_error'])
    
    init_err = errors[0]
    final_err = errors[-1]
    max_err = max(errors)
    avg_err = sum(errors) / len(errors)
    
    print(f"零调整控制:")
    print(f"  初始误差: {init_err:.4f}")
    print(f"  最终误差: {final_err:.4f}")
    print(f"  最大误差: {max_err:.4f}")
    print(f"  平均误差: {avg_err:.4f}")
    
    # 更新的判断标准：最终误差 < 1.0 且没有发散
    stable = final_err < 1.0 and max_err < 5.0
    print(f"系统是否稳定: {'是 ✅' if stable else '否 ⚠️'}")
    return stable

def test_optimal_threshold():
    """粗略搜索一个可用的环境阈值。

    Returns:
        最优阈值（环境实际阈值，非 raw action）。
    """
    topology = CommunicationTopology(NUM_FOLLOWERS, num_pinned=NUM_PINNED)
    
    print("\n寻找最优阈值:")
    print("-" * 60)
    
    best_threshold = None
    best_score = float('inf')
    
    # threshold_env：环境实际阈值（应位于 [THRESHOLD_MIN, THRESHOLD_MAX]）
    for threshold in [0.10, 0.12, 0.15, 0.18, 0.20, 0.25, 0.35, 0.50]:
        env = ModelFreeEnv(topology)
        state = env.reset()
        
        total_comm = 0
        errors = []
        
        for step in range(MAX_STEPS):
            action = torch.zeros(env.num_followers, 2, device=DEVICE)
            action[:, 1] = _raw_threshold_from_env_threshold(threshold)
            state, reward, done, info = env.step(action)
            total_comm += info['comm_rate']
            errors.append(info['tracking_error'])
        
        avg_comm = total_comm / MAX_STEPS
        avg_err = sum(errors) / len(errors)
        final_err = errors[-1]
        
        # 综合评分 = 误差 + 通信惩罚
        score = avg_err + avg_comm * 0.5
        
        marker = ""
        if score < best_score:
            best_score = score
            best_threshold = threshold
            marker = " ← Best"
        
        print(f"阈值(env)={threshold:.2f} | 平均误差={avg_err:.4f} | 最终误差={final_err:.4f} | "
              f"通信率={avg_comm*100:.1f}% | Score={score:.4f}{marker}")
    
    print(f"\n最优阈值: {best_threshold:.2f}")
    return best_threshold

def test_trajectory_tracking():
    """测试轨迹跟踪效果。

    采用 leader 位置与 follower 平均位置的相关系数作为粗指标。
    """
    topology = CommunicationTopology(NUM_FOLLOWERS, num_pinned=NUM_PINNED)
    env = ModelFreeEnv(topology)
    
    state = env.reset()
    
    leader_pos = []
    avg_follower_pos = []
    
    for step in range(MAX_STEPS):
        action = torch.zeros(env.num_followers, 2, device=DEVICE)
        action[:, 1] = _raw_threshold_from_env_threshold(0.15)
        state, reward, done, info = env.step(action)
        
        leader_pos.append(env.positions[0].item())
        avg_follower_pos.append(env.positions[1:].mean().item())
    
    # 计算相关系数
    import numpy as np
    correlation = np.corrcoef(leader_pos, avg_follower_pos)[0, 1]
    
    print(f"\n轨迹跟踪效果:")
    print(f"  领导者-跟随者相关系数: {correlation:.4f}")
    print(f"  跟踪质量: {'优秀 ✅' if correlation > 0.95 else '良好 ✓' if correlation > 0.8 else '需改进 ⚠️'}")

if __name__ == '__main__':
    print("=" * 60)
    print("环境稳定性测试 (更新版)")
    print("=" * 60)
    
    stable = test_zero_control()
    best_th = test_optimal_threshold()
    test_trajectory_tracking()
    
    print("\n" + "=" * 60)
    if stable:
        print("✅ 环境稳定，可以开始训练！")
        print(f"   建议初始阈值范围: [{best_th-0.05:.2f}, {best_th+0.05:.2f}]")
    else:
        print("⚠️ 需要进一步调整增益参数")
    print("=" * 60)