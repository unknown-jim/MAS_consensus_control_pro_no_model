"""
工具函数 - CTDE 版本
"""
import torch
import numpy as np

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

from config import DEVICE, MAX_STEPS


@torch.no_grad()
def collect_trajectory(agent, env, max_steps=MAX_STEPS):
    """收集轨迹用于可视化（CTDE 版本，含通信数据）"""
    state = env.reset()
    
    times = [0]
    leader_pos = [env.positions[0].item()]
    leader_vel = [env.velocities[0].item()]
    follower_pos = [env.positions[1:].cpu().numpy()]
    follower_vel = [env.velocities[1:].cpu().numpy()]
    
    # 🔧 通信数据
    comm_rates = []  # 每步的通信率
    thresholds = []  # 每步的阈值
    triggered = []   # 每步每个智能体是否触发通信
    
    for step in range(max_steps):
        # 🔧 state 需要是 (num_agents, state_dim) 格式
        if state.dim() == 1:
            state = state.unsqueeze(0)
        
        action = agent.select_action(state, deterministic=True)
        state, _, _, info = env.step(action)
        
        times.append(env.t)
        leader_pos.append(env.positions[0].item())
        leader_vel.append(env.velocities[0].item())
        follower_pos.append(env.positions[1:].cpu().numpy())
        follower_vel.append(env.velocities[1:].cpu().numpy())
        
        # 🔧 记录通信数据
        comm_rates.append(info['comm_rate'])
        # 阈值从 action 中提取 (action shape: num_followers, 2)
        thresholds.append(action[:, 1].cpu().numpy())
    
    return {
        'times': np.array(times),
        'leader_pos': np.array(leader_pos),
        'leader_vel': np.array(leader_vel),
        'follower_pos': np.array(follower_pos),
        'follower_vel': np.array(follower_vel),
        # 🔧 通信数据
        'comm_rates': np.array(comm_rates),
        'thresholds': np.array(thresholds),  # (steps, num_followers)
    }


@torch.no_grad()
def evaluate_agent(agent, env, num_episodes=5):
    """评估智能体性能（CTDE 版本）"""
    results = {
        'rewards': [],
        'tracking_errors': [],
        'comm_rates': []
    }
    
    for _ in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        episode_tracking_err = 0
        episode_comm = 0
        
        for step in range(MAX_STEPS):
            action = agent.select_action(state, deterministic=True)
            state, reward, _, info = env.step(action)
            
            episode_reward += reward
            episode_tracking_err += info['tracking_error']
            episode_comm += info['comm_rate']
        
        results['rewards'].append(episode_reward)
        results['tracking_errors'].append(episode_tracking_err / MAX_STEPS)
        results['comm_rates'].append(episode_comm / MAX_STEPS)
    
    return {
        'mean_reward': np.mean(results['rewards']),
        'std_reward': np.std(results['rewards']),
        'mean_tracking_error': np.mean(results['tracking_errors']),
        'mean_comm_rate': np.mean(results['comm_rates'])
    }


def plot_evaluation(agent, topology, num_tests=3, save_path=None):
    """绘制评估结果"""
    if not HAS_MATPLOTLIB:
        print("matplotlib not available")
        return
    
    from environment import ModelFreeEnv
    
    env = ModelFreeEnv(topology)
    
    fig, axes = plt.subplots(num_tests, 2, figsize=(14, 4 * num_tests))
    if num_tests == 1:
        axes = axes.reshape(1, -1)
    
    results = []
    
    for test_idx in range(num_tests):
        traj = collect_trajectory(agent, env, MAX_STEPS)
        
        pos_errors = (traj['follower_pos'] - traj['leader_pos'][:, np.newaxis])**2
        final_error = np.mean(pos_errors[-1])
        avg_error = np.mean(pos_errors)
        
        results.append({'final_error': final_error, 'avg_error': avg_error})
        
        ax1 = axes[test_idx, 0]
        ax1.plot(traj['times'], traj['leader_pos'], 'r-', linewidth=2.5, label='Leader')
        colors = plt.cm.Blues(np.linspace(0.3, 0.9, traj['follower_pos'].shape[1]))
        for i in range(min(5, traj['follower_pos'].shape[1])):
            ax1.plot(traj['times'], traj['follower_pos'][:, i], color=colors[i], 
                    alpha=0.8, linewidth=1.2, label=f'F{i+1}')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Position')
        ax1.set_title(f'Test {test_idx+1}: Position (Final Err: {final_error:.4f})')
        ax1.legend(loc='upper right', fontsize=8)
        ax1.grid(True, alpha=0.3)
        
        ax2 = axes[test_idx, 1]
        ax2.plot(traj['times'], traj['leader_vel'], 'r-', linewidth=2.5, label='Leader')
        for i in range(min(5, traj['follower_vel'].shape[1])):
            ax2.plot(traj['times'], traj['follower_vel'][:, i], color=colors[i], 
                    alpha=0.8, linewidth=1.2)
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Velocity')
        ax2.set_title(f'Test {test_idx+1}: Velocity')
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"📁 Figure saved to {save_path}")
    
    plt.show()
    
    print("\n📊 CTDE Evaluation Results:")
    print("-" * 40)
    for i, r in enumerate(results):
        print(f"Test {i+1}: Final Err = {r['final_error']:.4f}, Avg Err = {r['avg_error']:.4f}")
    
    return results