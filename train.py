"""
训练脚本
"""
from config import (
    NUM_FOLLOWERS, NUM_PINNED, MAX_STEPS, BATCH_SIZE,
    NUM_EPISODES, VIS_INTERVAL, SAVE_MODEL_PATH, 
    print_config, set_seed, SEED
)
from topology import DirectedSpanningTreeTopology
from environment import LeaderFollowerMASEnv
from agent import SACAgent
from dashboard import TrainingDashboard
from utils import collect_trajectory, plot_evaluation


def train(num_episodes=NUM_EPISODES, vis_interval=VIS_INTERVAL, 
          show_dashboard=True, seed=SEED):
    """训练主函数
    
    Args:
        num_episodes: 训练回合数
        vis_interval: 可视化间隔
        show_dashboard: 是否显示仪表盘
        seed: 随机种子
        
    Returns:
        agent: 训练好的智能体
        topology: 拓扑结构
        dashboard: 仪表盘对象
    """
    # 设置随机种子
    set_seed(seed)
    
    # 打印配置
    print_config()
    
    # 初始化
    topology = DirectedSpanningTreeTopology(NUM_FOLLOWERS, num_pinned=NUM_PINNED)
    env = LeaderFollowerMASEnv(topology)
    agent = SACAgent(topology)
    
    # 仪表盘
    dashboard = None
    if show_dashboard:
        dashboard = TrainingDashboard(num_episodes, vis_interval)
        dashboard.display()
    
    best_reward = -float('inf')
    
    # 训练循环
    for episode in range(1, num_episodes + 1):
        state = env.reset()
        episode_reward = 0
        episode_tracking_err = 0
        episode_comm = 0
        
        for step in range(MAX_STEPS):
            if dashboard:
                dashboard.update_step(step + 1, MAX_STEPS)
            
            # 选择动作 (训练时使用随机策略)
            action = agent.select_action(state, deterministic=False)
            
            # 环境交互
            next_state, reward, done, info = env.step(action)
            
            # 存储经验
            agent.store_transition(state, action, reward, next_state, done)
            
            # 更新网络
            agent.update(BATCH_SIZE)
            
            # 统计
            episode_reward += reward
            episode_tracking_err += info['tracking_error']
            episode_comm += info['comm_rate']
            state = next_state
            
            if done:
                break
        
        # 计算平均值
        avg_tracking_err = episode_tracking_err / MAX_STEPS
        avg_comm = episode_comm / MAX_STEPS
        
        # 收集轨迹用于可视化
        trajectory_data = None
        if episode % vis_interval == 0 or episode == 1 or episode_reward > best_reward:
            trajectory_data = collect_trajectory(agent, env, MAX_STEPS)
        
        # 保存最佳模型
        if episode_reward > best_reward:
            best_reward = episode_reward
            agent.save(SAVE_MODEL_PATH)
        
        # 更新仪表盘
        if dashboard:
            dashboard.update_episode(
                episode, episode_reward, avg_tracking_err, avg_comm,
                agent.last_losses, trajectory_data
            )
        elif episode % 20 == 0:
            print(f"Ep {episode:4d} | R:{episode_reward:7.2f} | Err:{avg_tracking_err:.4f} | Comm:{avg_comm*100:.1f}%")
    
    # 训练完成
    if dashboard:
        dashboard.finish()
    
    print(f"\n✅ Training complete! Best reward: {best_reward:.2f}")
    return agent, topology, dashboard


if __name__ == '__main__':
    agent, topology, _ = train(show_dashboard=False)
    plot_evaluation(agent, topology, num_tests=3, save_path='evaluation.png')