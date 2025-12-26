"""
训练脚本 - 修复版
"""
import torch
import time

from config import (
    NUM_FOLLOWERS, NUM_PINNED, MAX_STEPS, BATCH_SIZE,
    NUM_EPISODES, VIS_INTERVAL, SAVE_MODEL_PATH, 
    print_config, set_seed, SEED,
    NUM_PARALLEL_ENVS, UPDATE_FREQUENCY, GRADIENT_STEPS,
    USE_AMP, DEVICE
)
from topology import DirectedSpanningTreeTopology
from environment import BatchedLeaderFollowerEnv, LeaderFollowerMASEnv
from agent import SACAgent
from dashboard import TrainingDashboard
from utils import collect_trajectory, plot_evaluation


# CUDA 优化
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def train(num_episodes=NUM_EPISODES, vis_interval=VIS_INTERVAL, 
          show_dashboard=True, seed=SEED):
    """速度优化训练 - 修复版"""
    set_seed(seed)
    print_config()
    
    # 初始化
    topology = DirectedSpanningTreeTopology(NUM_FOLLOWERS, num_pinned=NUM_PINNED)
    batched_env = BatchedLeaderFollowerEnv(topology, num_envs=NUM_PARALLEL_ENVS)
    eval_env = LeaderFollowerMASEnv(topology)
    
    agent = SACAgent(topology, use_amp=USE_AMP)
    
    dashboard = None
    if show_dashboard:
        dashboard = TrainingDashboard(num_episodes, vis_interval)
        dashboard.display()
    
    best_reward = -float('inf')
    global_step = 0
    
    start_time = time.time()
    log_interval = 10
    
    # 训练循环
    for episode in range(1, num_episodes + 1):
        states = batched_env.reset()
        
        episode_rewards = torch.zeros(NUM_PARALLEL_ENVS, device=DEVICE)
        episode_tracking_err = torch.zeros(NUM_PARALLEL_ENVS, device=DEVICE)
        episode_comm = torch.zeros(NUM_PARALLEL_ENVS, device=DEVICE)
        
        for step in range(MAX_STEPS):
            global_step += NUM_PARALLEL_ENVS
            
            # 🔧 更新 Episode 进度条
            if dashboard and step % 10 == 0:  # 每10步更新一次，避免太频繁
                dashboard.update_step(step, MAX_STEPS)
            
            actions = agent.select_action(states, deterministic=False)
            next_states, rewards, dones, infos = batched_env.step(actions)
            
            agent.store_transitions_batch(states, actions, rewards, next_states, dones)
            
            # 减少更新频率
            if step % UPDATE_FREQUENCY == 0 and step > 0:
                agent.update(BATCH_SIZE, GRADIENT_STEPS)
            
            episode_rewards += rewards
            episode_tracking_err += infos['tracking_error']
            episode_comm += infos['comm_rate']
            states = next_states
        
        avg_reward = episode_rewards.mean().item()
        avg_tracking_err = (episode_tracking_err / MAX_STEPS).mean().item()
        avg_comm = (episode_comm / MAX_STEPS).mean().item()
        
        # 减少可视化频率
        trajectory_data = None
        if episode % vis_interval == 0 or episode == 1:
            trajectory_data = collect_trajectory(agent, eval_env, MAX_STEPS)
        
        if avg_reward > best_reward:
            best_reward = avg_reward
            agent.save(SAVE_MODEL_PATH)
            trajectory_data = collect_trajectory(agent, eval_env, MAX_STEPS)
        
        if dashboard:
            dashboard.update_episode(
                episode, avg_reward, avg_tracking_err, avg_comm,
                agent.last_losses, trajectory_data
            )
        elif episode % log_interval == 0:
            elapsed = time.time() - start_time
            speed = episode / elapsed
            print(f"Ep {episode:4d} | R:{avg_reward:7.2f} | Err:{avg_tracking_err:.4f} | "
                  f"Comm:{avg_comm*100:.1f}% | Speed:{speed:.2f} ep/s | "
                  f"Steps:{global_step/1e6:.2f}M")
    
    if dashboard:
        dashboard.finish()
    
    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"✅ Training Complete!")
    print(f"   Total time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"   Speed: {num_episodes/elapsed:.2f} ep/s")
    print(f"   Total steps: {global_step:,}")
    print(f"   Best reward: {best_reward:.2f}")
    print(f"{'='*60}")
    
    return agent, topology, dashboard


if __name__ == '__main__':
    agent, topology, _ = train(show_dashboard=False)
    plot_evaluation(agent, topology, num_tests=3, save_path='evaluation.png')