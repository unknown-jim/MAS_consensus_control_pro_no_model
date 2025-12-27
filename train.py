"""
训练脚本 - CTDE 架构版本
"""
import torch
import time

from config import (
    NUM_FOLLOWERS, NUM_PINNED, MAX_STEPS, BATCH_SIZE,
    NUM_EPISODES, VIS_INTERVAL, SAVE_MODEL_PATH, 
    print_config, set_seed, SEED,
    NUM_PARALLEL_ENVS, UPDATE_FREQUENCY, GRADIENT_STEPS,
    USE_AMP, DEVICE, COMM_PENALTY, THRESHOLD_MIN, THRESHOLD_MAX,
    WARMUP_STEPS
)
from topology import CommunicationTopology
from environment import BatchedModelFreeEnv, ModelFreeEnv
from agent import CTDESACAgent
from utils import collect_trajectory, plot_evaluation

try:
    from dashboard import TrainingDashboard
    HAS_DASHBOARD = True
except ImportError:
    HAS_DASHBOARD = False
    print("⚠️ Dashboard not available, using console logging")


torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def train(num_episodes=NUM_EPISODES, vis_interval=VIS_INTERVAL, 
          show_dashboard=True, seed=SEED):
    """CTDE 训练"""
    set_seed(seed)
    print_config()
    
    print("\n" + "="*60)
    print("🚀 CTDE Training (Centralized Training Decentralized Execution)")
    print("   • Actor: Decentralized (local observation only)")
    print("   • Critic: Centralized (global state + joint action)")
    print("   • Execution: Each agent uses only local information")
    print(f"   • Warmup Steps: {WARMUP_STEPS}")
    print("="*60)
    print(f"\n📡 Communication Settings:")
    print(f"   Comm Penalty: {COMM_PENALTY}")
    print(f"   Threshold Range: [{THRESHOLD_MIN}, {THRESHOLD_MAX}]")
    print()
    
    # 初始化
    topology = CommunicationTopology(NUM_FOLLOWERS, num_pinned=NUM_PINNED)
    batched_env = BatchedModelFreeEnv(topology, num_envs=NUM_PARALLEL_ENVS)
    eval_env = ModelFreeEnv(topology)
    
    agent = CTDESACAgent(topology, use_amp=USE_AMP)
    
    dashboard = None
    if show_dashboard and HAS_DASHBOARD:
        dashboard = TrainingDashboard(num_episodes, vis_interval, topology=topology)
        dashboard.display()
    
    best_reward = -float('inf')
    global_step = 0
    
    start_time = time.time()
    log_interval = 10
    
    for episode in range(1, num_episodes + 1):
        
        local_states = batched_env.reset()
        global_states = batched_env.get_global_state()  # 🔧 获取全局状态
        
        episode_rewards = torch.zeros(NUM_PARALLEL_ENVS, device=DEVICE)
        episode_tracking_err = torch.zeros(NUM_PARALLEL_ENVS, device=DEVICE)
        episode_comm = torch.zeros(NUM_PARALLEL_ENVS, device=DEVICE)
        
        for step in range(MAX_STEPS):
            global_step += NUM_PARALLEL_ENVS
            
            if dashboard and step % 10 == 0:
                dashboard.update_step(step, MAX_STEPS)
            
            # 🔧 Actor 只用本地状态
            actions = agent.select_action(local_states, deterministic=False)
            next_local_states, rewards, dones, infos = batched_env.step(actions)
            next_global_states = batched_env.get_global_state()  # 🔧 获取下一步全局状态
            
            # 🔧 存储时包含全局状态
            # 时间截断：最后一步视为终止，避免跨 episode 的 bootstrapping 偏差
            time_limit_done = torch.zeros_like(dones)
            if step == MAX_STEPS - 1:
                time_limit_done[:] = True
            store_dones = dones | time_limit_done

            agent.store_transitions_batch(
                local_states, global_states, actions, rewards,
                next_local_states, next_global_states, store_dones
            )
            
            if step % UPDATE_FREQUENCY == 0 and step > 0 and global_step > WARMUP_STEPS:
                agent.update(BATCH_SIZE, GRADIENT_STEPS)
            
            episode_rewards += rewards
            episode_tracking_err += infos['tracking_error']
            episode_comm += infos['comm_rate']
            
            local_states = next_local_states
            global_states = next_global_states
        
        avg_reward = episode_rewards.mean().item()
        avg_tracking_err = (episode_tracking_err / MAX_STEPS).mean().item()
        avg_comm = (episode_comm / MAX_STEPS).mean().item()
        
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
                  f"Comm:{avg_comm*100:.1f}% | {speed:.2f} ep/s")
    
    if dashboard:
        dashboard.finish()
    
    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"✅ CTDE Training Complete!")
    print(f"   Total time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"   Speed: {num_episodes/elapsed:.2f} ep/s")
    print(f"   Total steps: {global_step:,}")
    print(f"   Best reward: {best_reward:.2f}")
    print(f"{'='*60}")
    
    return agent, topology, dashboard


if __name__ == '__main__':
    agent, topology, _ = train(show_dashboard=False)
    plot_evaluation(agent, topology, num_tests=3, save_path='evaluation_ctde.png')