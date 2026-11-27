"""CTDE 训练入口。

本脚本提供一个可直接运行的训练入口：集中训练、分散执行（CTDE）。
- Actor：仅使用本地观测（每个 follower 独立决策）
- Critic/Value：使用全局状态（以及 SAC 的联合动作）

注意：`show_dashboard=True` 时会在运行期导入 `mas_cc.dashboard`，需要在 Jupyter 环境下使用。
"""
import torch
import time

from mas_cc.config import (
    NUM_FOLLOWERS, NUM_PINNED, MAX_STEPS, BATCH_SIZE,
    NUM_EPISODES, VIS_INTERVAL, SAVE_MODEL_PATH,
    print_config, set_seed, SEED,
    NUM_PARALLEL_ENVS, UPDATE_FREQUENCY, GRADIENT_STEPS,
    USE_AMP, DEVICE, COMM_PENALTY, THRESHOLD_MIN, THRESHOLD_MAX,
    WARMUP_STEPS,

    ALGO, PPO_ROLLOUT_STEPS,
)
from mas_cc.topology import CommunicationTopology
from mas_cc.environment import BatchedModelFreeEnv, ModelFreeEnv
from mas_cc.agent import CTDESACAgent, CTDEMAPPOAgent
from mas_cc.utils import collect_trajectory, plot_evaluation


torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# PyTorch 2.x：提升 matmul 精度/性能（对 Transformer/MLP 常有收益）
torch.set_float32_matmul_precision("high")


def train(
    num_episodes=NUM_EPISODES,
    vis_interval=VIS_INTERVAL,
    show_dashboard: bool = False,
    seed=SEED,
    profile_timing: bool = False,
):
    """运行 CTDE 训练。

    Args:
        num_episodes: 训练 episode 数。
        vis_interval: 可视化/轨迹采样的间隔（每隔多少个 episode 采样一次）。
        show_dashboard: 是否启用 Jupyter 仪表盘；启用会额外依赖 `ipywidgets` 等。
        seed: 随机种子。
        profile_timing: 是否统计粗粒度耗时（step/update 的累计均值）。

    Returns:
        (agent, topology, dashboard)：
        - agent: 训练得到的智能体实例（`CTDESACAgent` 或 `CTDEMAPPOAgent`）。
        - topology: 本次训练使用的 `CommunicationTopology` 实例。
        - dashboard: 若启用仪表盘返回 `TrainingDashboard`，否则为 `None`。
    """
    set_seed(seed)
    print_config()
    
    print("\n" + "="*60)
    print("🚀 CTDE Training (Centralized Training Decentralized Execution)")
    print(f"   • Algorithm: {ALGO}")
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
    
    algo = str(ALGO).upper().strip()
    is_mappo = (algo == 'MAPPO')

    if is_mappo:
        agent = CTDEMAPPOAgent(topology, use_amp=False)
        print(f"   • MAPPO Rollout Steps: {PPO_ROLLOUT_STEPS}")
    else:
        # 默认 MASAC（CTDE-SAC）
        agent = CTDESACAgent(topology, use_amp=USE_AMP)
    
    dashboard = None
    if show_dashboard:
        # 仅在需要可视化时导入（依赖 Jupyter + ipywidgets）
        from mas_cc.dashboard import TrainingDashboard

        dashboard = TrainingDashboard(num_episodes, vis_interval, topology=topology)
        dashboard.display()
    
    best_reward = -float('inf')
    best_model_state = None  # 记录最优模型状态（内存中）
    global_step = 0
    
    start_time = time.time()
    log_interval = 10

    # 低成本 profiling（默认关闭）
    step_time_s = 0.0
    update_time_s = 0.0
    update_calls = 0

    for episode in range(1, num_episodes + 1):
        
        local_states = batched_env.reset()
        # CTDE：critic/value 的输入（与 actor 的本地输入区分开）
        global_states = batched_env.get_global_state()

        episode_rewards = torch.zeros(NUM_PARALLEL_ENVS, device=DEVICE)
        episode_tracking_err = torch.zeros(NUM_PARALLEL_ENVS, device=DEVICE)
        episode_comm = torch.zeros(NUM_PARALLEL_ENVS, device=DEVICE)

        for step in range(MAX_STEPS):
            global_step += NUM_PARALLEL_ENVS

            if dashboard and step % 10 == 0:
                dashboard.update_step(step, MAX_STEPS)

            # 执行阶段：actor 只使用本地状态；训练阶段：MAPPO 还会用 global_state 估计 value
            if is_mappo:
                actions, logp, values, pre_squash_v = agent.act(
                    local_states, global_states, deterministic=False, return_pre_squash=True
                )
            else:
                actions = agent.select_action(local_states, deterministic=False)

            if profile_timing:
                t0 = time.perf_counter()
            next_local_states, rewards, dones, infos = batched_env.step(actions)
            next_global_states = batched_env.get_global_state()
            if profile_timing:
                step_time_s += (time.perf_counter() - t0)

            # 区分真实终止 (dones) 与时间截断 (time_limit_done)：
            # 时间截断保留末步 value bootstrap，避免人为制造末步价值悬崖
            time_limit_done = torch.zeros_like(dones, dtype=torch.bool)
            if step == MAX_STEPS - 1:
                time_limit_done[:] = True

            if is_mappo:
                # MAPPO：on-policy rollout（显式传入 pre_squash_v 与截断步重置前 next_global_states）
                agent.store_rollout_step(
                    local_states, global_states, actions, logp, values,
                    rewards, dones, truncated=time_limit_done,
                    next_global_states=next_global_states if time_limit_done.any() else None,
                    pre_squash_v=pre_squash_v,
                )

                do_update = agent.buffer.is_full() or (step == MAX_STEPS - 1)
                if do_update:
                    if profile_timing:
                        t1 = time.perf_counter()
                    agent.update(next_global_states=next_global_states, next_dones=dones)
                    if profile_timing:
                        update_time_s += (time.perf_counter() - t1)
                        update_calls += 1
            else:
                # MASAC（CTDE-SAC）：off-policy replay（仅真实终止置 done，时间截断保留 bootstrap）
                agent.store_transitions_batch(
                    local_states, global_states, actions, rewards,
                    next_local_states, next_global_states, dones
                )

                if step % UPDATE_FREQUENCY == 0 and step > 0 and global_step > WARMUP_STEPS:
                    if profile_timing:
                        t1 = time.perf_counter()
                    agent.update(BATCH_SIZE, GRADIENT_STEPS)
                    if profile_timing:
                        update_time_s += (time.perf_counter() - t1)
                        update_calls += 1
            
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
            # 记录最优模型状态（内存中），训练结束后再落盘
            best_model_state = {
                'actor': {k: v.clone() for k, v in agent.actor.state_dict().items()},
            }
            if is_mappo:
                best_model_state['value'] = {k: v.clone() for k, v in agent.value_net.state_dict().items()}
            else:
                best_model_state['q1'] = {k: v.clone() for k, v in agent.q1.state_dict().items()}
                best_model_state['q2'] = {k: v.clone() for k, v in agent.q2.state_dict().items()}
                best_model_state['q1_target'] = {k: v.clone() for k, v in agent.q1_target.state_dict().items()}
                best_model_state['q2_target'] = {k: v.clone() for k, v in agent.q2_target.state_dict().items()}
            trajectory_data = collect_trajectory(agent, eval_env, MAX_STEPS)
        
        if dashboard:
            dashboard.update_episode(
                episode, avg_reward, avg_tracking_err, avg_comm,
                agent.last_losses, trajectory_data
            )
        elif episode % log_interval == 0:
            elapsed = time.time() - start_time
            speed = episode / elapsed

            if profile_timing:
                # 以“每 episode”展示一个大致占比（跨 episode 累计的均值）
                avg_step_ms = (step_time_s / max(1, episode)) * 1000
                avg_update_ms = (update_time_s / max(1, episode)) * 1000
                upd_per_ep = update_calls / max(1, episode)
                timing_str = f" | step:{avg_step_ms:.0f}ms/ep | upd:{avg_update_ms:.0f}ms/ep ({upd_per_ep:.2f}/ep)"
            else:
                timing_str = ""

            print(f"Ep {episode:4d} | R:{avg_reward:7.2f} | Err:{avg_tracking_err:.4f} | "
                  f"Comm:{avg_comm*100:.1f}% | {speed:.2f} ep/s{timing_str}")
    
    if dashboard:
        dashboard.finish()
    
    # 训练完成后，保存最优模型到磁盘
    if best_model_state is not None:
        import os
        parent = os.path.dirname(SAVE_MODEL_PATH)
        if parent:
            os.makedirs(parent, exist_ok=True)
        
        if is_mappo:
            torch.save({
                'actor': best_model_state['actor'],
                'value': best_model_state['value'],
            }, SAVE_MODEL_PATH)
        else:
            torch.save({
                'actor': best_model_state['actor'],
                'q1': best_model_state['q1'],
                'q2': best_model_state['q2'],
                'q1_target': best_model_state['q1_target'],
                'q2_target': best_model_state['q2_target'],
            }, SAVE_MODEL_PATH)
        print(f"✅ Best model saved to {SAVE_MODEL_PATH}")
    
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
    # 统一从 config 读取评估保存路径（自动落到 results/.../figs/）
    from mas_cc.config import EVAL_NUM_TESTS, EVAL_SAVE_PATH
    plot_evaluation(agent, topology, num_tests=EVAL_NUM_TESTS, save_path=EVAL_SAVE_PATH)
