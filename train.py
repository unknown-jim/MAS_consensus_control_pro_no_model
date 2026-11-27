"""CTDE 训练入口。

本脚本提供一个可直接运行的训练入口：集中训练、分散执行（CTDE）。
- Actor：仅使用本地观测（每个 follower 独立决策）
- Critic/Value：使用全局状态（以及 SAC 的联合动作）

注意：`show_dashboard=True` 时会在运行期导入 `mas_cc.dashboard`，需要在 Jupyter 环境下使用。
"""
import os
import time
from numbers import Integral

import numpy as np
import torch

from mas_cc.config import (
    NUM_FOLLOWERS, TRAIN_FOLLOWER_SCALES, MAX_STEPS, BATCH_SIZE,
    NUM_EPISODES, VIS_INTERVAL, SAVE_MODEL_PATH,
    print_config, set_seed, SEED,
    NUM_PARALLEL_ENVS, UPDATE_FREQUENCY, GRADIENT_STEPS,
    USE_AMP, DEVICE, COMM_PENALTY, THRESHOLD_MIN, THRESHOLD_MAX,
    WARMUP_STEPS,

    ALGO,
)
from mas_cc.topology import CommunicationTopology
from mas_cc.environment import BatchedModelFreeEnv, ModelFreeEnv
from mas_cc.agent import CTDESACAgent, CTDEMAPPOAgent
from mas_cc.utils import actor_only_inference, collect_trajectory, plot_evaluation


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
    train_follower_scales=TRAIN_FOLLOWER_SCALES,
    topology_mix=(0.50, 0.25, 0.25),
):
    """运行 CTDE 训练。

    Args:
        num_episodes: 训练 episode 数。
        vis_interval: 可视化/轨迹采样的间隔（每隔多少个 episode 采样一次）。
        show_dashboard: 是否启用 Jupyter 仪表盘；启用会额外依赖 `ipywidgets` 等。
        seed: 随机种子。
        profile_timing: 是否统计粗粒度耗时（step/update 的累计均值）。
        train_follower_scales: 每块各出现一次的规模，单元素表示固定规模。
        topology_mix: 普通随机、低 pinned 随机、低 pinned 深图的三项概率。

    Returns:
        (agent, topology, dashboard)：
        - agent: 训练得到的智能体实例（`CTDESACAgent` 或 `CTDEMAPPOAgent`）。
        - topology: 最后一局的冻结通信拓扑。
        - dashboard: 若启用仪表盘返回 `TrainingDashboard`，否则为 `None`。
    """
    scales = tuple(train_follower_scales)
    if not scales or any(isinstance(n, bool) or not isinstance(n, Integral) or n < 1 for n in scales):
        raise ValueError("train_follower_scales must contain positive integers")
    if len(set(scales)) != len(scales):
        raise ValueError("train_follower_scales must contain distinct scales")
    mix = np.asarray(topology_mix, dtype=float)
    if mix.shape != (3,) or not np.isfinite(mix).all() or (mix < 0).any() or not np.isclose(mix.sum(), 1):
        raise ValueError("topology_mix must contain three nonnegative probabilities summing to one")
    mix = mix / mix.sum()
    if num_episodes < 1:
        raise ValueError("num_episodes must be positive")

    is_mappo = str(ALGO).upper().strip() == "MAPPO"
    if not is_mappo and len(scales) > 1:
        raise ValueError("MASAC replay requires a single fixed follower scale")

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
    
    # 独立的验证图与初态随机流使每次预览可重复，不消耗训练抽样。
    schedule_seed, graph_seed, eval_seed = np.random.SeedSequence(seed).spawn(3)
    rng = np.random.default_rng(schedule_seed)
    eval_n = scales[0] if len(scales) == 1 else NUM_FOLLOWERS
    eval_topology = CommunicationTopology(eval_n, seed=eval_seed)
    eval_env = ModelFreeEnv(eval_topology, auto_reset=False, randomize_topology_on_reset=False)
    eval_state_seed = int(eval_seed.generate_state(1)[0])

    env_by_n = {}
    agent = None
    dashboard = None
    if show_dashboard:
        from mas_cc.dashboard import TrainingDashboard

        dashboard_graph = CommunicationTopology.from_frozen(eval_topology.adj, eval_topology.pinned_ids)
        dashboard = TrainingDashboard(num_episodes, vis_interval, topology=dashboard_graph)
        dashboard.display()

    best_reward = -float('inf')
    global_step = 0
    
    start_time = time.time()
    log_interval = 10

    # 低成本 profiling（默认关闭）
    step_time_s = 0.0
    update_time_s = 0.0
    update_calls = 0

    graph_modes = (("normal", "random"), ("low", "random"), ("low", "deep"))
    for episode in range(1, num_episodes + 1):
        offset = (episode - 1) % len(scales)
        if offset == 0:
            block = rng.permutation(scales)
        n = int(block[offset])
        pinned_mode, depth_mode = graph_modes[rng.choice(len(graph_modes), p=mix)]
        if n not in env_by_n:
            # 构造拓扑已经抽取首图，第一次 reset 只初始化物理状态。
            topology = CommunicationTopology(
                n, seed=graph_seed.spawn(1)[0], pinned_mode=pinned_mode, depth_mode=depth_mode,
            )
            env_by_n[n] = BatchedModelFreeEnv(
                topology, num_envs=NUM_PARALLEL_ENVS, max_steps=MAX_STEPS,
                auto_reset=False, randomize_topology_on_reset=False,
            )
        batched_env = env_by_n[n]
        topology = batched_env.topology
        topology.pinned_mode, topology.depth_mode = pinned_mode, depth_mode
        local_states = batched_env.reset()
        batched_env.randomize_topology = True

        if agent is None:
            if is_mappo:
                agent = CTDEMAPPOAgent(topology, use_amp=False)
            else:
                agent = CTDESACAgent(topology, use_amp=USE_AMP, train_follower_scales=scales)
        if is_mappo:
            session = agent.start_episode(batched_env)
        else:
            global_states = batched_env.get_global_state()

        episode_rewards = torch.zeros(NUM_PARALLEL_ENVS, device=DEVICE)
        episode_tracking_err = torch.zeros(NUM_PARALLEL_ENVS, device=DEVICE)
        episode_comm = torch.zeros(NUM_PARALLEL_ENVS, device=DEVICE)

        for step in range(MAX_STEPS):
            global_step += NUM_PARALLEL_ENVS

            if dashboard and step % 10 == 0:
                dashboard.update_step(step, MAX_STEPS)

            if profile_timing:
                t0 = time.perf_counter()
            if is_mappo:
                rewards, dones, infos = session.step_collect()
            else:
                actions = agent.select_action(local_states, deterministic=False)
                next_local_states, rewards, dones, infos = batched_env.step(actions)
                next_global_states = batched_env.get_global_state()
                agent.store_transitions_batch(
                    local_states, global_states, actions, rewards, next_local_states, next_global_states, dones,
                )
                local_states, global_states = next_local_states, next_global_states
            if profile_timing:
                step_time_s += time.perf_counter() - t0

            is_episode_end = step == MAX_STEPS - 1
            if is_mappo:
                do_update = session.buffer.is_full() or is_episode_end
            else:
                do_update = step > 0 and step % UPDATE_FREQUENCY == 0 and global_step > WARMUP_STEPS
            if do_update:
                if profile_timing:
                    t1 = time.perf_counter()
                if is_mappo:
                    session.flush_update(is_episode_end=is_episode_end)
                else:
                    agent.update(BATCH_SIZE, GRADIENT_STEPS)
                if profile_timing:
                    update_time_s += time.perf_counter() - t1
                    update_calls += 1

            episode_rewards += rewards
            episode_tracking_err += infos['tracking_error']
            episode_comm += infos['comm_rate']
            
        if is_mappo:
            session.close()
        
        avg_reward = episode_rewards.mean().item()
        avg_tracking_err = (episode_tracking_err / MAX_STEPS).mean().item()
        avg_comm = (episode_comm / MAX_STEPS).mean().item()
        
        improved = avg_reward > best_reward
        if improved:
            best_reward = avg_reward
            agent.save(SAVE_MODEL_PATH)

        trajectory_data = None
        if improved or episode % vis_interval == 0 or episode == 1:
            devices = [torch.cuda.current_device()] if DEVICE.type == "cuda" else []
            with torch.random.fork_rng(devices=devices), actor_only_inference(agent) as preview:
                torch.random.default_generator.manual_seed(eval_state_seed)
                if devices:
                    torch.cuda.manual_seed(eval_state_seed)
                trajectory_data = collect_trajectory(preview, eval_env, MAX_STEPS)

        if dashboard:
            dashboard.update_episode(
                episode, avg_reward, avg_tracking_err, avg_comm,
                agent.last_losses, trajectory_data, current_num_followers=n
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
    
    agent.save_actor(os.path.join(os.path.dirname(SAVE_MODEL_PATH), "final_actor.pt"))
    final_topology = CommunicationTopology.from_frozen(topology.adj, topology.pinned_ids)

    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"✅ CTDE Training Complete!")
    print(f"   Total time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"   Speed: {num_episodes/elapsed:.2f} ep/s")
    print(f"   Total steps: {global_step:,}")
    print(f"   Best reward: {best_reward:.2f}")
    print(f"{'='*60}")
    
    return agent, final_topology, dashboard


if __name__ == '__main__':
    agent, topology, _ = train(show_dashboard=False)
    # 统一从 config 读取评估保存路径（自动落到 results/.../figs/）
    from mas_cc.config import EVAL_NUM_TESTS, EVAL_SAVE_PATH
    with actor_only_inference(agent) as policy:
        plot_evaluation(policy, topology, num_tests=EVAL_NUM_TESTS, save_path=EVAL_SAVE_PATH)
