"""评估与可视化工具（CTDE 版本）。

该模块主要提供：
- 轨迹采样（用于绘图/诊断）
- 多次评估汇总
- 评估绘图（position/velocity 曲线）

约定：评估阶段默认使用 deterministic 动作。
"""

from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from .agent import ActorOnlyPolicy, CTDEMAPPOAgent, CTDESACAgent
from .config import DEVICE, DT, MAX_STEPS, OBS_LAYOUT
from .environment import ModelFreeEnv
from .topology import CommunicationTopology


@contextmanager
def actor_only_inference(agent):
    """暂时共享 actor 权重，并保留 MAPPO 逐层关闭 dropout 的训练状态。"""
    modes = [(module, module.training) for module in agent.actor.modules()]
    try:
        if isinstance(agent, (CTDEMAPPOAgent, CTDESACAgent)):
            yield ActorOnlyPolicy(agent.actor, use_amp=agent.use_amp)
        else:
            agent.actor.eval()
            yield agent
    finally:
        for module, training in modes:
            module.training = training


@torch.no_grad()
def collect_trajectory(agent, env, max_steps: int = MAX_STEPS):
    """采样一条轨迹用于可视化（包含通信率与阈值）。

    Args:
        agent: 智能体实例（需实现 `select_action(state, deterministic=True)`）。
        env: 环境实例（通常为 `ModelFreeEnv`）。
        max_steps: 采样步数。

    Returns:
        轨迹字典，包含：
        - `num_followers/pinned_ids/topology_stats`: 本轨迹的拓扑快照
        - `times`: shape=(T+1,)
        - `leader_pos/leader_vel`: shape=(T+1,)
        - `follower_pos/follower_vel`: shape=(T+1, num_followers)
        - `comm_rates`: shape=(T,)
        - `theta_norms`: shape=(T, num_followers)（智能体输出的阈值归一化值）
    """

    state = env.reset()

    times = [0]
    leader_pos = [env.positions[0].item()]
    leader_vel = [env.velocities[0].item()]
    follower_pos = [env.positions[1:].cpu().numpy().copy()]
    follower_vel = [env.velocities[1:].cpu().numpy().copy()]

    comm_rates = []
    theta_norms = []

    with actor_only_inference(agent) as policy:
        for _ in range(int(max_steps)):
            if state.dim() == 1:
                state = state.unsqueeze(0)

            action = policy.select_action(state, deterministic=True)
            state, _, _, info = env.step(action)

            times.append(env.t)
            leader_pos.append(env.positions[0].item())
            leader_vel.append(env.velocities[0].item())
            follower_pos.append(env.positions[1:].cpu().numpy().copy())
            follower_vel.append(env.velocities[1:].cpu().numpy().copy())

            comm_rates.append(info["comm_rate"])

            # 阈值归一化值直接从 action 获取
            theta_norm = action[:, 1].clamp(0.0, 1.0)
            theta_norms.append(theta_norm.cpu().numpy().copy())

    out = {
        "num_followers": env.num_followers,
        "pinned_ids": list(env.topology.pinned_ids),
        "topology_stats": deepcopy(env.topology.stats),
        "times": np.array(times),
        "leader_pos": np.array(leader_pos),
        "leader_vel": np.array(leader_vel),
        "follower_pos": np.array(follower_pos),
        "follower_vel": np.array(follower_vel),
        "comm_rates": np.array(comm_rates),
        "theta_norms": np.array(theta_norms),
    }

    return out


@torch.no_grad()
def evaluate_agent(agent, env, num_episodes: int = 5):
    """多次运行环境并汇总评估指标。

    Args:
        agent: 智能体实例。
        env: 环境实例。
        num_episodes: 评估 episode 数。

    Returns:
        指标字典：`mean_reward/std_reward/mean_tracking_error/mean_comm_rate`。
    """

    results = {"rewards": [], "tracking_errors": [], "comm_rates": []}

    with actor_only_inference(agent) as policy:
        for _ in range(int(num_episodes)):
            state = env.reset()
            episode_reward = 0
            episode_tracking_err = 0
            episode_comm = 0

            for _ in range(int(MAX_STEPS)):
                action = policy.select_action(state, deterministic=True)
                state, reward, _, info = env.step(action)

                episode_reward += reward
                episode_tracking_err += info["tracking_error"]
                episode_comm += info["comm_rate"]

            results["rewards"].append(episode_reward)
            results["tracking_errors"].append(episode_tracking_err / MAX_STEPS)
            results["comm_rates"].append(episode_comm / MAX_STEPS)

    out = {
        "mean_reward": float(np.mean(results["rewards"])) ,
        "std_reward": float(np.std(results["rewards"])) ,
        "mean_tracking_error": float(np.mean(results["tracking_errors"])) ,
        "mean_comm_rate": float(np.mean(results["comm_rates"])) ,
    }

    return out


DEFAULT_CANONICAL_EVAL_CASES = (
    (20, "normal", "shallow", 1000),
    (50, "low", "deep", 1000),
)


@torch.no_grad()
def evaluate_canonical_score(
    agent,
    *,
    max_steps: int = MAX_STEPS,
    cases: tuple[tuple[int, str, str, int], ...] = DEFAULT_CANONICAL_EVAL_CASES,
) -> dict:
    """在固定、可复现的验证场景上评估策略，用于比较同算法跨次训练的最优权重。

    不污染全局 NumPy / PyTorch 随机流。
    """
    steps = int(max_steps)
    if steps < 1:
        raise ValueError(f"max_steps must be positive, got {max_steps}")
    case_list = tuple((int(n), str(pin), str(depth), int(seed)) for n, pin, depth, seed in cases)
    if not case_list:
        raise ValueError("cases must not be empty")

    rewards = []
    errors = []
    comms = []
    devices = [torch.cuda.current_device()] if DEVICE.type == "cuda" else []

    reward_cfg = None
    with torch.random.fork_rng(devices=devices), actor_only_inference(agent) as policy:
        for n, pin_mode, depth_mode, seed in case_list:
            topo = CommunicationTopology(n, seed=seed, pinned_mode=pin_mode, depth_mode=depth_mode)
            env = ModelFreeEnv(topo, max_steps=steps, auto_reset=False, randomize_topology_on_reset=False)
            if reward_cfg is None:
                benv = env.batched_env
                sig_payload = {
                    "dt": repr(float(DT)),
                    "pos_limit": repr(float(benv.pos_limit)),
                    "vel_limit": repr(float(benv.vel_limit)),
                    "v_scale": repr(float(benv.v_scale)),
                    "etc_vel_coef": repr(float(benv._etc_vel_coef)),
                    "age_max_steps": int(benv._age_max_steps),
                    "cooldown_steps": int(benv._cooldown_steps),
                    "comm_penalty_base": repr(float(benv.comm_penalty_base)),
                    "tracking_penalty_scale": repr(float(benv.tracking_penalty_scale)),
                    "tracking_penalty_max": repr(float(benv.tracking_penalty_max)),
                    "comm_weight_decay": repr(float(benv.comm_weight_decay)),
                    "improvement_scale": repr(float(benv.improvement_scale)),
                    "improvement_clip": repr(float(benv.improvement_clip)),
                    "info_gain_scale": repr(float(benv.info_gain_scale)),
                    "use_soft_scaling": bool(benv.use_soft_scaling),
                    "reward_min": repr(float(benv.reward_min)),
                    "reward_max": repr(float(benv.reward_max)),
                    "threshold_min": repr(float(benv._threshold_min)),
                    "threshold_max": repr(float(benv._threshold_max)),
                    "obs_layout": str(OBS_LAYOUT),
                }
                reward_cfg = hashlib.sha256(
                    json.dumps(sig_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
                ).hexdigest()[:16]
            torch.random.default_generator.manual_seed(seed)
            if devices:
                torch.cuda.manual_seed(seed)
            state = env.reset()
            ep_reward = 0.0
            ep_err = 0.0
            ep_comm = 0.0
            for _ in range(steps):
                action = policy.select_action(state, deterministic=True)
                state, reward, _, info = env.step(action)
                ep_reward += float(reward)
                ep_err += float(info["tracking_error"])
                ep_comm += float(info["comm_rate"])
            rewards.append(ep_reward)
            errors.append(ep_err / steps)
            comms.append(ep_comm / steps)

    sig_parts = [f"n{n}-{pin}-{dep}-s{s}" for n, pin, dep, s in case_list]
    eval_signature = f"canonical_v2:steps={steps}:{reward_cfg}:" + ",".join(sig_parts)
    return {
        "score": float(np.mean(rewards)),
        "mean_reward": float(np.mean(rewards)),
        "mean_tracking_error": float(np.mean(errors)),
        "mean_comm_rate": float(np.mean(comms)),
        "eval_signature": eval_signature,
    }


def plot_evaluation(agent, topology, num_tests: int = 3, save_path: str | None = None, max_plot_followers: int | None = 5):
    """绘制评估结果（位置/速度曲线）。

    Args:
        agent: 智能体实例。
        topology: `CommunicationTopology` 实例。
        num_tests: 测试次数（每次重新 reset 环境并采样一条轨迹）。
        save_path: 若提供则保存图片到该路径。
        max_plot_followers: 最多展示多少个 follower 曲线；为 None 或 <=0 则全部展示。

    Returns:
        每次测试的误差统计列表（final/avg）。
    """

    env = ModelFreeEnv(topology)

    fig, axes = plt.subplots(num_tests, 2, figsize=(14, 4 * num_tests))
    if num_tests == 1:
        axes = axes.reshape(1, -1)

    results = []

    for test_idx in range(int(num_tests)):
        traj = collect_trajectory(agent, env, MAX_STEPS)

        pos_errors = (traj["follower_pos"] - traj["leader_pos"][:, np.newaxis]) ** 2
        final_error = float(np.mean(pos_errors[-1]))
        avg_error = float(np.mean(pos_errors))

        results.append({"final_error": final_error, "avg_error": avg_error})

        n_followers = int(traj["follower_pos"].shape[1])
        if (max_plot_followers is None) or (int(max_plot_followers) <= 0):
            n_show = n_followers
        else:
            n_show = min(int(max_plot_followers), n_followers)

        ax1 = axes[test_idx, 0]
        ax1.plot(traj["times"], traj["leader_pos"], "r-", linewidth=2.5, label="Leader")
        colors = plt.cm.Blues(np.linspace(0.3, 0.9, n_followers))
        for i in range(n_show):
            ax1.plot(
                traj["times"],
                traj["follower_pos"][:, i],
                color=colors[i],
                alpha=0.8,
                linewidth=1.2,
                label=f"F{i+1}",
            )
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Position")
        ax1.set_title(f"Test {test_idx+1}: Position (Final Err: {final_error:.4f})")
        ax1.legend(loc="upper right", fontsize=8)
        ax1.grid(True, alpha=0.3)

        ax2 = axes[test_idx, 1]
        ax2.plot(traj["times"], traj["leader_vel"], "r-", linewidth=2.5, label="Leader")
        for i in range(n_show):
            ax2.plot(
                traj["times"],
                traj["follower_vel"][:, i],
                color=colors[i],
                alpha=0.8,
                linewidth=1.2,
            )
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Velocity")
        ax2.set_title(f"Test {test_idx+1}: Velocity")
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        parent = os.path.dirname(str(save_path))
        if parent:
            os.makedirs(parent, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"📁 Figure saved to {save_path}")

    plt.show()

    print("\n📊 CTDE Evaluation Results:")
    print("-" * 40)
    for i, r in enumerate(results):
        print(f"Test {i+1}: Final Err = {r['final_error']:.4f}, Avg Err = {r['avg_error']:.4f}")

    return results
