"""工具函数 - CTDE 版本。"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from .config import DEVICE, MAX_STEPS, THRESHOLD_MAX, THRESHOLD_MIN, TH_SCALE
from .environment import ModelFreeEnv


def _set_eval_for_inference(agent):
    """评估/可视化时临时关闭 Dropout 等随机性（不影响训练）。"""

    modules = [(agent.actor, bool(agent.actor.training))]
    if agent.value_net is not None:
        modules.append((agent.value_net, bool(agent.value_net.training)))

    for m, _ in modules:
        m.eval()

    def _restore():
        for m, was_training in modules:
            m.train(was_training)

    return _restore


@torch.no_grad()
def collect_trajectory(agent, env, max_steps: int = MAX_STEPS):
    """收集轨迹用于可视化（含通信数据）。"""

    restore = _set_eval_for_inference(agent)
    state = env.reset()

    times = [0]
    leader_pos = [env.positions[0].item()]
    leader_vel = [env.velocities[0].item()]
    follower_pos = [env.positions[1:].cpu().numpy()]
    follower_vel = [env.velocities[1:].cpu().numpy()]

    comm_rates = []
    thresholds = []

    for _ in range(int(max_steps)):
        if state.dim() == 1:
            state = state.unsqueeze(0)

        action = agent.select_action(state, deterministic=True)
        state, _, _, info = env.step(action)

        times.append(env.t)
        leader_pos.append(env.positions[0].item())
        leader_vel.append(env.velocities[0].item())
        follower_pos.append(env.positions[1:].cpu().numpy())
        follower_vel.append(env.velocities[1:].cpu().numpy())

        comm_rates.append(info["comm_rate"])

        th_raw = action[:, 1]
        th_norm = (th_raw / TH_SCALE).clamp(0.0, 1.0)
        th_env = THRESHOLD_MIN + (THRESHOLD_MAX - THRESHOLD_MIN) * th_norm
        thresholds.append(th_env.cpu().numpy())

    out = {
        "times": np.array(times),
        "leader_pos": np.array(leader_pos),
        "leader_vel": np.array(leader_vel),
        "follower_pos": np.array(follower_pos),
        "follower_vel": np.array(follower_vel),
        "comm_rates": np.array(comm_rates),
        "thresholds": np.array(thresholds),
    }

    restore()
    return out


@torch.no_grad()
def evaluate_agent(agent, env, num_episodes: int = 5):
    """评估智能体性能。"""

    restore = _set_eval_for_inference(agent)
    results = {"rewards": [], "tracking_errors": [], "comm_rates": []}

    for _ in range(int(num_episodes)):
        state = env.reset()
        episode_reward = 0
        episode_tracking_err = 0
        episode_comm = 0

        for _ in range(int(MAX_STEPS)):
            action = agent.select_action(state, deterministic=True)
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

    restore()
    return out


def plot_evaluation(agent, topology, num_tests: int = 3, save_path: str | None = None, max_plot_followers: int | None = 5):
    """绘制评估结果。"""

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
