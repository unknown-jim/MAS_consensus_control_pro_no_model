"""通信拓扑（仅用于环境模拟，不作为神经网络输入）。

该模块维护一个有向邻接矩阵 `adj_matrix`，用于表示“广播可达性”。
约定：`adj_matrix[receiver, sender] = 1` 表示 `receiver` 可以接收 `sender` 的广播。

注意：拓扑只影响环境中的通信/估计过程，不会直接作为网络输入特征。
"""

from __future__ import annotations

import numpy as np
import torch

import matplotlib.pyplot as plt
import networkx as nx

from .config import DEVICE, NUM_PINNED, TOPOLOGY_SEED, NUM_PINNED_RANGE, EXTRA_EDGE_PROB


class CommunicationTopology:
    """通信拓扑（可随机化）。

    拓扑会生成以下核心属性：
    - `adj_matrix`: 形状为 `(num_agents, num_agents)` 的邻接矩阵（receiver, sender）。
    - `pinned_followers`: 直接与 leader 相连的 follower 列表。

    Args:
        num_followers: follower 数量（不含 leader）。
        num_pinned: pinned follower 数量（会被截断到 `[0, num_followers]`）。
        seed: 随机种子（影响 pinned 选择与随机边）。

    Attributes:
        num_agents: 智能体总数（`num_followers + 1`）。
        leader_id: leader 的节点 id（固定为 0）。
    """

    def __init__(self, num_followers: int, num_pinned: int = NUM_PINNED, seed: int = TOPOLOGY_SEED):
        self.num_followers = int(num_followers)
        self.num_agents = self.num_followers + 1
        self.num_pinned = min(int(num_pinned), self.num_followers)
        self.leader_id = 0

        self.pinned_range = NUM_PINNED_RANGE
        self.extra_edge_prob = EXTRA_EDGE_PROB

        np.random.seed(int(seed))
        self._build_topology()

    def _build_topology(self, num_pinned: int | None = None):
        follower_ids = list(range(1, self.num_agents))

        if num_pinned is not None:
            self.num_pinned = min(int(num_pinned), self.num_followers)

        self.pinned_followers = sorted(
            np.random.choice(follower_ids, self.num_pinned, replace=False).tolist()
        )

        self.adj_matrix = torch.zeros(self.num_agents, self.num_agents, device=DEVICE)

        for f in self.pinned_followers:
            self.adj_matrix[f, self.leader_id] = 1.0

        edges = []
        unpinned = [f for f in follower_ids if f not in self.pinned_followers]
        connected = set(self.pinned_followers)

        for f in unpinned:
            parent = int(np.random.choice(list(connected)))
            edges.append((parent, f))
            connected.add(f)

        for src, dst in edges:
            self.adj_matrix[dst, src] = 1.0

        for i in follower_ids:
            for j in follower_ids:
                if i != j and self.adj_matrix[i, j] == 0:
                    if np.random.random() < float(self.extra_edge_prob):
                        self.adj_matrix[i, j] = 1.0

        self._compute_stats()

    def randomize(self):
        """随机化拓扑结构。

        会在 `NUM_PINNED_RANGE` 内随机采样 pinned 数量，并重建邻接矩阵。

        Returns:
            新的 pinned follower 列表（按升序）。
        """
        num_pinned = int(np.random.randint(self.pinned_range[0], self.pinned_range[1] + 1))
        self._build_topology(num_pinned=num_pinned)
        return self.pinned_followers

    def _compute_stats(self):
        self.in_degree = self.adj_matrix.sum(dim=1)
        self.out_degree = self.adj_matrix.sum(dim=0)
        self.num_edges = int(self.adj_matrix.sum().item())

    def get_receivers(self, sender_id: int):
        """获取某个 sender 的接收者列表。

        Args:
            sender_id: 发送者 id。

        Returns:
            能接收该 sender 广播的节点 id 列表。
        """
        return torch.where(self.adj_matrix[:, int(sender_id)] > 0)[0].tolist()

    def can_receive(self, receiver_id: int, sender_id: int):
        """判断 receiver 是否能接收 sender 的广播。

        Args:
            receiver_id: 接收者 id。
            sender_id: 发送者 id。

        Returns:
            若可接收返回 True，否则 False。
        """
        return self.adj_matrix[int(receiver_id), int(sender_id)] > 0

    def get_neighbors(self, node_id: int):
        """获取某个节点的“可接收邻居”（入邻居）列表。

        这里的“邻居”指 `node_id` 可以接收其广播的节点集合。

        Args:
            node_id: 节点 id。

        Returns:
            邻居节点 id 列表。
        """
        return torch.where(self.adj_matrix[int(node_id), :] > 0)[0].tolist()

    def visualize(self, save_path: str | None = None):
        """可视化拓扑结构。

        Args:
            save_path: 若提供则保存图片到该路径，否则仅展示。

        Raises:
            ImportError: 当缺少 `matplotlib` 或 `networkx` 时会在导入阶段报错。
        """

        G = nx.DiGraph()
        G.add_nodes_from(range(self.num_agents))

        for i in range(self.num_agents):
            for j in range(self.num_agents):
                if self.adj_matrix[i, j] > 0:
                    G.add_edge(j, i)

        pos = nx.spring_layout(G, seed=42, k=2)
        pos[0] = np.array([0.5, 1.0])

        plt.figure(figsize=(10, 8))

        nx.draw_networkx_nodes(G, pos, nodelist=[0], node_color="gold", node_size=800, label="Leader")
        nx.draw_networkx_nodes(
            G, pos, nodelist=self.pinned_followers, node_color="lightgreen", node_size=500, label="Pinned"
        )
        other_nodes = [n for n in range(1, self.num_agents) if n not in self.pinned_followers]
        nx.draw_networkx_nodes(G, pos, nodelist=other_nodes, node_color="lightblue", node_size=400, label="Others")

        nx.draw_networkx_edges(G, pos, edge_color="gray", arrows=True, arrowsize=15, alpha=0.5)

        labels = {0: "L"}
        labels.update({i: f"F{i}" for i in range(1, self.num_agents)})
        nx.draw_networkx_labels(G, pos, labels, font_size=10)

        plt.title("Communication Topology (for simulation only)", fontsize=12)
        plt.legend(loc="upper left")
        plt.axis("off")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()

        print("\n📊 Topology Statistics (Simulation Only):")
        print(f"   Nodes: {self.num_agents}, Edges: {self.num_edges}")
        print(f"   Pinned Followers: {self.pinned_followers}")
        print("   ⚠️ Note: This topology is NOT used by neural networks!")
