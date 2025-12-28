"""通信拓扑（仅用于模拟物理通信范围，不作为神经网络输入）。"""

from __future__ import annotations

import numpy as np
import torch

import matplotlib.pyplot as plt
import networkx as nx

from .config import DEVICE, NUM_PINNED, TOPOLOGY_SEED, NUM_PINNED_RANGE, EXTRA_EDGE_PROB


class CommunicationTopology:
    """通信拓扑类 - 支持动态随机化。

    约定：`adj_matrix[i, j] = 1` 表示智能体 i 可以接收智能体 j 的广播。
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
        num_pinned = int(np.random.randint(self.pinned_range[0], self.pinned_range[1] + 1))
        self._build_topology(num_pinned=num_pinned)
        return self.pinned_followers

    def _compute_stats(self):
        self.in_degree = self.adj_matrix.sum(dim=1)
        self.out_degree = self.adj_matrix.sum(dim=0)
        self.num_edges = int(self.adj_matrix.sum().item())

    def get_receivers(self, sender_id: int):
        return torch.where(self.adj_matrix[:, int(sender_id)] > 0)[0].tolist()

    def can_receive(self, receiver_id: int, sender_id: int):
        return self.adj_matrix[int(receiver_id), int(sender_id)] > 0

    def get_neighbors(self, node_id: int):
        return torch.where(self.adj_matrix[int(node_id), :] > 0)[0].tolist()

    def visualize(self, save_path: str | None = None):
        """可视化拓扑结构（需要 matplotlib + networkx）。"""

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
