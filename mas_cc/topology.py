"""通信拓扑（仅用于环境模拟，不作为神经网络输入）。

该模块维护一个有向邻接矩阵 `adj_matrix`，用于表示“广播可达性”。
约定：`adj_matrix[receiver, sender] = 1` 表示 `receiver` 可以接收 `sender` 的广播。

注意：拓扑只影响环境中的通信/估计过程，不会直接作为网络输入特征。
"""

from __future__ import annotations

import math
from collections import deque
from contextlib import contextmanager
from threading import Lock

import numpy as np
import torch

import matplotlib.pyplot as plt
import networkx as nx

from .config import DEVICE, TOPOLOGY_SEED


@contextmanager
def _try_lock(lock):
    if not lock.acquire(blocking=False):
        raise RuntimeError("Environment or topology is busy collecting/updating a rollout")
    try:
        yield
    finally:
        lock.release()


class CommunicationTopology:
    """有向广播图。BFS 人口包含第 0 层 leader，平均入度只统计 follower。"""

    def __init__(self, num_followers: int, num_pinned: int | None = None,
                 seed: int = TOPOLOGY_SEED, *, pinned_mode="normal", depth_mode="random",
                 rng=None, device=DEVICE, adj_matrix=None, pinned_ids=None):
        if not self._positive_int(num_followers):
            raise ValueError("num_followers must be a positive integer")
        self.num_followers = int(num_followers)
        self.num_agents = self.num_followers + 1
        self.leader_id = 0
        self._mutation_lock = Lock()
        self.device = device
        if rng is not None and not isinstance(rng, np.random.Generator):
            raise ValueError("rng must be a numpy.random.Generator")
        self.rng = np.random.default_rng(seed) if rng is None else rng
        self.pinned_mode = pinned_mode
        self.depth_mode = depth_mode
        self._explicit_pinned = num_pinned
        self.revision = 0
        self.fixed = adj_matrix is not None
        if self.fixed:
            matrix = self._cpu_array(adj_matrix)
            if matrix.shape != (self.num_agents, self.num_agents):
                raise ValueError("adj_matrix shape must be (N+1, N+1)")
            if np.diag(matrix).any() or not np.isin(matrix, [0, 1]).all():
                raise ValueError("adj_matrix must be binary and have no self-loops")
            if pinned_ids is None:
                raise ValueError("pinned_ids required with adj_matrix")
            pins = self._cpu_array(pinned_ids)
            if (pins.ndim != 1 or pins.dtype.kind not in "iu" or not pins.size
                    or sorted(pins.tolist()) != np.flatnonzero(matrix[:, 0]).tolist()):
                raise ValueError("pinned_ids must match leader column 0 without duplicates")
            self._install(matrix)
        else:
            if pinned_ids is not None:
                raise ValueError("pinned_ids requires adj_matrix")
            self.randomize()

    @staticmethod
    def _positive_int(value):
        return isinstance(value, (int, np.integer)) and not isinstance(value, bool) and value > 0

    @staticmethod
    def _cpu_array(value):
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        return np.asarray(value)

    @classmethod
    def from_frozen(cls, adj_matrix, pinned_ids, *, device=DEVICE):
        matrix = cls._cpu_array(adj_matrix)
        if matrix.ndim != 2:
            raise ValueError("adj_matrix must be a square matrix with leader at index 0")
        return cls(matrix.shape[0]-1, adj_matrix=matrix, pinned_ids=pinned_ids, device=device)

    @property
    def adj_matrix(self):
        return self._adj.clone() if self.fixed else self._adj

    @property
    def adj(self):
        return self.adj_matrix

    @property
    def pinned_ids(self):
        return list(self._pins)

    @property
    def pinned_followers(self):
        return self.pinned_ids

    def randomize(self, num_pinned=None, *, pinned_mode=None, depth_mode=None):
        """用本图的 Generator 重建。显式构造的 pinned 数量在后续 reset 中保留。"""
        with _try_lock(self._mutation_lock):
            return self._randomize(num_pinned, pinned_mode, depth_mode)

    def _randomize(self, num_pinned, pinned_mode, depth_mode):
        if self.fixed:
            raise ValueError("Cannot randomize a frozen topology")
        mode = self.pinned_mode if pinned_mode is None else pinned_mode
        depth = self.depth_mode if depth_mode is None else depth_mode
        ratios = {"normal": (.15, .30), "low": (.02, .06)}
        if mode not in ratios or depth not in ("random", "deep", "shallow"):
            raise ValueError("Invalid pinned_mode or depth_mode")
        n = self.num_followers
        count = self._explicit_pinned if num_pinned is None else num_pinned
        if count is None:
            ratio = self.rng.uniform(*ratios[mode])
            count = int(np.clip(np.floor(n * ratio + .5), 1, n))
        elif not self._positive_int(count) or count > n:
            raise ValueError("num_pinned must be an integer in [1, N]")

        nodes = self.rng.permutation(np.arange(1, n + 1))
        pins, remaining = nodes[:count], nodes[count:]
        matrix = np.zeros((n + 1, n + 1), dtype=np.float32)
        matrix[pins, 0] = 1
        layers = np.zeros(n + 1, dtype=int)
        layers[pins] = 1
        if depth == "random":
            for i, dst in enumerate(remaining, start=count):
                matrix[dst, self.rng.choice(nodes[:i])] = 1
        elif len(remaining):
            height = 2 if depth == "shallow" else max(4, math.ceil(math.sqrt(n)))
            groups = np.array_split(remaining, min(len(remaining), height - 1))
            parents = pins
            for level, group in enumerate(groups, start=2):
                layers[group] = level
                matrix[group, self.rng.choice(parents, size=len(group))] = 1
                parents = group

        for dst in nodes:
            eligible = (nodes != dst) & (matrix[dst, nodes] == 0)
            if depth != "random":
                eligible &= layers[dst] <= layers[nodes] + 1
            senders = nodes[eligible]
            if senders.size:
                chosen = self.rng.random(senders.size) < min(1., 2. / senders.size)
                matrix[dst, senders[chosen]] = 1

        self._install(matrix)
        self.depth_mode, self.pinned_mode = depth, mode
        return self.pinned_ids

    def _install(self, matrix):
        depths = np.full(self.num_agents, -1, dtype=int)
        depths[0] = 0
        queue = deque([0])
        while queue:
            sender = queue.popleft()
            receivers = np.flatnonzero((matrix[:, sender] != 0) & (depths < 0))
            depths[receivers] = depths[sender] + 1
            queue.extend(receivers)
        self.reachable_count = int(np.count_nonzero(depths >= 0))
        self.is_connected = self.reachable_count == self.num_agents
        if not self.is_connected:
            raise ValueError("Every node must be reachable from leader 0")
        self.node_depths = tuple(depths.tolist())
        self.layer_counts = tuple(np.bincount(depths).tolist())
        self.max_depth = int(depths.max())
        degrees = matrix.sum(axis=1, dtype=int)[1:]
        self.mean_in_degree = float(degrees.mean())
        self.max_in_degree = int(degrees.max())
        self.num_edges = int(matrix.sum())
        self._pins = tuple(np.flatnonzero(matrix[:, 0]).tolist())
        self.num_pinned = len(self._pins)
        self._adj = torch.tensor(matrix, device=self.device, dtype=torch.float32)
        self.in_degree = self._adj.sum(dim=1)
        self.out_degree = self._adj.sum(dim=0)
        self.revision += 1

    @property
    def stats(self):
        return {key: getattr(self, key) for key in (
            "reachable_count", "max_depth", "layer_counts", "node_depths",
            "mean_in_degree", "max_in_degree", "is_connected")}

    def get_receivers(self, sender_id: int):
        """获取某个 sender 的接收者列表。

        Args:
            sender_id: 发送者 id。

        Returns:
            能接收该 sender 广播的节点 id 列表。
        """
        return torch.where(self._adj[:, int(sender_id)] > 0)[0].tolist()

    def can_receive(self, receiver_id: int, sender_id: int):
        """判断 receiver 是否能接收 sender 的广播。

        Args:
            receiver_id: 接收者 id。
            sender_id: 发送者 id。

        Returns:
            若可接收返回 True，否则 False。
        """
        return self._adj[int(receiver_id), int(sender_id)] > 0

    def get_neighbors(self, node_id: int):
        """获取某个节点的“可接收邻居”（入邻居）列表。

        这里的“邻居”指 `node_id` 可以接收其广播的节点集合。

        Args:
            node_id: 节点 id。

        Returns:
            邻居节点 id 列表。
        """
        return torch.where(self._adj[int(node_id), :] > 0)[0].tolist()

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
                if self._adj[i, j] > 0:
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
