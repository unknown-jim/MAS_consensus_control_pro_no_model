"""
通信拓扑 - 支持动态随机化（仅用于模拟物理通信范围，不作为神经网络输入）
"""
import torch
import numpy as np

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

from config import DEVICE, NUM_PINNED, TOPOLOGY_SEED, NUM_PINNED_RANGE, EXTRA_EDGE_PROB


class CommunicationTopology:
    """
    通信拓扑类 - 支持动态随机化
    
    关键区别：
    - 拓扑仅用于模拟"谁能接收到谁的广播"
    - 神经网络不使用拓扑结构（无 edge_index）
    - 智能体不知道拓扑，只知道收到了什么数据
    - 支持每 episode 随机重建拓扑
    """
    
    def __init__(self, num_followers, num_pinned=NUM_PINNED, seed=TOPOLOGY_SEED):
        self.num_followers = num_followers
        self.num_agents = num_followers + 1
        self.num_pinned = min(num_pinned, num_followers)
        self.leader_id = 0
        
        # 随机化参数
        self.pinned_range = NUM_PINNED_RANGE
        self.extra_edge_prob = EXTRA_EDGE_PROB
        
        np.random.seed(seed)
        self._build_topology()
    
    def _build_topology(self, num_pinned=None):
        """构建基础连接关系（用于模拟通信）"""
        follower_ids = list(range(1, self.num_agents))
        
        # 使用指定的 num_pinned 或默认值
        if num_pinned is not None:
            self.num_pinned = min(num_pinned, self.num_followers)
        
        # 随机选择 pinned followers（可以接收领导者信息）
        self.pinned_followers = sorted(np.random.choice(
            follower_ids, self.num_pinned, replace=False
        ).tolist())
        
        # 构建邻接矩阵（用于模拟谁能接收谁的广播）
        # adj_matrix[i, j] = 1 表示智能体 i 可以接收智能体 j 的广播
        self.adj_matrix = torch.zeros(self.num_agents, self.num_agents, device=DEVICE)
        
        # 领导者 -> pinned followers
        for f in self.pinned_followers:
            self.adj_matrix[f, self.leader_id] = 1.0
        
        # 构建跟随者之间的连接（确保连通性）
        edges = []
        unpinned = [f for f in follower_ids if f not in self.pinned_followers]
        connected = set(self.pinned_followers)
        
        # 确保每个 unpinned follower 至少有一个连接
        for f in unpinned:
            parent = np.random.choice(list(connected))
            edges.append((parent, f))
            connected.add(f)
        
        # 添加跟随者之间的连接
        for src, dst in edges:
            self.adj_matrix[dst, src] = 1.0
        
        # 添加额外的随机连接（增加连通性）
        for i in follower_ids:
            for j in follower_ids:
                if i != j and self.adj_matrix[i, j] == 0:
                    if np.random.random() < self.extra_edge_prob:
                        self.adj_matrix[i, j] = 1.0
        
        # 计算统计信息
        self._compute_stats()
    
    def randomize(self):
        """
        🔧 随机化拓扑结构（保证连通性）
        
        随机化内容：
        - Pinned followers 数量（在 NUM_PINNED_RANGE 范围内）
        - Pinned followers 选择
        - 跟随者之间的连接
        
        Returns:
            pinned_followers: 新的 pinned followers 列表
        """
        # 随机选择 pinned followers 数量
        num_pinned = np.random.randint(self.pinned_range[0], self.pinned_range[1] + 1)
        
        # 重建拓扑
        self._build_topology(num_pinned=num_pinned)
        
        return self.pinned_followers
    
    def _compute_stats(self):
        """计算拓扑统计信息"""
        self.in_degree = self.adj_matrix.sum(dim=1)
        self.out_degree = self.adj_matrix.sum(dim=0)
        self.num_edges = int(self.adj_matrix.sum().item())
    
    def get_receivers(self, sender_id):
        """获取能接收 sender 广播的智能体列表"""
        return torch.where(self.adj_matrix[:, sender_id] > 0)[0].tolist()
    
    def can_receive(self, receiver_id, sender_id):
        """检查 receiver 是否能接收 sender 的广播"""
        return self.adj_matrix[receiver_id, sender_id] > 0
    
    def get_neighbors(self, node_id):
        """获取节点可以接收数据的邻居列表"""
        return torch.where(self.adj_matrix[node_id, :] > 0)[0].tolist()
    
    def visualize(self, save_path=None):
        """可视化拓扑结构"""
        if not HAS_MATPLOTLIB:
            print("matplotlib not available")
            return
        
        try:
            import networkx as nx
        except ImportError:
            print("Please install networkx: pip install networkx")
            return
        
        G = nx.DiGraph()
        G.add_nodes_from(range(self.num_agents))
        
        # 添加边 (j -> i 表示 i 可以接收 j 的数据)
        for i in range(self.num_agents):
            for j in range(self.num_agents):
                if self.adj_matrix[i, j] > 0:
                    G.add_edge(j, i)
        
        pos = nx.spring_layout(G, seed=42, k=2)
        pos[0] = np.array([0.5, 1.0])
        
        plt.figure(figsize=(10, 8))
        
        nx.draw_networkx_nodes(G, pos, nodelist=[0], 
                              node_color='gold', node_size=800, label='Leader')
        nx.draw_networkx_nodes(G, pos, nodelist=self.pinned_followers,
                              node_color='lightgreen', node_size=500, label='Pinned')
        other_nodes = [n for n in range(1, self.num_agents) if n not in self.pinned_followers]
        nx.draw_networkx_nodes(G, pos, nodelist=other_nodes,
                              node_color='lightblue', node_size=400, label='Others')
        
        nx.draw_networkx_edges(G, pos, edge_color='gray', arrows=True, arrowsize=15, alpha=0.5)
        
        labels = {0: 'L'}
        labels.update({i: f'F{i}' for i in range(1, self.num_agents)})
        nx.draw_networkx_labels(G, pos, labels, font_size=10)
        
        plt.title('Communication Topology (for simulation only)', fontsize=12)
        plt.legend(loc='upper left')
        plt.axis('off')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
        
        print(f"\n📊 Topology Statistics (Simulation Only):")
        print(f"   Nodes: {self.num_agents}, Edges: {self.num_edges}")
        print(f"   Pinned Followers: {self.pinned_followers}")
        print(f"   ⚠️ Note: This topology is NOT used by neural networks!")


# 保留旧名称以兼容
DirectedSpanningTreeTopology = CommunicationTopology