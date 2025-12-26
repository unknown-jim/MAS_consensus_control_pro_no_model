"""
有向生成树通信拓扑
"""
import torch
import numpy as np

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

from config import DEVICE, NUM_PINNED, TOPOLOGY_SEED


class DirectedSpanningTreeTopology:
    r"""
    有向生成树通信拓扑
    
    拓扑结构示意 (领导者为根节点):
    
                    Leader (0)
                   /    |    \
                 F1    F2    F3     <- Pinned Followers
                / \     |     \
              F4  F5   F6    F7     <- Other Followers
                       / \
                     F8  F9
    """
    
    def __init__(self, num_followers, num_pinned=NUM_PINNED, seed=TOPOLOGY_SEED):
        self.num_followers = num_followers
        self.num_agents = num_followers + 1
        self.num_pinned = min(num_pinned, num_followers)
        self.leader_id = 0
        
        np.random.seed(seed)
        self._build_topology()
        
    def _build_topology(self):
        """构建有向生成树"""
        edges = []
        follower_ids = list(range(1, self.num_agents))
        
        # 随机选择 pinned followers
        self.pinned_followers = sorted(np.random.choice(
            follower_ids, self.num_pinned, replace=False
        ).tolist())
        
        # 领导者 -> pinned followers
        for f in self.pinned_followers:
            edges.append((self.leader_id, f))
        
        # 构建生成树
        unpinned = [f for f in follower_ids if f not in self.pinned_followers]
        connected = set(self.pinned_followers)
        
        for f in unpinned:
            parent = np.random.choice(list(connected))
            edges.append((parent, f))
            connected.add(f)
        
        # 添加额外边
        for f in follower_ids:
            potential_neighbors = [n for n in follower_ids if n != f]
            if potential_neighbors and np.random.random() < 0.3:
                neighbor = np.random.choice(potential_neighbors)
                if (neighbor, f) not in edges and (f, neighbor) not in edges:
                    edges.append((neighbor, f))
        
        # 转换为 PyTorch 格式
        src, dst = zip(*edges)
        self.edge_index = torch.tensor([src, dst], dtype=torch.long).to(DEVICE)
        self._compute_degrees()
        
    def _compute_degrees(self):
        """计算节点度数"""
        self.in_degree = torch.zeros(self.num_agents, device=DEVICE)
        self.out_degree = torch.zeros(self.num_agents, device=DEVICE)
        
        for i in range(self.edge_index.shape[1]):
            src, dst = self.edge_index[0, i].item(), self.edge_index[1, i].item()
            self.out_degree[src] += 1
            self.in_degree[dst] += 1
    
    def get_neighbors(self, node_id):
        """获取节点的入边邻居"""
        mask = self.edge_index[1] == node_id
        return self.edge_index[0, mask].tolist()
    
    def has_leader_access(self, node_id):
        """检查节点是否直接连接领导者"""
        return node_id in self.pinned_followers
    
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
        
        edges = list(zip(
            self.edge_index[0].cpu().numpy(),
            self.edge_index[1].cpu().numpy()
        ))
        G.add_edges_from(edges)
        
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
        
        nx.draw_networkx_edges(G, pos, edge_color='gray', arrows=True, arrowsize=20, alpha=0.7)
        
        labels = {0: 'L'}
        labels.update({i: f'F{i}' for i in range(1, self.num_agents)})
        nx.draw_networkx_labels(G, pos, labels, font_size=10)
        
        plt.title('Directed Spanning Tree Topology', fontsize=12)
        plt.legend(loc='upper left')
        plt.axis('off')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
        
        print(f"\n📊 Topology Statistics:")
        print(f"   Nodes: {self.num_agents}, Edges: {self.edge_index.shape[1]}")
        print(f"   Pinned Followers: {self.pinned_followers}")