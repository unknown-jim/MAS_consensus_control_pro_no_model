"""
经验回放缓冲区
"""
import torch
import random
from collections import deque

from config import DEVICE, BUFFER_SIZE


class ReplayBuffer:
    """Off-policy 经验回放缓冲区"""
    
    def __init__(self, capacity=BUFFER_SIZE):
        self.buffer = deque(maxlen=capacity)
        self.device = DEVICE
    
    def push(self, state, action, reward, next_state, done):
        """存储一条经验
        
        Args:
            state: 状态张量
            action: 动作张量
            reward: 奖励值 (float)
            next_state: 下一状态张量
            done: 是否终止 (bool or float)
        """
        # 确保数据在 CPU 上存储以节省 GPU 内存
        if isinstance(state, torch.Tensor):
            state = state.detach().cpu()
        if isinstance(action, torch.Tensor):
            action = action.detach().cpu()
        if isinstance(next_state, torch.Tensor):
            next_state = next_state.detach().cpu()
        
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        """随机采样一批经验
        
        Args:
            batch_size: 批量大小
            
        Returns:
            states, actions, rewards, next_states, dones (都在正确的设备上)
        """
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # 确保所有数据都转移到正确的设备
        return (
            torch.stack(states).to(self.device),
            torch.stack(actions).to(self.device),
            torch.tensor(rewards, dtype=torch.float32, device=self.device),
            torch.stack(next_states).to(self.device),
            torch.tensor(dones, dtype=torch.float32, device=self.device)
        )
    
    def __len__(self):
        return len(self.buffer)
    
    def is_ready(self, batch_size):
        """检查缓冲区是否有足够的样本"""
        return len(self.buffer) >= batch_size
    
    def clear(self):
        """清空缓冲区"""
        self.buffer.clear()