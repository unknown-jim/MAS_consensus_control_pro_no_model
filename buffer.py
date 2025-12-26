"""
经验回放缓冲区 - GPU 优化版
"""
import torch
from config import DEVICE, BUFFER_SIZE, STATE_DIM, ACTION_DIM, NUM_AGENTS


class OptimizedReplayBuffer:
    """
    GPU 预分配的高效经验回放缓冲区
    
    支持批量存储和采样
    """
    
    def __init__(self, capacity=BUFFER_SIZE, num_agents=NUM_AGENTS, 
                 state_dim=STATE_DIM, action_dim=ACTION_DIM):
        self.capacity = capacity
        self.num_agents = num_agents
        self.num_followers = num_agents - 1
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        self.ptr = 0
        self.size = 0
        
        # 预分配 GPU 内存
        self.states = torch.zeros(capacity, num_agents, state_dim, device=DEVICE)
        self.actions = torch.zeros(capacity, self.num_followers, action_dim, device=DEVICE)
        self.rewards = torch.zeros(capacity, device=DEVICE)
        self.next_states = torch.zeros(capacity, num_agents, state_dim, device=DEVICE)
        self.dones = torch.zeros(capacity, device=DEVICE)
    
    def push(self, state, action, reward, next_state, done):
        """存储单条经验"""
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = float(done)
        
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def push_batch(self, states, actions, rewards, next_states, dones):
        """
        批量存储经验
        
        Args:
            states: (batch, num_agents, state_dim)
            actions: (batch, num_followers, action_dim)
            rewards: (batch,)
            next_states: (batch, num_agents, state_dim)
            dones: (batch,)
        """
        batch_size = states.shape[0]
        
        # 计算存储位置
        if self.ptr + batch_size <= self.capacity:
            # 不需要环绕
            idx = slice(self.ptr, self.ptr + batch_size)
            self.states[idx] = states
            self.actions[idx] = actions
            self.rewards[idx] = rewards
            self.next_states[idx] = next_states
            self.dones[idx] = dones.float()
        else:
            # 需要环绕处理
            first_part = self.capacity - self.ptr
            second_part = batch_size - first_part
            
            self.states[self.ptr:] = states[:first_part]
            self.states[:second_part] = states[first_part:]
            
            self.actions[self.ptr:] = actions[:first_part]
            self.actions[:second_part] = actions[first_part:]
            
            self.rewards[self.ptr:] = rewards[:first_part]
            self.rewards[:second_part] = rewards[first_part:]
            
            self.next_states[self.ptr:] = next_states[:first_part]
            self.next_states[:second_part] = next_states[first_part:]
            
            self.dones[self.ptr:] = dones[:first_part].float()
            self.dones[:second_part] = dones[first_part:].float()
        
        self.ptr = (self.ptr + batch_size) % self.capacity
        self.size = min(self.size + batch_size, self.capacity)
    
    def sample(self, batch_size):
        """随机采样"""
        indices = torch.randint(0, self.size, (batch_size,), device=DEVICE)
        
        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.dones[indices]
        )
    
    def __len__(self):
        return self.size
    
    def is_ready(self, batch_size):
        return self.size >= batch_size