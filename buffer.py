"""
经验回放缓冲区 - CTDE 版本

存储：
- 本地状态（用于 Actor）
- 全局状态（用于 Critic）
- 联合动作
- 奖励
"""
import torch
from config import DEVICE, BUFFER_SIZE, STATE_DIM, ACTION_DIM, NUM_AGENTS, GLOBAL_STATE_DIM, NUM_FOLLOWERS


class CTDEReplayBuffer:
    """CTDE 架构的经验回放缓冲区"""
    
    def __init__(self, capacity=BUFFER_SIZE, num_agents=NUM_AGENTS, 
                 state_dim=STATE_DIM, action_dim=ACTION_DIM,
                 global_state_dim=GLOBAL_STATE_DIM):
        self.capacity = capacity
        self.num_agents = num_agents
        self.num_followers = num_agents - 1
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.global_state_dim = global_state_dim
        
        self.ptr = 0
        self.size = 0
        
        # 预分配 GPU 内存
        # 本地状态（用于 Actor）
        self.local_states = torch.zeros(capacity, num_agents, state_dim, device=DEVICE)
        self.next_local_states = torch.zeros(capacity, num_agents, state_dim, device=DEVICE)
        
        # 🔧 全局状态（用于 Critic）
        self.global_states = torch.zeros(capacity, global_state_dim, device=DEVICE)
        self.next_global_states = torch.zeros(capacity, global_state_dim, device=DEVICE)
        
        # 联合动作
        self.actions = torch.zeros(capacity, self.num_followers, action_dim, device=DEVICE)
        
        # 奖励和终止标志
        self.rewards = torch.zeros(capacity, device=DEVICE)
        self.dones = torch.zeros(capacity, device=DEVICE)
    
    def push_batch(self, local_states, global_states, actions, rewards, 
                   next_local_states, next_global_states, dones):
        """批量存储经验"""
        batch_size = local_states.shape[0]
        
        if self.ptr + batch_size <= self.capacity:
            idx = slice(self.ptr, self.ptr + batch_size)
            self.local_states[idx] = local_states
            self.global_states[idx] = global_states
            self.actions[idx] = actions
            self.rewards[idx] = rewards
            self.next_local_states[idx] = next_local_states
            self.next_global_states[idx] = next_global_states
            self.dones[idx] = dones.float()
        else:
            first_part = self.capacity - self.ptr
            second_part = batch_size - first_part
            
            # 本地状态
            self.local_states[self.ptr:] = local_states[:first_part]
            self.local_states[:second_part] = local_states[first_part:]
            
            self.next_local_states[self.ptr:] = next_local_states[:first_part]
            self.next_local_states[:second_part] = next_local_states[first_part:]
            
            # 全局状态
            self.global_states[self.ptr:] = global_states[:first_part]
            self.global_states[:second_part] = global_states[first_part:]
            
            self.next_global_states[self.ptr:] = next_global_states[:first_part]
            self.next_global_states[:second_part] = next_global_states[first_part:]
            
            # 动作
            self.actions[self.ptr:] = actions[:first_part]
            self.actions[:second_part] = actions[first_part:]
            
            # 奖励
            self.rewards[self.ptr:] = rewards[:first_part]
            self.rewards[:second_part] = rewards[first_part:]
            
            # 终止
            self.dones[self.ptr:] = dones[:first_part].float()
            self.dones[:second_part] = dones[first_part:].float()
        
        self.ptr = (self.ptr + batch_size) % self.capacity
        self.size = min(self.size + batch_size, self.capacity)
    
    def sample(self, batch_size):
        """随机采样"""
        indices = torch.randint(0, self.size, (batch_size,), device=DEVICE)
        
        return (
            self.local_states[indices],      # (batch, num_agents, state_dim)
            self.global_states[indices],      # (batch, global_state_dim)
            self.actions[indices],            # (batch, num_followers, action_dim)
            self.rewards[indices],            # (batch,)
            self.next_local_states[indices],  # (batch, num_agents, state_dim)
            self.next_global_states[indices], # (batch, global_state_dim)
            self.dones[indices]               # (batch,)
        )
    
    def __len__(self):
        return self.size
    
    def is_ready(self, batch_size):
        return self.size >= batch_size


# 保留旧名称以兼容
OptimizedReplayBuffer = CTDEReplayBuffer