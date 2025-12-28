"""
经验回放缓冲区 - CTDE 版本

存储：
- 本地状态（用于 Actor）: (batch, num_agents, state_dim)
- 全局状态（用于 Critic）: (batch, global_state_dim)
- 联合动作: (batch, num_followers, action_dim)
- 奖励、终止标志

性能/资源优化：
- 默认在 GPU 训练时把 replay buffer 存在 CPU（可选 pinned memory），避免巨大显存占用。
- 存储 dtype 默认用 float16（GPU 训练场景），减少 RAM/带宽压力；采样后统一转 float32 计算。
"""

import torch

from config import (
    DEVICE,
    BUFFER_SIZE,
    STATE_DIM,
    ACTION_DIM,
    NUM_AGENTS,
    GLOBAL_STATE_DIM,
    REPLAY_BUFFER_DEVICE,
    REPLAY_BUFFER_DTYPE,
    REPLAY_BUFFER_PIN_MEMORY,
)


class CTDEReplayBuffer:
    """CTDE 架构的经验回放缓冲区"""

    def __init__(
        self,
        capacity: int = BUFFER_SIZE,
        num_agents: int = NUM_AGENTS,
        state_dim: int = STATE_DIM,
        action_dim: int = ACTION_DIM,
        global_state_dim: int = GLOBAL_STATE_DIM,
        storage_device: torch.device | None = None,
        storage_dtype: torch.dtype | None = None,
        pin_memory: bool | None = None,
    ):
        self.capacity = int(capacity)
        self.num_agents = int(num_agents)
        self.num_followers = self.num_agents - 1
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.global_state_dim = int(global_state_dim)

        self.storage_device = storage_device if storage_device is not None else REPLAY_BUFFER_DEVICE
        self.storage_dtype = storage_dtype if storage_dtype is not None else REPLAY_BUFFER_DTYPE

        if pin_memory is None:
            self.pin_memory = bool(REPLAY_BUFFER_PIN_MEMORY and self.storage_device.type == 'cpu')
        else:
            self.pin_memory = bool(pin_memory and self.storage_device.type == 'cpu')

        # 只有 CPU pinned -> CUDA 的拷贝才能 non_blocking
        self._non_blocking = bool(self.pin_memory and DEVICE.type == 'cuda' and self.storage_device.type == 'cpu')

        self.ptr = 0
        self.size = 0

        # 预分配存储
        # 注意：CPU pinned memory 只能用于 CPU tensor
        alloc_kwargs = {
            'device': self.storage_device,
            'dtype': self.storage_dtype,
        }
        if self.storage_device.type == 'cpu':
            alloc_kwargs['pin_memory'] = self.pin_memory

        self.local_states = torch.zeros(self.capacity, self.num_agents, self.state_dim, **alloc_kwargs)
        self.next_local_states = torch.zeros(self.capacity, self.num_agents, self.state_dim, **alloc_kwargs)

        self.global_states = torch.zeros(self.capacity, self.global_state_dim, **alloc_kwargs)
        self.next_global_states = torch.zeros(self.capacity, self.global_state_dim, **alloc_kwargs)

        self.actions = torch.zeros(self.capacity, self.num_followers, self.action_dim, **alloc_kwargs)

        # rewards/dones 也按 storage_dtype 存；采样后转 float32
        self.rewards = torch.zeros(self.capacity, **alloc_kwargs)
        self.dones = torch.zeros(self.capacity, **alloc_kwargs)

    def _to_storage(self, x: torch.Tensor) -> torch.Tensor:
        if x.device == self.storage_device and x.dtype == self.storage_dtype:
            return x
        return x.to(device=self.storage_device, dtype=self.storage_dtype, non_blocking=self._non_blocking)

    def _to_compute(self, x: torch.Tensor) -> torch.Tensor:
        # 统一返回 float32 给网络/损失，AMP 会在 forward 内自动降精度
        if x.device != DEVICE:
            x = x.to(device=DEVICE, non_blocking=self._non_blocking)
        if x.dtype != torch.float32:
            x = x.float()
        return x

    def push_batch(
        self,
        local_states: torch.Tensor,
        global_states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_local_states: torch.Tensor,
        next_global_states: torch.Tensor,
        dones: torch.Tensor,
    ):
        """批量存储经验"""
        batch_size = int(local_states.shape[0])

        # 转存储 device/dtype
        local_states_s = self._to_storage(local_states)
        global_states_s = self._to_storage(global_states)
        actions_s = self._to_storage(actions)
        rewards_s = self._to_storage(rewards)
        next_local_states_s = self._to_storage(next_local_states)
        next_global_states_s = self._to_storage(next_global_states)
        dones_s = self._to_storage(dones.float())

        if self.ptr + batch_size <= self.capacity:
            idx = slice(self.ptr, self.ptr + batch_size)
            self.local_states[idx] = local_states_s
            self.global_states[idx] = global_states_s
            self.actions[idx] = actions_s
            self.rewards[idx] = rewards_s
            self.next_local_states[idx] = next_local_states_s
            self.next_global_states[idx] = next_global_states_s
            self.dones[idx] = dones_s
        else:
            first_part = self.capacity - self.ptr
            second_part = batch_size - first_part

            self.local_states[self.ptr:] = local_states_s[:first_part]
            self.local_states[:second_part] = local_states_s[first_part:]

            self.next_local_states[self.ptr:] = next_local_states_s[:first_part]
            self.next_local_states[:second_part] = next_local_states_s[first_part:]

            self.global_states[self.ptr:] = global_states_s[:first_part]
            self.global_states[:second_part] = global_states_s[first_part:]

            self.next_global_states[self.ptr:] = next_global_states_s[:first_part]
            self.next_global_states[:second_part] = next_global_states_s[first_part:]

            self.actions[self.ptr:] = actions_s[:first_part]
            self.actions[:second_part] = actions_s[first_part:]

            self.rewards[self.ptr:] = rewards_s[:first_part]
            self.rewards[:second_part] = rewards_s[first_part:]

            self.dones[self.ptr:] = dones_s[:first_part]
            self.dones[:second_part] = dones_s[first_part:]

        self.ptr = (self.ptr + batch_size) % self.capacity
        self.size = min(self.size + batch_size, self.capacity)

    def sample(self, batch_size: int):
        """随机采样（返回 DEVICE 上的 float32 张量）"""
        if self.size <= 0:
            raise RuntimeError("Replay buffer is empty")

        # CPU 存储时 indices 必须是 CPU tensor
        if self.storage_device.type == 'cpu':
            indices = torch.randint(0, self.size, (batch_size,), device='cpu')
        else:
            indices = torch.randint(0, self.size, (batch_size,), device=self.storage_device)

        local_states = self.local_states[indices]
        global_states = self.global_states[indices]
        actions = self.actions[indices]
        rewards = self.rewards[indices]
        next_local_states = self.next_local_states[indices]
        next_global_states = self.next_global_states[indices]
        dones = self.dones[indices]

        return (
            self._to_compute(local_states),
            self._to_compute(global_states),
            self._to_compute(actions),
            self._to_compute(rewards),
            self._to_compute(next_local_states),
            self._to_compute(next_global_states),
            self._to_compute(dones),
        )

    def __len__(self):
        return self.size

    def is_ready(self, batch_size: int):
        return self.size >= batch_size


# 保留旧名称以兼容
OptimizedReplayBuffer = CTDEReplayBuffer
