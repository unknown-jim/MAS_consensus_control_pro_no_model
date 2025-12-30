"""多智能体一致性控制（CTDE）包。

该包包含训练/评估所需的核心模块：环境、拓扑、网络、算法、缓冲区与可视化。

CTDE = Centralized Training, Decentralized Execution（集中训练、分散执行）。
"""

from .agent import CTDESACAgent, CTDEMAPPOAgent
from .environment import BatchedModelFreeEnv, ModelFreeEnv
from .topology import CommunicationTopology

__all__ = [
    "CTDESACAgent",
    "CTDEMAPPOAgent",
    "BatchedModelFreeEnv",
    "ModelFreeEnv",
    "CommunicationTopology",
]
