"""算法实现（CTDE-SAC / CTDE-MAPPO）。"""

from __future__ import annotations

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows fallback
    fcntl = None
import copy
import io
import math
import os
import pickle
import stat
import struct
import tempfile
import weakref
import zipfile
from contextlib import ExitStack, contextmanager
from threading import Lock

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from .buffer import CTDEReplayBuffer
from .config import (
    ACTION_DIM,
    ACTOR_MICROBATCH_MAX_FOLLOWER_ROWS,
    ALPHA_LR,
    BATCH_SIZE,
    DEVICE,
    GAMMA,
    GRADIENT_STEPS,
    HIDDEN_DIM,
    INIT_ALPHA,
    LEARNING_RATE,
    NUM_ATTENTION_HEADS,
    global_state_dim_for,
    NUM_TRANSFORMER_LAYERS,
    OBS_LAYOUT,
    OBS_VERSION,
    normalize_algo_name,
    TOP_K,
    POLICY_DELAY,
    PPO_CLIP_EPS,
    PPO_ENTROPY_COEF,
    PPO_EPOCHS,
    PPO_GAE_LAMBDA,
    PPO_LR,
    PPO_MAX_GRAD_NORM,
    PPO_MINIBATCH_SIZE,
    PPO_ROLLOUT_STEPS,
    PPO_TARGET_KL,
    PPO_VALUE_COEF,
    STATE_DIM,
    TARGET_ENTROPY_RATIO,
    TARGET_UPDATE_INTERVAL,
    TAU,
)
from .networks import CentralizedCritic, CentralizedValue, DecentralizedActor
from .topology import _try_lock


def _actor_schema(hidden_dim, algorithm):
    return {
        "obs_version": OBS_VERSION, "top_k": TOP_K, "state_dim": STATE_DIM,
        "hidden_dim": hidden_dim, "num_heads": NUM_ATTENTION_HEADS,
        "num_layers": NUM_TRANSFORMER_LAYERS, "algorithm": algorithm,
    }


def _save_checkpoint(path, actor, algorithm, **training_state):
    parent = os.path.dirname(str(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    schema = _actor_schema(actor.local_encoder[0].out_features, algorithm)
    torch.save({**schema, "actor": actor.state_dict(), **training_state}, path)


_CANONICAL_LOCKS_GUARD = Lock()
_CANONICAL_PATH_LOCKS: dict[str, Lock] = {}


@contextmanager
def _canonical_file_lock(path_str: str):
    """对同一目标最优权重路径加进程内线程锁 + 跨进程文件锁。"""
    abs_path = os.path.abspath(path_str)
    parent = os.path.dirname(abs_path) or "."
    os.makedirs(parent, exist_ok=True)
    with _CANONICAL_LOCKS_GUARD:
        tlock = _CANONICAL_PATH_LOCKS.setdefault(abs_path, Lock())
    lock_path = abs_path + ".lock"
    with tlock:
        if fcntl is None:
            yield
            return
        with open(lock_path, "a+") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _to_builtin_primitive(value):
    """将常见标量（含 NumPy 标量/字符串）显式转为 Python 内建类型，确保 weights_only=True 可加载。"""
    if value is None:
        return None
    if isinstance(value, (str, np.str_)):
        return str(value)
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        v = float(value)
        return v if math.isfinite(v) else None
    return None


_OOM_TYPES = tuple(
    exc for exc in (
        MemoryError,
        getattr(torch, "OutOfMemoryError", None),
        getattr(torch.cuda, "OutOfMemoryError", None),
    )
    if isinstance(exc, type) and issubclass(exc, BaseException)
)


def _atomic_save_checkpoint(path, actor, algorithm, **training_state):
    path_str = str(path)
    parent = os.path.dirname(path_str) or "."
    os.makedirs(parent, exist_ok=True)
    schema = _actor_schema(actor.local_encoder[0].out_features, algorithm)
    payload = {**schema, "actor": {k: v.detach().cpu() for k, v in actor.state_dict().items()}, **training_state}
    tmp_file = None
    replaced = False
    try:
        tmp_file = tempfile.NamedTemporaryFile(prefix=".tmp_ckpt_", suffix=".pt", dir=parent, delete=False)
        try:
            torch.save(payload, tmp_file)
            tmp_file.flush()
        finally:
            tmp_file.close()
        os.replace(tmp_file.name, path_str)
        replaced = True
    finally:
        if tmp_file is not None:
            try:
                if not tmp_file.closed:
                    tmp_file.close()
            finally:
                if not replaced:
                    try:
                        os.remove(tmp_file.name)
                    except OSError:
                        pass


def _read_file_bytes_exact(path_str: str) -> bytes:
    """完整读取文件字节；若常规文件实际读取字节数与 fstat 不一致则判为短读并抛 OSError。"""
    with open(path_str, "rb") as f:
        expected_size = None
        try:
            fd = f.fileno()
        except (AttributeError, io.UnsupportedOperation):
            fd = None
        if isinstance(fd, int) and fd >= 0:
            st = os.fstat(fd)
            if stat.S_ISREG(st.st_mode):
                expected_size = int(st.st_size)
        chunks = []
        while True:
            chunk = f.read(65536)
            if not chunk:
                break
            chunks.append(chunk)
        data = b"".join(chunks)
        if expected_size is not None and len(data) != expected_size:
            raise OSError(f"Short read while verifying checkpoint {path_str}: got {len(data)} of {expected_size} bytes")
        return data


def _is_corruption_load_error(exc: BaseException) -> bool:
    if isinstance(exc, _OOM_TYPES):
        return False
    if isinstance(
        exc,
        (ValueError, EOFError, IndexError, struct.error, pickle.UnpicklingError, zipfile.BadZipFile, KeyError, TypeError),
    ):
        return True
    if isinstance(exc, RuntimeError):
        msg = str(exc).lower()
        format_markers = (
            "pytorchstreamreader",
            "failed finding central directory",
            "invalid load key",
            "unexpected eof",
            "zip archive",
            "magic number",
            "corrupt",
            "inline_container",
        )
        if any(tok in msg for tok in ("memory", "alloc")):
            return False
        return any(m in msg for m in format_markers)
    return False


def _actor_weights_complete(weights: dict, actor) -> bool:
    expected = actor.state_dict()
    if not isinstance(weights, dict) or weights.keys() != expected.keys():
        return False
    for key, param in expected.items():
        tensor = weights.get(key)
        if (
            not isinstance(tensor, torch.Tensor)
            or tensor.is_meta
            or tensor.layout != torch.strided
            or tensor.shape != param.shape
        ):
            return False
    try:
        probe = copy.deepcopy(actor).to("cpu")
        probe.load_state_dict(weights, strict=True)
    except _OOM_TYPES:
        raise
    except RuntimeError as exc:
        if any(tok in str(exc).lower() for tok in ("memory", "alloc")):
            raise
        return False
    except (ValueError, TypeError):
        return False
    return True


def save_canonical_best_actor(
    path,
    actor,
    algorithm: str,
    *,
    score: float,
    metrics: dict | None = None,
    obs_layout: str = OBS_LAYOUT,
    eval_signature: str | None = None,
    allow_eval_signature_replace: bool = False,
) -> dict:
    """按单算法唯一路径维护仓库最优 Actor 权重（带跨进程锁与原子替换）。

    规则：
    - 同算法、同观测/网络协议且评估签名完全一致时，仅当 `score > existing_score` 才替换；
    - 评估签名不一致（含一侧缺失）时默认保留原件不覆盖；
    - 已有文件经重读复核确实损坏、缺少参数或协议变化时可替换恢复；临时读写权限/IO/短读/OOM 失败则保留原件并抛错。
    """
    algo = normalize_algo_name(algorithm)
    if isinstance(score, (bool, np.bool_)) or not isinstance(score, (int, float, np.number)) or not math.isfinite(float(score)):
        raise ValueError(f"Canonical score must be a finite number, got {score!r}")
    score_val = float(score)
    width = int(actor.local_encoder[0].out_features)
    expected_schema = _actor_schema(width, algo)
    cur_sig = "" if eval_signature is None else str(eval_signature)

    path_str = str(path)
    previous_score = None
    reason = "created"

    _oom_types = tuple(
        exc for exc in (MemoryError, getattr(torch, "OutOfMemoryError", None),
                        getattr(torch.cuda, "OutOfMemoryError", None))
        if isinstance(exc, type) and issubclass(exc, BaseException)
    )

    with _canonical_file_lock(path_str):
        try:
            os.stat(path_str)
            file_exists = True
        except FileNotFoundError:
            file_exists = False
        if file_exists:
            existing = None
            existing_width = None
            try:
                existing, existing_width = _read_checkpoint(
                    path_str, algorithm=None, hidden_dim=width, map_location="cpu"
                )
            except _oom_types:
                raise
            except Exception as first_exc:
                if not _is_corruption_load_error(first_exc):
                    raise
                # 首次读取疑似格式损坏时，重新完整打开并校验字节数后复核，避免短读/瞬时 I/O 误覆盖高分原件
                raw_bytes = _read_file_bytes_exact(path_str)
                try:
                    existing, existing_width = _read_checkpoint(
                        io.BytesIO(raw_bytes), algorithm=None, hidden_dim=width, map_location="cpu"
                    )
                except _oom_types:
                    raise
                except Exception as second_exc:
                    if not _is_corruption_load_error(second_exc):
                        raise
                    reason = "schema_changed_or_unreadable"

            if existing is not None:
                weights_ok = _actor_weights_complete(existing.get("actor"), actor)
                same_schema = (
                    weights_ok
                    and existing_width == width
                    and all(existing.get(k) == v for k, v in expected_schema.items())
                    and existing.get("obs_layout") == str(obs_layout)
                )
                raw_prev = existing.get("canonical_score")
                valid_prev = (
                    not isinstance(raw_prev, (bool, np.bool_))
                    and isinstance(raw_prev, (int, float, np.number))
                    and math.isfinite(float(raw_prev))
                )
                if same_schema and valid_prev:
                    previous_score = float(raw_prev)
                    prev_sig = existing.get("eval_signature")
                    prev_sig = "" if prev_sig is None else str(prev_sig)
                    if cur_sig != prev_sig and not allow_eval_signature_replace:
                        return {
                            "updated": False,
                            "reason": "eval_signature_mismatch",
                            "path": path_str,
                            "new_score": score_val,
                            "previous_score": previous_score,
                        }
                    if score_val <= previous_score:
                        return {
                            "updated": False,
                            "reason": "existing_better_or_equal",
                            "path": path_str,
                            "new_score": score_val,
                            "previous_score": previous_score,
                        }
                    reason = "improved"
                else:
                    reason = "schema_changed_or_unreadable"

        clean_metrics = {}
        if isinstance(metrics, dict):
            for k, v in metrics.items():
                coerced = _to_builtin_primitive(v)
                if coerced is not None or v is None:
                    clean_metrics[str(k)] = coerced

        _atomic_save_checkpoint(
            path_str,
            actor,
            algo,
            obs_layout=str(obs_layout),
            canonical_score=score_val,
            eval_signature=cur_sig,
            canonical_metrics=clean_metrics,
        )
        return {
            "updated": True,
            "reason": reason,
            "path": path_str,
            "new_score": score_val,
            "previous_score": previous_score,
        }


def _read_checkpoint(path, *, algorithm=None, hidden_dim=None, map_location=None):
    """先校验 v2 输入契约；只有推理入口接受带持久版本标记的裸权重。"""
    checkpoint = torch.load(path, map_location=DEVICE if map_location is None else map_location, weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError("Expected a v2 checkpoint dictionary")
    wrapped = "actor" in checkpoint
    if not wrapped and algorithm is not None:
        raise ValueError(f"Expected a full {algorithm} checkpoint for training")
    weights = checkpoint["actor"] if wrapped else checkpoint
    if not isinstance(weights, dict):
        raise ValueError("Expected an actor state_dict in this checkpoint")
    if any("pos_embed" in key for key in weights):
        raise ValueError("Legacy pos_embed found; load v1 weights with the v1 evaluation worker")
    for key, expected in (("_obs_version", OBS_VERSION), ("_top_k", TOP_K)):
        value = weights.get(key)
        if (not isinstance(value, torch.Tensor) or value.is_meta or value.layout != torch.strided
                or value.shape != () or value.dtype != torch.int64 or value.item() != expected):
            raise ValueError(f"Expected v2 actor {key}={expected}; select matching weights and TOP_K")

    weight = weights.get("local_encoder.0.weight")
    if not isinstance(weight, torch.Tensor) or weight.ndim != 2:
        raise ValueError("Cannot determine actor hidden_dim from local_encoder weights")
    width = weight.shape[0]
    if hidden_dim is not None and width != hidden_dim:
        raise ValueError(f"Expected hidden_dim={hidden_dim}, got weights with width {width}")
    if wrapped:
        saved_algo = checkpoint.get("algorithm")
        if saved_algo not in ("MAPPO", "MASAC"):
            raise ValueError(f"Unknown checkpoint algorithm {saved_algo!r}; expected MAPPO or MASAC")
        if ((saved_algo == "MASAC" and "value" in checkpoint)
                or (saved_algo == "MAPPO" and "q1" in checkpoint)):
            raise ValueError(f"Checkpoint algorithm={saved_algo!r} conflicts with its training networks")
        schema = _actor_schema(width, algorithm or saved_algo)
        for key, expected in schema.items():
            actual = checkpoint.get(key)
            if type(actual) is not type(expected) or actual != expected:
                raise ValueError(f"Incompatible checkpoint {key}: expected {expected!r}, got {actual!r}")
    return checkpoint if wrapped else {"actor": weights}, width


def _restore_modules(checkpoint, **modules):
    # 一次检完所有网络，再写入参数。损坏或只含 Actor 的文件不能部分覆盖训练模型。
    for name, module in modules.items():
        weights = checkpoint.get(name)
        expected = module.state_dict()
        if not isinstance(weights, dict) or weights.keys() != expected.keys():
            raise ValueError(f"Missing or incompatible {name} state_dict; select a matching v2 checkpoint")
        for key, parameter in expected.items():
            if not isinstance(weights[key], torch.Tensor) or weights[key].shape != parameter.shape:
                raise ValueError(f"Incompatible {name}.{key} in checkpoint; expected shape {parameter.shape}")
    for name, module in modules.items():
        module.load_state_dict(checkpoint[name], strict=True)


class ActorOnlyPolicy:
    """只加载共享 Actor 的评估策略（支持任意 follower 数）。

    设计目标：
    - **评估/部署**场景不需要 centralized critic/value，也不需要 replay/rollout buffer。
    - 允许你在训练结束后，把 `num_followers` 设得很大（重新创建 topology/env），
      仍然能用同一个共享 actor 做闭环仿真。

    约束：
    - `STATE_DIM`/`ACTION_DIM`/`MAX_NEIGHBORS` 等网络输入输出维度必须与训练时一致。
    - `local_states` 约定 index=0 为 leader，follower 为 1..N。
    """

    def __init__(self, actor: DecentralizedActor, use_amp: bool = False):
        self.actor = actor.to(DEVICE)
        self.actor.eval()
        self.value_net = None  # 与 CTDESACAgent / CTDEMAPPOAgent 接口对齐，兼容评估工具
        self.use_amp = bool(use_amp and DEVICE.type == "cuda")
        self._autocast = None
        if self.use_amp:
            from torch.amp import autocast

            self._autocast = lambda: autocast("cuda")

    @classmethod
    def from_checkpoint(cls, path: str, hidden_dim: int | None = None, strict: bool = True, use_amp: bool = False):
        if not strict:
            raise ValueError("v2 checkpoint loading requires strict=True")
        checkpoint, width = _read_checkpoint(path, hidden_dim=hidden_dim)
        actor = DecentralizedActor(hidden_dim=width).to(DEVICE)
        _restore_modules(checkpoint, actor=actor)
        return cls(actor=actor, use_amp=use_amp)

    @torch.inference_mode()
    def select_action(self, local_states: torch.Tensor, deterministic: bool = True):
        """根据本地状态选择动作（面向 follower）。

        Args:
            local_states:
                - 单环境：shape=(num_agents, STATE_DIM)
                - 并行环境：shape=(E, num_agents, STATE_DIM)
            deterministic: 评估默认 True。

        Returns:
            follower 动作张量：
                - 单环境：shape=(num_followers, ACTION_DIM)
                - 并行环境：shape=(E, num_followers, ACTION_DIM)
        """
        is_batched = local_states.dim() == 3

        if is_batched:
            E = int(local_states.shape[0])
            num_followers = int(local_states.shape[1]) - 1
            follower_states = local_states[:, 1:, :]
            flat_states = follower_states.reshape(-1, STATE_DIM)

            if self.use_amp:
                with self._autocast():
                    action, _ = self.actor(flat_states, deterministic=bool(deterministic))
            else:
                action, _ = self.actor(flat_states, deterministic=bool(deterministic))

            action = action.view(E, num_followers, ACTION_DIM)
            return action.float()

        num_followers = int(local_states.shape[0]) - 1
        follower_states = local_states[1:, :]
        if self.use_amp:
            with self._autocast():
                action, _ = self.actor(follower_states, deterministic=bool(deterministic))
        else:
            action, _ = self.actor(follower_states, deterministic=bool(deterministic))
        return action.float()


class CTDESACAgent:
    """CTDE-SAC 智能体（集中训练、分散执行）。

    - Actor：对每个 follower 使用本地 flat state 生成动作。
    - Critic：集中式 Q 网络，输入为 global state + joint action。

    Args:
        topology: `CommunicationTopology` 实例（提供 follower 数量等信息）。
        auto_entropy: 是否启用 entropy 系数自动调节（SAC 的 temperature）。
        use_amp: 是否启用 AMP 混合精度（仅 CUDA 环境有效）。

    Attributes:
        actor: `DecentralizedActor`，用于执行/策略更新。
        q1/q2: `CentralizedCritic`，双 Q 网络。
        buffer: `CTDEReplayBuffer`。
        value_net: 为了评估工具统一保留（SAC 不使用，恒为 None）。
    """

    def __init__(self, topology, auto_entropy: bool = True, use_amp: bool = True, *, train_follower_scales=None):
        if train_follower_scales is not None and tuple(train_follower_scales) != (topology.num_followers,):
            raise ValueError("MASAC requires one fixed follower scale matching its topology")
        self.topology = topology
        self.num_followers = topology.num_followers
        self.num_agents = topology.num_agents
        self.auto_entropy = bool(auto_entropy)

        self.use_amp = bool(use_amp and DEVICE.type == "cuda")
        self.scaler = None
        self._autocast = None
        if self.use_amp:
            from torch.amp import GradScaler
            from torch.amp import autocast

            self.scaler = GradScaler("cuda")
            self._autocast = lambda: autocast("cuda")
            print("🚀 AMP (混合精度训练) 已启用 - CTDE 架构")

        # 标准化接口：SAC 没有 value_net，但为了评估工具一致，保留该属性。
        self.value_net = None

        self.actor = DecentralizedActor(hidden_dim=HIDDEN_DIM).to(DEVICE)

        critic_args = dict(global_state_dim=global_state_dim_for(self.num_followers),
                           num_followers=self.num_followers, hidden_dim=HIDDEN_DIM)
        self.q1 = CentralizedCritic(**critic_args).to(DEVICE)
        self.q2 = CentralizedCritic(**critic_args).to(DEVICE)
        self.q1_target = CentralizedCritic(**critic_args).to(DEVICE)
        self.q2_target = CentralizedCritic(**critic_args).to(DEVICE)

        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        for p in self.q1_target.parameters():
            p.requires_grad = False
        for p in self.q2_target.parameters():
            p.requires_grad = False

        self.target_entropy = -float(ACTION_DIM * self.num_followers) * float(TARGET_ENTROPY_RATIO)
        self.log_alpha = torch.tensor(np.log(INIT_ALPHA), requires_grad=True, device=DEVICE)
        self.alpha = float(self.log_alpha.exp().item())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=LEARNING_RATE)
        self.q1_optimizer = optim.Adam(self.q1.parameters(), lr=LEARNING_RATE)
        self.q2_optimizer = optim.Adam(self.q2.parameters(), lr=LEARNING_RATE)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=ALPHA_LR)

        self.buffer = CTDEReplayBuffer(num_agents=self.num_agents, global_state_dim=self.q1.global_state_dim)

        self.last_losses = {"q1": 0, "q2": 0, "actor": 0, "alpha": INIT_ALPHA}
        self.update_count = 0

        print("📊 CTDE SAC Agent initialized:")
        print(f"   Actor input: Local state ({STATE_DIM})")
        print(f"   Critic input: Global state ({self.q1.global_state_dim}) + Joint action ({self.num_followers * ACTION_DIM})")

    @torch.inference_mode()
    def select_action(self, local_states: torch.Tensor, deterministic: bool = False):
        """根据本地状态选择动作（面向 follower）。

        Args:
            local_states: 本地状态。
                - 单环境：shape=(num_agents, STATE_DIM)
                - 并行环境：shape=(E, num_agents, STATE_DIM)
                其中 index=0 为 leader，follower 为 1..N。
            deterministic: 是否使用确定性动作（评估/可视化常用 True）。

        Returns:
            follower 动作张量：
            - 单环境：shape=(num_followers, ACTION_DIM)
            - 并行环境：shape=(E, num_followers, ACTION_DIM)
        """
        is_batched = local_states.dim() == 3

        if is_batched:
            batch_size = int(local_states.shape[0])
            follower_states = local_states[:, 1:, :]
            flat_states = follower_states.reshape(-1, STATE_DIM)

            if self.use_amp:
                with self._autocast():
                    action, _ = self.actor(flat_states, deterministic=deterministic)
            else:
                action, _ = self.actor(flat_states, deterministic=deterministic)

            action = action.view(batch_size, self.num_followers, ACTION_DIM)
        else:
            follower_states = local_states[1:, :]
            if self.use_amp:
                with self._autocast():
                    action, _ = self.actor(follower_states, deterministic=deterministic)
            else:
                action, _ = self.actor(follower_states, deterministic=deterministic)

        return action.float()

    def store_transitions_batch(
        self,
        local_states: torch.Tensor,
        global_states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_local_states: torch.Tensor,
        next_global_states: torch.Tensor,
        dones: torch.Tensor,
    ):
        """向 replay buffer 追加一批 transition。

        Args:
            local_states: shape=(E, num_agents, STATE_DIM)。
            global_states: shape=(E, GLOBAL_STATE_DIM)。
            actions: shape=(E, num_followers, ACTION_DIM)。
            rewards: shape=(E,)。
            next_local_states: shape=(E, num_agents, STATE_DIM)。
            next_global_states: shape=(E, GLOBAL_STATE_DIM)。
            dones: shape=(E,)；True 表示该并行环境在该步终止（含时间截断）。
        """
        self.buffer.push_batch(local_states, global_states, actions, rewards, next_local_states, next_global_states, dones)

    def update(self, batch_size: int = BATCH_SIZE, gradient_steps: int = GRADIENT_STEPS):
        """执行一次或多次 SAC 更新。

        Args:
            batch_size: 每次从 replay buffer 采样的 batch 大小。
            gradient_steps: 连续更新的次数（每次都会重新采样）。

        Returns:
            最近一次更新的损失字典（用于日志/仪表盘）。若 buffer 数据不足返回空字典。
        """
        if not self.buffer.is_ready(batch_size):
            return {}

        total_q1_loss = 0.0
        total_q2_loss = 0.0
        total_actor_loss = 0.0

        total_q1_mean = 0.0
        total_q2_mean = 0.0
        total_target_q_mean = 0.0
        total_logp_joint = 0.0
        total_entropy_joint = 0.0
        total_alpha_loss = 0.0
        policy_updates = 0

        for _ in range(int(gradient_steps)):
            self.update_count += 1

            (local_states, global_states, actions, rewards, next_local_states, next_global_states, dones) = self.buffer.sample(batch_size)

            follower_states = local_states[:, 1:, :].reshape(-1, STATE_DIM)
            follower_next_states = next_local_states[:, 1:, :].reshape(-1, STATE_DIM)
            joint_actions = actions.view(batch_size, -1)

            with torch.no_grad():
                if self.use_amp:
                    with self._autocast():
                        next_actions, next_log_probs = self.actor(follower_next_states)
                else:
                    next_actions, next_log_probs = self.actor(follower_next_states)

                next_joint_actions = next_actions.view(batch_size, -1)
                next_log_probs_joint = next_log_probs.view(batch_size, self.num_followers, 1).sum(dim=1)

                q1_next = self.q1_target(next_global_states, next_joint_actions)
                q2_next = self.q2_target(next_global_states, next_joint_actions)
                q_next = torch.min(q1_next, q2_next)

                target_q = rewards.unsqueeze(1) + GAMMA * (1 - dones.unsqueeze(1)) * (q_next - self.alpha * next_log_probs_joint)
                target_q = target_q.float()

            self.q1_optimizer.zero_grad(set_to_none=True)
            self.q2_optimizer.zero_grad(set_to_none=True)

            if self.use_amp:
                with self._autocast():
                    q1_curr = self.q1(global_states, joint_actions)
                    q2_curr = self.q2(global_states, joint_actions)
                    q1_loss = F.mse_loss(q1_curr.float(), target_q)
                    q2_loss = F.mse_loss(q2_curr.float(), target_q)
                    critic_loss = q1_loss + q2_loss

                self.scaler.scale(critic_loss).backward()

                self.scaler.unscale_(self.q1_optimizer)
                self.scaler.unscale_(self.q2_optimizer)
                torch.nn.utils.clip_grad_norm_(self.q1.parameters(), 1.0)
                torch.nn.utils.clip_grad_norm_(self.q2.parameters(), 1.0)

                self.scaler.step(self.q1_optimizer)
                self.scaler.step(self.q2_optimizer)
            else:
                q1_curr = self.q1(global_states, joint_actions)
                q2_curr = self.q2(global_states, joint_actions)
                q1_loss = F.mse_loss(q1_curr, target_q)
                q2_loss = F.mse_loss(q2_curr, target_q)
                critic_loss = q1_loss + q2_loss

                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.q1.parameters(), 1.0)
                torch.nn.utils.clip_grad_norm_(self.q2.parameters(), 1.0)
                self.q1_optimizer.step()
                self.q2_optimizer.step()

            do_policy_update = self.update_count % max(1, int(POLICY_DELAY)) == 0
            actor_loss = torch.tensor(0.0, device=DEVICE)

            if do_policy_update:
                self.actor_optimizer.zero_grad(set_to_none=True)

                if self.use_amp:
                    with self._autocast():
                        new_actions, log_probs = self.actor(follower_states)
                        new_joint_actions = new_actions.view(batch_size, -1)

                        q1_new = self.q1(global_states, new_joint_actions)
                        q2_new = self.q2(global_states, new_joint_actions)
                        q_new = torch.min(q1_new, q2_new)

                        log_probs_joint = log_probs.view(batch_size, self.num_followers, 1).sum(dim=1)
                        actor_loss = (self.alpha * log_probs_joint - q_new).mean()

                    self.scaler.scale(actor_loss).backward()
                    self.scaler.unscale_(self.actor_optimizer)
                    torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
                    self.scaler.step(self.actor_optimizer)
                else:
                    new_actions, log_probs = self.actor(follower_states)
                    new_joint_actions = new_actions.view(batch_size, -1)

                    q1_new = self.q1(global_states, new_joint_actions)
                    q2_new = self.q2(global_states, new_joint_actions)
                    q_new = torch.min(q1_new, q2_new)

                    log_probs_joint = log_probs.view(batch_size, self.num_followers, 1).sum(dim=1)
                    actor_loss = (self.alpha * log_probs_joint - q_new).mean()

                    actor_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
                    self.actor_optimizer.step()

                if self.auto_entropy:
                    self.alpha_optimizer.zero_grad(set_to_none=True)

                    log_probs_joint_detached = log_probs.view(batch_size, self.num_followers, 1).sum(dim=1).detach()
                    mean_log_prob = log_probs_joint_detached.mean()

                    alpha_loss = -(self.log_alpha * (mean_log_prob + self.target_entropy))
                    alpha_loss.backward()
                    self.alpha_optimizer.step()
                    self.alpha = float(self.log_alpha.exp().item())

            if self.use_amp:
                self.scaler.update()

            if self.update_count % max(1, int(TARGET_UPDATE_INTERVAL)) == 0:
                tau_eff = 1.0 - (1.0 - TAU) ** max(1, int(TARGET_UPDATE_INTERVAL))
                self._soft_update(self.q1, self.q1_target, tau=tau_eff)
                self._soft_update(self.q2, self.q2_target, tau=tau_eff)

            total_q1_loss += float(q1_loss.item())
            total_q2_loss += float(q2_loss.item())
            total_actor_loss += float(actor_loss.item())

            total_q1_mean += float(q1_curr.detach().mean().item())
            total_q2_mean += float(q2_curr.detach().mean().item())
            total_target_q_mean += float(target_q.detach().mean().item())

            if do_policy_update:
                policy_updates += 1
                total_logp_joint += float(log_probs_joint.detach().mean().item())
                total_entropy_joint += float((-log_probs_joint).detach().mean().item())
                if self.auto_entropy:
                    total_alpha_loss += float(alpha_loss.detach().item())

        pol_denom = max(1, int(policy_updates))
        self.last_losses = {
            "q1": total_q1_loss / float(gradient_steps),
            "q2": total_q2_loss / float(gradient_steps),
            "actor": total_actor_loss / float(gradient_steps),
            "alpha": float(self.alpha),
            "q1_mean": total_q1_mean / float(gradient_steps),
            "q2_mean": total_q2_mean / float(gradient_steps),
            "target_q_mean": total_target_q_mean / float(gradient_steps),
            "logp_joint": total_logp_joint / float(pol_denom),
            "entropy_joint": total_entropy_joint / float(pol_denom),
            "alpha_loss": total_alpha_loss / float(pol_denom) if self.auto_entropy else float("nan"),
            "policy_updates": int(policy_updates),
        }

        return self.last_losses

    @torch.no_grad()
    def _soft_update(self, source: torch.nn.Module, target: torch.nn.Module, tau: float = TAU):
        for param, target_param in zip(source.parameters(), target.parameters()):
            target_param.data.lerp_(param.data, tau)

    def save(self, path: str):
        """保存模型参数到文件。

        Args:
            path: 保存路径。
        """
        _save_checkpoint(
            path, self.actor, "MASAC", q1=self.q1.state_dict(), q2=self.q2.state_dict(),
            q1_target=self.q1_target.state_dict(), q2_target=self.q2_target.state_dict(),
            log_alpha=self.log_alpha, update_count=self.update_count,
        )
        print(f"✅ CTDE Model saved to {path}")

    def save_actor(self, path: str):
        _save_checkpoint(path, self.actor, "MASAC")

    def load(self, path: str):
        """从文件加载模型参数。

        Args:
            path: checkpoint 路径。
        """
        checkpoint, _ = _read_checkpoint(path, algorithm="MASAC", hidden_dim=self.actor.local_encoder[0].out_features)
        _restore_modules(checkpoint, actor=self.actor, q1=self.q1, q2=self.q2,
                         q1_target=self.q1_target, q2_target=self.q2_target)

        if "log_alpha" in checkpoint:
            ckpt_log_alpha = checkpoint["log_alpha"]
            if isinstance(ckpt_log_alpha, torch.Tensor):
                ckpt_log_alpha = ckpt_log_alpha.to(device=DEVICE, dtype=self.log_alpha.dtype)
                self.log_alpha.data.copy_(ckpt_log_alpha)
            else:
                self.log_alpha.data.fill_(float(ckpt_log_alpha))
            self.alpha = float(self.log_alpha.exp().item())

        if "update_count" in checkpoint:
            self.update_count = int(checkpoint["update_count"])

        print(f"✅ CTDE Model loaded from {path}")


class _PPOBuffer:
    """On-policy rollout buffer（按时间步存储，支持并行环境与 per-agent logp）。"""

    def __init__(
        self,
        rollout_steps: int = PPO_ROLLOUT_STEPS,
        *,
        num_followers: int,
        state_dim: int,
        global_state_dim: int,
        action_dim: int,
    ):
        self.rollout_steps = int(rollout_steps)
        self.num_followers = int(num_followers)
        self.state_dim = int(state_dim)
        self.global_state_dim = int(global_state_dim)
        self.action_dim = int(action_dim)
        if min(self.num_followers, self.state_dim, self.global_state_dim, self.action_dim) < 1:
            raise ValueError(
                "num_followers, state_dim, global_state_dim, action_dim must all be >= 1, got "
                f"({self.num_followers}, {self.state_dim}, {self.global_state_dim}, {self.action_dim})"
            )
        self.clear()

    def clear(self):
        self.follower_states = []
        self.global_states = []
        self.actions = []
        self.pre_squash_v = []
        self.has_pre_squash_v = []
        self.logp = []
        self.values = []
        self.rewards = []
        self.dones = []
        self.truncated = []
        self.truncated_next_values = []
        self.has_truncated_next_values = []

    def add(
        self,
        follower_states,
        global_states,
        actions,
        logp,
        values,
        rewards,
        dones,
        truncated=None,
        truncated_next_values=None,
        has_truncated_next_values=None,
        pre_squash_v=None,
    ):
        # 先完成全部形状校验与独立快照准备，全部成功后才原子追加各列，防止异常留下半写入脏状态
        expected_F = self.num_followers
        expected_S = self.state_dim
        expected_G = self.global_state_dim
        expected_A = self.action_dim

        if follower_states.dim() != 3:
            raise ValueError(
                f"follower_states: expected shape (>=1, {expected_F}, {expected_S}), got {tuple(follower_states.shape)}"
            )
        E, F_dim, S_dim = (int(x) for x in follower_states.shape)

        if E < 1:
            raise ValueError(
                f"follower_states: expected shape (>=1, {expected_F}, {expected_S}), got {tuple(follower_states.shape)}"
            )
        if len(self.rewards) > 0:
            prev_E = int(self.rewards[0].shape[0])
            if E != prev_E:
                raise ValueError(
                    f"follower_states: expected shape ({prev_E}, {expected_F}, {expected_S}), got {tuple(follower_states.shape)}"
                )
        if (F_dim, S_dim) != (expected_F, expected_S):
            raise ValueError(
                f"follower_states: expected shape ({E}, {expected_F}, {expected_S}), got {tuple(follower_states.shape)}"
            )
        if global_states.dim() != 2 or tuple(global_states.shape) != (E, expected_G):
            raise ValueError(
                f"global_states: expected shape ({E}, {expected_G}), got {tuple(global_states.shape)}"
            )
        if actions.dim() != 3 or tuple(actions.shape) != (E, expected_F, expected_A):
            raise ValueError(
                f"actions: expected shape ({E}, {expected_F}, {expected_A}), got {tuple(actions.shape)}"
            )
        if tuple(logp.shape) != (E, expected_F, 1):
            raise ValueError(f"logp: expected shape ({E}, {expected_F}, 1), got {tuple(logp.shape)}")
        if tuple(values.shape) != (E, 1):
            raise ValueError(f"values: expected shape ({E}, 1), got {tuple(values.shape)}")
        if tuple(rewards.shape) != (E,):
            raise ValueError(f"rewards: expected shape ({E},), got {tuple(rewards.shape)}")
        if tuple(dones.shape) != (E,):
            raise ValueError(f"dones: expected shape ({E},), got {tuple(dones.shape)}")

        if truncated is not None and tuple(truncated.shape) != (E,):
            raise ValueError(f"truncated: expected shape ({E},), got {tuple(truncated.shape)}")
        if truncated_next_values is not None and tuple(truncated_next_values.shape) != (E,):
            raise ValueError(
                f"truncated_next_values: expected shape ({E},), got {tuple(truncated_next_values.shape)}"
            )
        if has_truncated_next_values is not None and tuple(has_truncated_next_values.shape) != (E,):
            raise ValueError(
                f"has_truncated_next_values: expected shape ({E},), got {tuple(has_truncated_next_values.shape)}"
            )
        if pre_squash_v is not None and tuple(pre_squash_v.shape) not in ((E, expected_F, 1), (E, expected_F)):
            raise ValueError(
                f"pre_squash_v: expected shape ({E}, {expected_F}, 1) or ({E}, {expected_F}), "
                f"got {tuple(pre_squash_v.shape)}"
            )

        fs_c = follower_states.detach().float().clone()
        gs_c = global_states.detach().float().clone()
        act_c = actions.detach().float().clone()
        logp_c = logp.detach().float().clone()
        val_c = values.detach().float().clone()
        rew_c = rewards.detach().float().clone()
        done_c = dones.detach().bool().clone()

        trunc_c = torch.zeros_like(done_c, dtype=torch.bool) if truncated is None else truncated.detach().bool().clone()
        trunc_nv_c = (
            torch.zeros_like(rew_c, dtype=torch.float32)
            if truncated_next_values is None
            else truncated_next_values.detach().float().clone()
        )
        has_trunc_nv_c = (
            torch.zeros_like(done_c, dtype=torch.bool)
            if has_truncated_next_values is None
            else has_truncated_next_values.detach().bool().clone()
        )

        if pre_squash_v is None:
            pre_v_c = torch.zeros_like(logp_c, dtype=torch.float32)
            has_pre_v_c = torch.zeros_like(logp_c, dtype=torch.bool)
        else:
            pre_v_c = pre_squash_v.detach().float().reshape_as(logp_c).clone()
            has_pre_v_c = torch.ones_like(logp_c, dtype=torch.bool)

        # 所有准备就绪后一次性追加
        self.follower_states.append(fs_c)
        self.global_states.append(gs_c)
        self.actions.append(act_c)
        self.pre_squash_v.append(pre_v_c)
        self.has_pre_squash_v.append(has_pre_v_c)
        self.logp.append(logp_c)
        self.values.append(val_c)
        self.rewards.append(rew_c)
        self.dones.append(done_c)
        self.truncated.append(trunc_c)
        self.truncated_next_values.append(trunc_nv_c)
        self.has_truncated_next_values.append(has_trunc_nv_c)

    def __len__(self):
        return len(self.rewards)

    def is_full(self):
        return len(self) >= self.rollout_steps


class CTDEMAPPOAgent:
    """CTDE-MAPPO 智能体。

    - Policy：对每个 follower 使用本地状态（factorized policy，参数共享）。
    - Value：集中式 V 网络，输入 global state。

    Args:
        topology: `CommunicationTopology` 实例。
        use_amp: 是否启用 AMP（当前实现默认关闭）。

    Notes:
        MAPPO 是 on-policy 算法，需要先通过 `store_rollout_step()` 收集 rollout，
        再调用 `update()` 进行多轮 PPO 优化。
    """

    def __init__(self, topology, use_amp: bool = False, *,
                 actor_microbatch_max_rows: int | None = ACTOR_MICROBATCH_MAX_FOLLOWER_ROWS):
        self.actor_microbatch_max_rows = actor_microbatch_max_rows
        self._validate_microbatch_cap(topology.num_followers)
        self._session = None
        self._session_lock = Lock()
        self.topology = topology
        self.num_followers = topology.num_followers
        self.num_agents = topology.num_agents
        self.use_amp = False

        self.actor = DecentralizedActor(hidden_dim=HIDDEN_DIM).to(DEVICE)
        self.value_net = CentralizedValue(global_state_dim=global_state_dim_for(self.num_followers), hidden_dim=HIDDEN_DIM).to(DEVICE)
        self._disable_actor_dropout()

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=PPO_LR)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=PPO_LR)

        self.expected_global_state_dim = int(self.value_net.ctx_dim + self.num_agents * self.value_net.agent_feat_dim)
        self._fixed_buffer = _PPOBuffer(
            rollout_steps=PPO_ROLLOUT_STEPS,
            num_followers=self.num_followers,
            state_dim=STATE_DIM,
            global_state_dim=self.expected_global_state_dim,
            action_dim=ACTION_DIM,
        )
        self.last_losses = {}
        # 按返回动作 Tensor 的对象身份 (id + weakref) 精确绑定随机采样时的原始 pre-squash 速度，
        # 避免按值匹配串用不同采样，并由 weakref.finalize 自动清理防止内存泄漏。
        self._sampled_pre_squash_by_id: dict[int, tuple[weakref.ref, torch.Tensor]] = {}

        print("📊 CTDE MAPPO Agent initialized:")
        print(f"   Actor input: Local state ({STATE_DIM})")
        print(f"   Value input: Global state ({self.expected_global_state_dim})")
        print(f"   Rollout steps: {PPO_ROLLOUT_STEPS} | Epochs: {PPO_EPOCHS} | Minibatch: {PPO_MINIBATCH_SIZE}")

    @property
    def buffer(self):
        return self._fixed_buffer if self._session is None else self._session.buffer

    def start_episode(self, env):
        self._validate_microbatch_cap(env.num_followers)
        return MultiScaleEpisodeSession(self, env)

    def _validate_microbatch_cap(self, num_followers):
        if self.actor_microbatch_max_rows is not None and self.actor_microbatch_max_rows < num_followers:
            raise ValueError(f"One environment record contains num_followers={num_followers} rows; "
                             f"actor_microbatch_max_rows={self.actor_microbatch_max_rows} must fit one record")

    def _disable_actor_dropout(self) -> None:
        """在 MAPPO 采样与更新阶段一致关闭 Actor 内部 Dropout。

        原因：PPO 重要度采样比率 ratio = exp(new_logp - old_logp) 与 k3 KL 近似要求
        采样与重评估对应同一个确定性策略分布；若训练期启用随机 Dropout，即使参数未更新
        也会产生随机 mask 偏差，破坏 ratio=1 / KL=0 契约。关闭 Dropout 同时保留全部参数梯度。
        """
        for module in self.actor.modules():
            if isinstance(module, torch.nn.Dropout):
                module.eval()

    @torch.inference_mode()
    def select_action(self, local_states: torch.Tensor, deterministic: bool = False):
        self._disable_actor_dropout()
        follower_states = local_states[..., 1:, :]
        action, _ = self.actor(follower_states.reshape(-1, STATE_DIM), deterministic=deterministic)
        return action.reshape(*follower_states.shape[:-1], ACTION_DIM).float()

    @torch.inference_mode()
    def act(
        self,
        local_states: torch.Tensor,
        global_states: torch.Tensor | None = None,
        deterministic: bool = False,
        return_pre_squash: bool = False,
    ):
        """采样动作并（可选）返回 value 估计及原始 pre-squash 速度样本。

        Args:
            local_states: shape=(E, num_agents, STATE_DIM)。
            global_states: shape=(E, GLOBAL_STATE_DIM)；若为 None 则不计算 value。
            deterministic: 是否使用确定性动作。
            return_pre_squash: 若为 True，额外返回第 4 个元素 `pre_squash_v`（shape=(E, F, 1)），
                便于调用方显式传给 `store_rollout_step(..., pre_squash_v=...)`。

        Returns:
            - return_pre_squash=False: (actions, logp, values)
            - return_pre_squash=True:  (actions, logp, values, pre_squash_v)
        """
        return self._act_for_buffer(local_states, global_states, self._fixed_buffer,
                                    deterministic, return_pre_squash, bind_sample=True)

    @torch.inference_mode()
    def _act_for_buffer(self, local_states, global_states, buffer, deterministic,
                        return_pre_squash, *, bind_sample):
        num_followers = buffer.num_followers
        num_agents = num_followers + 1
        assert local_states.dim() == 3, "MAPPO 训练需要 batched local_states"
        if tuple(local_states.shape[1:]) != (num_agents, STATE_DIM) or int(local_states.shape[0]) < 1:
            raise ValueError(
                f"local_states: expected shape (>=1, {num_agents}, {STATE_DIM}), got {tuple(local_states.shape)}"
            )
        E = int(local_states.shape[0])
        if global_states is not None:
            if global_states.dim() != 2 or tuple(global_states.shape) != (E, buffer.global_state_dim):
                raise ValueError(
                    f"global_states: expected shape {(E, buffer.global_state_dim)}, "
                    f"got {tuple(global_states.shape)}"
                )
        self._disable_actor_dropout()

        follower_states = local_states[:, 1:, :].reshape(-1, STATE_DIM)
        act_flat, logp_flat, v_pre_flat = self.actor(
            follower_states, deterministic=deterministic, return_pre_squash=True
        )
        actions = act_flat.view(E, num_followers, ACTION_DIM).float()
        v_pre = v_pre_flat.view(E, num_followers, 1).float().detach().clone()

        if bind_sample and not deterministic:
            # 绑定表独占一份独立快照，避免调用方复用 return_pre_squash 返回张量时改写绑定值
            act_id = id(actions)
            self._sampled_pre_squash_by_id[act_id] = (weakref.ref(actions), v_pre.clone())
            weakref.finalize(actions, self._sampled_pre_squash_by_id.pop, act_id, None)

        if logp_flat is None:
            logp = torch.zeros((E, num_followers, 1), device=DEVICE)
        else:
            logp = logp_flat.view(E, num_followers, 1).float()

        if global_states is None:
            values = torch.full((E, 1), float("nan"), device=DEVICE)
        else:
            values = self.value_net(global_states).float()

        if return_pre_squash:
            return actions, logp, values, v_pre.clone()
        return actions, logp, values

    def store_rollout_step(
        self,
        local_states,
        global_states,
        actions,
        logp=None,
        values=None,
        rewards=None,
        dones=None,
        truncated=None,
        next_global_states=None,
        truncated_next_values=None,
        pre_squash_v=None,
        *,
        logp_joint=None,
    ):
        if self._session is not None:
            raise RuntimeError("Active episode owns collection; call session.step_collect()")
        # 兼容旧关键字参数 `logp_joint=` 别名，并统一校验必需参数与 per-agent 形状
        if logp is not None and logp_joint is not None:
            raise TypeError("Specify only one of 'logp' or legacy alias 'logp_joint', not both.")
        if logp is None:
            logp = logp_joint
        if logp is None or values is None or rewards is None or dones is None:
            raise TypeError("store_rollout_step requires logp (or logp_joint), values, rewards, and dones.")

        if local_states.dim() != 3 or tuple(local_states.shape[1:]) != (self.num_agents, STATE_DIM):
            raise ValueError(
                f"local_states: expected shape (>=1, {self.num_agents}, {STATE_DIM}), got {tuple(local_states.shape)}"
            )
        E = int(local_states.shape[0])
        if E < 1:
            raise ValueError(
                f"local_states: expected shape (>=1, {self.num_agents}, {STATE_DIM}), got {tuple(local_states.shape)}"
            )
        if len(self.buffer) > 0:
            expected_E = int(self.buffer.rewards[0].shape[0])
            if E != expected_E:
                raise ValueError(
                    f"local_states: expected shape {(expected_E, self.num_agents, STATE_DIM)}, "
                    f"got {tuple(local_states.shape)}"
                )

        expected_gs_dim = self.expected_global_state_dim
        if tuple(global_states.shape) != (E, expected_gs_dim):
            raise ValueError(
                f"global_states: expected shape {(E, expected_gs_dim)}, got {tuple(global_states.shape)}"
            )
        expected_act_shape = (E, self.num_followers, ACTION_DIM)
        if tuple(actions.shape) != expected_act_shape:
            raise ValueError(f"actions: expected shape {expected_act_shape}, got {tuple(actions.shape)}")
        expected_logp_shape = (E, self.num_followers, 1)
        if tuple(logp.shape) != expected_logp_shape:
            raise ValueError(f"logp: expected shape {expected_logp_shape}, got {tuple(logp.shape)}")
        if tuple(values.shape) != (E, 1):
            raise ValueError(f"values: expected shape {(E, 1)}, got {tuple(values.shape)}")
        if tuple(rewards.shape) != (E,):
            raise ValueError(f"rewards: expected shape {(E,)}, got {tuple(rewards.shape)}")
        if tuple(dones.shape) != (E,):
            raise ValueError(f"dones: expected shape {(E,)}, got {tuple(dones.shape)}")
        if truncated is not None and tuple(truncated.shape) != (E,):
            raise ValueError(f"truncated: expected shape {(E,)}, got {tuple(truncated.shape)}")
        if next_global_states is not None and tuple(next_global_states.shape) != (E, expected_gs_dim):
            raise ValueError(
                f"next_global_states: expected shape {(E, expected_gs_dim)}, got {tuple(next_global_states.shape)}"
            )
        if truncated_next_values is not None and tuple(truncated_next_values.shape) not in ((E,), (E, 1)):
            raise ValueError(
                f"truncated_next_values: expected shape {(E,)} or {(E, 1)}, got {tuple(truncated_next_values.shape)}"
            )
        if pre_squash_v is not None and tuple(pre_squash_v.shape) not in (expected_logp_shape, (E, self.num_followers)):
            raise ValueError(
                f"pre_squash_v: expected shape {expected_logp_shape} or {(E, self.num_followers)}, "
                f"got {tuple(pre_squash_v.shape)}"
            )

        follower_states = local_states[:, 1:, :].detach()
        actions_f = actions.detach().float()
        dones_b = dones.detach().bool()
        truncated_b = truncated.detach().bool() if truncated is not None else torch.zeros_like(dones_b, dtype=torch.bool)
        rewards_f = rewards.detach().float()

        # 先查询但不消费对象身份绑定；待 buffer.add 校验和快照全部成功后再消费
        bound_entry = self._sampled_pre_squash_by_id.get(id(actions))
        has_bound_entry = bound_entry is not None and bound_entry[0]() is actions

        if pre_squash_v is not None:
            pre_squash_v_f = pre_squash_v.detach().float().reshape(expected_logp_shape).clone()
        elif has_bound_entry:
            pre_squash_v_f = bound_entry[1].detach().float().clone()
        else:
            pre_squash_v_f = None

        trunc_nv = torch.zeros_like(rewards_f)
        has_trunc_nv = torch.zeros_like(dones_b, dtype=torch.bool)

        if truncated_b.any():
            if truncated_next_values is not None:
                trunc_nv = truncated_next_values.detach().float().reshape(E).clone()
                has_trunc_nv = truncated_b.clone()
            elif next_global_states is not None:
                with torch.no_grad():
                    trunc_nv = self.value_net(next_global_states.detach().float()).squeeze(-1).float().clone()
                has_trunc_nv = truncated_b.clone()

        self.buffer.add(
            follower_states.float(),
            global_states.detach().float(),
            actions_f,
            logp.detach().float(),
            values.detach().float(),
            rewards_f,
            dones_b,
            truncated_b,
            trunc_nv,
            has_trunc_nv,
            pre_squash_v=pre_squash_v_f,
        )

        # 仅在原子入库成功后才消费该动作的对象身份绑定
        if has_bound_entry:
            self._sampled_pre_squash_by_id.pop(id(actions), None)

    def _compute_gae(
        self,
        rewards,
        dones,
        values,
        last_value,
        truncated=None,
        truncated_next_values=None,
        has_truncated_next_values=None,
    ):
        """计算 GAE 与回报，正确区分真实终止 (dones) 与时间截断 (truncated)。

        - 真实终止 (`dones[t]=True`，即使同时 `truncated[t]=True`)：不 bootstrap (`next_value * 0`)，且中断 GAE 递推。
        - 时间截断 (`truncated[t]=True, dones[t]=False`)：中断跨 episode 的 GAE 递推，
          并使用截断前真实的 `V(s_{t+1})`（`truncated_next_values[t]` 或末步 `last_value`）做 bootstrap，
          避免跨 episode 错用下一 episode 初始状态价值或 `values[t]`。
        """
        T, E = rewards.shape
        if truncated is None:
            truncated = torch.zeros_like(dones, dtype=torch.bool)
        if truncated_next_values is None:
            truncated_next_values = torch.zeros_like(rewards, dtype=torch.float32)
        if has_truncated_next_values is None:
            has_truncated_next_values = torch.zeros_like(dones, dtype=torch.bool)

        advantages = torch.zeros(T, E, device=DEVICE)
        gae = torch.zeros(E, device=DEVICE)

        for t in reversed(range(T)):
            nonterminal = (~dones[t]).float()
            episode_continue = (~(dones[t] | truncated[t])).float()
            need_trunc_bootstrap = truncated[t] & (~dones[t])

            if t == T - 1:
                use_stored = need_trunc_bootstrap & has_truncated_next_values[t]
                next_value = torch.where(use_stored, truncated_next_values[t], last_value)
            else:
                if need_trunc_bootstrap.any() and not bool(has_truncated_next_values[t][need_trunc_bootstrap].all().item()):
                    raise ValueError(
                        "Truncated step before rollout end requires next_global_states or truncated_next_values "
                        "to compute exact bootstrap V(s_{t+1})."
                    )
                next_value = torch.where(need_trunc_bootstrap, truncated_next_values[t], values[t + 1])

            delta = rewards[t] + GAMMA * next_value * nonterminal - values[t]
            gae = delta + GAMMA * PPO_GAE_LAMBDA * episode_continue * gae
            advantages[t] = gae

        returns = advantages + values
        return advantages, returns

    def update(self, next_global_states=None, next_dones=None):
        if self._session is not None:
            raise RuntimeError("Active episode owns bootstrap; call session.flush_update()")
        return self._update_buffer(self._fixed_buffer, next_global_states, next_dones)

    def _update_buffer(self, buffer, next_global_states, next_dones, *, episode_end=False):
        if len(buffer) == 0:
            return {}
        self._validate_microbatch_cap(buffer.num_followers)

        self._disable_actor_dropout()

        follower_states = torch.stack(buffer.follower_states, dim=0)  # (T,E,F,S)
        global_states = torch.stack(buffer.global_states, dim=0)  # (T,E,G)
        actions = torch.stack(buffer.actions, dim=0)  # (T,E,F,A)
        old_logp = torch.stack(buffer.logp, dim=0)  # (T,E,F,1)
        values = torch.stack(buffer.values, dim=0)  # (T,E,1)
        rewards = torch.stack(buffer.rewards, dim=0)  # (T,E)
        dones = torch.stack(buffer.dones, dim=0)  # (T,E)
        truncated = torch.stack(buffer.truncated, dim=0)  # (T,E)
        truncated_next_values = torch.stack(buffer.truncated_next_values, dim=0)  # (T,E)
        has_truncated_next_values = torch.stack(buffer.has_truncated_next_values, dim=0)  # (T,E)

        T, E = rewards.shape

        with torch.no_grad():
            if next_global_states is None:
                # 逐环境检查末步：若存在截断且非真实终止、同时既无已存后继价值又无传入后继状态，则显式报错
                last_missing_trunc = truncated[-1] & (~dones[-1]) & (~has_truncated_next_values[-1])
                if last_missing_trunc.any():
                    raise ValueError(
                        "Final truncated step requires next_global_states or truncated_next_values "
                        "to compute exact bootstrap V(s_{t+1})."
                    )
                last_value = torch.zeros(E, device=DEVICE)
            else:
                if next_global_states.dim() != 2 or tuple(next_global_states.shape) != (
                    E,
                    buffer.global_state_dim,
                ):
                    raise ValueError(
                        f"next_global_states: expected shape {(E, buffer.global_state_dim)}, "
                        f"got {tuple(next_global_states.shape)}"
                    )
                if next_dones is not None and tuple(next_dones.shape) != (E,):
                    raise ValueError(f"next_dones: expected shape {(E,)}, got {tuple(next_dones.shape)}")
                lv = self.value_net(next_global_states).squeeze(-1)
                if next_dones is not None:
                    lv = lv * (~next_dones.bool()).float()
                last_value = lv

            if episode_end:
                # 时间边界由本局采集器声明，复用真实后继的这一次 Value 前向（含 Dropout）。
                truncated[-1] = True
                truncated_next_values[-1] = last_value
                has_truncated_next_values[-1] = True

            adv, ret = self._compute_gae(
                rewards=rewards,
                dones=dones,
                values=values.squeeze(-1),
                last_value=last_value,
                truncated=truncated,
                truncated_next_values=truncated_next_values,
                has_truncated_next_values=has_truncated_next_values,
            )

            adv_flat = adv.reshape(-1)
            adv = (adv - adv_flat.mean()) / (adv_flat.std(unbiased=False) + 1e-8)

        N = T * E
        follower_states_env = follower_states.reshape(N, buffer.num_followers, STATE_DIM)
        global_states_env = global_states.reshape(N, -1)
        actions_env = actions.reshape(N, buffer.num_followers, ACTION_DIM)
        pre_squash_v_env = torch.stack(buffer.pre_squash_v, dim=0).reshape(N, buffer.num_followers, 1)
        has_pre_squash_v_env = torch.stack(buffer.has_pre_squash_v, dim=0).reshape(N, buffer.num_followers, 1)
        any_pre_squash = bool(has_pre_squash_v_env.any().item())
        old_logp_env = old_logp.reshape(N, buffer.num_followers, 1)
        adv_env = adv.reshape(N, 1)
        ret_env = ret.reshape(N, 1)

        mb_size = int(PPO_MINIBATCH_SIZE)
        clip_eps = float(PPO_CLIP_EPS)

        pol_losses = []
        v_losses = []
        entropies = []
        kls = []
        clipfracs = []
        early_stop = False

        for _ in range(int(PPO_EPOCHS)):
            perm = torch.randperm(N, device=DEVICE)
            for start in range(0, N, mb_size):
                idx = perm[start : start + mb_size]

                mb_gs = global_states_env[idx]
                mb_ret = ret_env[idx]
                chunk_size = (len(idx) if self.actor_microbatch_max_rows is None else
                              self.actor_microbatch_max_rows // buffer.num_followers)
                self.actor_optimizer.zero_grad(set_to_none=True)
                policy_rows, entropy_rows, kl_rows, clipped_rows = [], [], [], []
                for chunk in idx.split(chunk_size):
                    pre_v = pre_squash_v_env[chunk].reshape(-1, 1) if any_pre_squash else None
                    has_pre_v = has_pre_squash_v_env[chunk].reshape(-1, 1) if any_pre_squash else None
                    new_logp, entropy = self.actor.evaluate_actions(
                        follower_states_env[chunk].reshape(-1, STATE_DIM),
                        actions_env[chunk].reshape(-1, ACTION_DIM),
                        return_entropy=True, pre_squash_v=pre_v, has_pre_squash_v=has_pre_v,
                    )
                    log_ratio = new_logp.reshape(-1, buffer.num_followers, 1) - old_logp_env[chunk]
                    ratio = log_ratio.exp()
                    advantage = adv_env[chunk].unsqueeze(1)
                    policy = -torch.minimum(
                        ratio * advantage, ratio.clamp(1.0 - clip_eps, 1.0 + clip_eps) * advantage
                    )
                    # 完整环境记录含相同数量的 follower；尾块按实际行数计权。
                    weight = len(chunk) / len(idx)
                    ((policy.mean() - float(PPO_ENTROPY_COEF) * entropy.mean()) * weight).backward()
                    with torch.no_grad():
                        policy_rows.append(policy.detach())
                        entropy_rows.append(entropy.detach())
                        kl_rows.append((ratio - 1.0) - log_ratio)
                        clipped_rows.append(((ratio - 1.0).abs() > clip_eps).float())
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), float(PPO_MAX_GRAD_NORM))
                self.actor_optimizer.step()

                v_pred = self.value_net(mb_gs)
                value_loss = F.mse_loss(v_pred, mb_ret)

                self.value_optimizer.zero_grad(set_to_none=True)
                (float(PPO_VALUE_COEF) * value_loss).backward()
                torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), float(PPO_MAX_GRAD_NORM))
                self.value_optimizer.step()

                # 在同一逻辑批内按原行序归约，避免分块舍入改变 KL 早停边界。
                stats = [torch.cat(parts).mean() for parts in (policy_rows, entropy_rows, kl_rows, clipped_rows)]
                policy_mean, entropy_mean, kl_val, clipfrac_mean = torch.stack(stats).tolist()
                pol_losses.append(policy_mean)
                v_losses.append(float(value_loss.detach().item()))
                entropies.append(entropy_mean)
                kls.append(kl_val)
                clipfracs.append(clipfrac_mean)

                if float(PPO_TARGET_KL) > 0 and kl_val > 1.5 * float(PPO_TARGET_KL):
                    early_stop = True
                    break

            if early_stop:
                break

        buffer.clear()

        mean_entropy = float(np.mean(entropies)) if entropies else float("nan")
        self.last_losses = {
            "policy": float(np.mean(pol_losses)) if pol_losses else float("nan"),
            "value": float(np.mean(v_losses)) if v_losses else float("nan"),
            "entropy": mean_entropy,
            "entropy_joint": mean_entropy * float(buffer.num_followers) if entropies else float("nan"),
            "kl": float(np.mean(kls)) if kls else float("nan"),
            "clipfrac": float(np.mean(clipfracs)) if clipfracs else float("nan"),
            "early_stopped": early_stop,
            "update_minibatches": len(pol_losses),
        }
        return self.last_losses

    def save(self, path: str):
        """保存模型参数到文件。

        Args:
            path: 保存路径。
        """
        _save_checkpoint(path, self.actor, "MAPPO", value=self.value_net.state_dict())
        print(f"✅ CTDE MAPPO Model saved to {path}")

    def save_actor(self, path: str):
        _save_checkpoint(path, self.actor, "MAPPO")

    def load(self, path: str):
        """从文件加载模型参数。

        Args:
            path: checkpoint 路径。
        """
        checkpoint, _ = _read_checkpoint(path, algorithm="MAPPO", hidden_dim=self.actor.local_encoder[0].out_features)
        _restore_modules(checkpoint, actor=self.actor, value=self.value_net)
        print(f"✅ CTDE MAPPO Model loaded from {path}")


class _RolloutProgress:
    """只读采集进度；批次内容及写口由 session 独占。"""

    def __init__(self, buffer):
        self._buffer = buffer

    def __len__(self):
        return len(self._buffer)

    def is_full(self):
        return self._buffer.is_full()

    def add(self, *args, **kwargs):
        raise RuntimeError("Active episode owns rollout records; call session.step_collect()")


class MultiScaleEpisodeSession:
    """在一次完整 reset 与末帧更新之间持有环境、当前帧以及专用 rollout。"""

    def __init__(self, agent, env):
        agent._validate_microbatch_cap(env.num_followers)
        with _try_lock(agent._session_lock), env._session_operation(self), ExitStack() as lease:
            if agent._session is not None or len(agent._fixed_buffer):
                raise RuntimeError("Close the active session and flush pending rollout records first")
            lease.enter_context(env.topology._hold_revision())
            self._bind(agent, env)
            self._graph_lease = lease.pop_all()
            agent._session = env._episode_session = self

    def _bind(self, agent, env):
        topology = env.topology
        if (env.frame_rev != env._episode_start_frame or
                env._precomputed_topology is not topology or env._cached_topology_rev != topology.revision):
            raise ValueError("Start the episode after a full reset with an up-to-date graph")
        self._agent = agent
        self._env = env
        self._topology = topology
        self._topology_revision = topology.revision
        self._episode_gen = env.episode_gen
        self._graph_rev = env.graph_rev
        self._frame_rev = env.frame_rev
        self._last_frame = env.frame_rev + env.max_steps
        self._local = env._get_state_optimized()
        self._global = env.get_global_state()
        self._num_envs = env.num_envs
        n = env.num_followers
        expected_g = agent.value_net.ctx_dim + (n + 1) * agent.value_net.agent_feat_dim
        if (env.global_state_dim != expected_g or env.state_dim != STATE_DIM or
                self._global.shape != (env.num_envs, expected_g) or
                self._local.shape != (env.num_envs, n + 1, STATE_DIM)):
            raise ValueError("Episode local/global state dimensions are incompatible with actor/value network")
        self._buffer = _PPOBuffer(num_followers=n, state_dim=STATE_DIM,
                                  global_state_dim=expected_g, action_dim=ACTION_DIM)
        self.buffer = _RolloutProgress(self._buffer)
        self._ended = False
        self._aborted = False

    def _check_frame(self, expected_frame):
        env = self._env
        if self._agent._session is not self or env._episode_session is not self or self._ended:
            raise RuntimeError("Episode session is closed or has already reached its final update")
        if (env.frame_rev != expected_frame or env.episode_gen != self._episode_gen or
                env.graph_rev != self._graph_rev or env.topology is not self._topology or
                env.topology.revision != self._topology_revision or
                env.topology.revision != env._cached_topology_rev or
                env.num_envs != self._num_envs or env.num_followers != self._buffer.num_followers or
                env.state_dim != self._buffer.state_dim or env.global_state_dim != self._buffer.global_state_dim):
            raise RuntimeError("Environment frame, graph or episode changed outside the active session")

    def step_collect(self):
        with self._env._session_operation(self) as token:
            self._check_frame(self._frame_rev)
            if self._frame_rev == self._last_frame:
                raise RuntimeError("Time limit reached; flush the final rollout before closing")
            if self._buffer.is_full():
                raise RuntimeError("Rollout is full; call flush_update() before collecting another step")
            actions, logp, values, pre = self._agent._act_for_buffer(
                self._local, self._global, self._buffer, False, True, bind_sample=False
            )
            expected_actions = (self._num_envs, self._buffer.num_followers, ACTION_DIM)
            expected_logp = (self._num_envs, self._buffer.num_followers, 1)
            expected_values = (self._num_envs, 1)
            if (not isinstance(actions, torch.Tensor) or tuple(actions.shape) != expected_actions or
                    not isinstance(logp, torch.Tensor) or tuple(logp.shape) != expected_logp or
                    not isinstance(values, torch.Tensor) or tuple(values.shape) != expected_values or
                    not isinstance(pre, torch.Tensor) or tuple(pre.shape) != expected_logp):
                raise ValueError(
                    f"Unexpected act() output shapes before env.step: "
                    f"actions={getattr(actions, 'shape', type(actions))}, "
                    f"logp={getattr(logp, 'shape', type(logp))}, "
                    f"values={getattr(values, 'shape', type(values))}, "
                    f"pre={getattr(pre, 'shape', type(pre))}"
                )
            self._check_frame(self._frame_rev)
            next_local, rewards, dones, infos = self._env.step(actions, _session=token)
            next_global = self._env.get_global_state()
            self._check_frame(self._frame_rev + 1)
            self._buffer.add(self._local[:, 1:], self._global, actions, logp, values,
                             rewards, dones, pre_squash_v=pre)
            self._local, self._global = next_local, next_global
            self._frame_rev += 1
            return rewards, dones, infos

    def flush_update(self, is_episode_end=False):
        with self._env._session_operation(self):
            self._check_frame(self._frame_rev)
            if bool(is_episode_end) != (self._frame_rev == self._last_frame):
                raise ValueError("is_episode_end must match the environment time limit")
            if not len(self._buffer):
                raise RuntimeError("Cannot update an empty episode rollout")
            losses = self._agent._update_buffer(
                self._buffer, self._global, self._buffer.dones[-1], episode_end=is_episode_end
            )
            self._ended = bool(is_episode_end)
            return losses

    def abort(self):
        """作废未完成或异常中断的采集会话，丢弃未刷新数据并释放 agent、环境与图租约。

        不视作成功 flush；作废后须执行完整 env.reset() 才能绑定新会话。
        """
        with _try_lock(self._agent._session_lock):
            if self._agent._session is not self:
                return
            with self._env._session_operation(self):
                self._buffer.clear()
                self._ended = True
                self._aborted = True
                self._graph_lease.close()
                self._agent._session = None
                self._env._episode_session = None
                self._env._episode_start_frame = None

    def close(self):
        with _try_lock(self._agent._session_lock), self._env._session_operation(self):
            if self._aborted:
                raise RuntimeError("Episode session was aborted; do not treat it as a flushed close()")
            if not self._ended or len(self._buffer):
                raise RuntimeError("Flush the final rollout with is_episode_end=True before close()")
            if self._agent._session is self:
                self._graph_lease.close()
                self._agent._session = None
                self._env._episode_session = None
