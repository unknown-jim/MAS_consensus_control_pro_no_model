"""冻结共识控制场景、跨 checkout 物理门禁、Actor 评测与配对统计。

示例（各行参数加在 python evaluate_scaling.py 后）：
  --generate-manifest scenes.json --scales 5 20 100 --num-seeds 2
  --parity-check --scenario-manifest scenes.json --v1-repo ../baseline --output parity.json
  --evaluate --scenario-manifest scenes.json --worker-version v1 --repo-path ../baseline
             --training-seed 11 --checkpoint old.pt --output v1.json
  --evaluate --scenario-manifest scenes.json --training-seed 11 --checkpoint new.pt --output v2.json
  --compare v1.json v2.json --output report.json --markdown-out report.md
  --compare --v1-results a11.json a23.json --v2-results b11.json b23.json --output report.json
"""
from __future__ import annotations

import argparse
from collections import defaultdict, deque
from contextlib import ExitStack, redirect_stdout
import hashlib
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parent
BASELINE = "4115d00412d14cc2600235aa0e99af5e90f131f9"
BASELINE_MAS_CC_SHA256 = "bca16136b2831bff6666cc695b4df9011325cb9fce2c152e16db03b1637754c0"
METRICS = ("mean_tracking_error", "last50_error", "final_tracking_error", "max_tracking_error",
           "deepest_layer_error", "mean_comm_rate", "mean_reward", "reward_per_step",
           "convergence_step", "converged", "penalized_convergence_step", "stable",
           "valid_est_coverage", "est_age_p95", "high_age_ratio", "first_est_step_mean",
           "all_finite")
CELL = ("N", "pinned_mode", "depth_mode")
STRATA = (*CELL, "actual_pinned", "actual_H")


def digest(obj):
    data = json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(data.encode()).hexdigest()


def write_json(path, obj):
    data = json.dumps(obj, indent=2, allow_nan=False)
    if path:
        Path(path).write_text(data + "\n", encoding="utf-8")
    else:
        print(data)


def validate_frozen_graph(s):
    """按 adj[receiver, sender] 从 leader 重算可达性及评测所用的分层。"""
    matrix = np.asarray(s["adj_matrix"])
    n = s["N"] + 1
    if (s["leader_id"] != 0 or matrix.shape != (n, n) or not np.isin(matrix, (0, 1)).all()
            or matrix.diagonal().any()):
        raise ValueError("Invalid frozen adjacency matrix or leader")
    pins = np.flatnonzero(matrix[:, 0]).tolist()
    declared = np.asarray(s["pinned_ids"])
    if (not pins or declared.ndim != 1 or declared.dtype.kind not in "iu"
            or pins != sorted(declared.tolist())):
        raise ValueError("Frozen pinned IDs must match leader column 0")
    depths = np.full(n, -1, dtype=int)
    depths[0] = 0
    queue = deque([0])
    while queue:
        sender = queue.popleft()
        receivers = np.flatnonzero((matrix[:, sender] != 0) & (depths < 0))
        depths[receivers] = depths[sender] + 1
        queue.extend(receivers)
    if (depths < 0).any():
        raise ValueError("Every frozen node must be reachable from leader 0")
    stats = s.get("bfs_stats", {})
    if not np.array_equal(stats.get("node_depths"), depths) or stats.get("max_depth") != depths.max():
        raise ValueError("Frozen BFS statistics disagree with directed distances from leader 0")
    return matrix, pins


def read_manifest(path):
    m = json.loads(Path(path).read_text())
    if m["schema_version"] != 1 or m["steps"] < 1 or not m["scenarios"]:
        raise ValueError("Invalid or unsupported manifest")
    for obj, key in [(m, "manifest_sha256"), *((s, "scenario_sha256") for s in m["scenarios"])]:
        if obj[key] != digest({k: v for k, v in obj.items() if k != key}):
            raise ValueError(f"Invalid {key} for {obj.get('scenario_id', path)}")
    ids = [s["scenario_id"] for s in m["scenarios"]]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate scenario_id")
    for s in m["scenarios"]:
        validate_frozen_graph(s)
    return m


def trajectory_extrema(kind, params, duration):
    """首末帧和区间内所有四分之一周期点覆盖位置、速度的全部极值。"""
    a, w, p = (params[k] for k in ("amplitude", "omega", "phase"))
    end = w * duration + p
    lo, hi = sorted((p, end))
    quarter = np.pi / 2
    corners = np.arange(np.ceil(lo / quarter), np.floor(hi / quarter) + 1)
    phase = np.r_[p, corners * quarter, end]
    if kind == "sine":
        return a * np.sin(phase), a * w * np.cos(phase)
    if kind == "cosine":
        return a * np.cos(phase), -a * w * np.sin(phase)
    raise ValueError(f"Unsupported trajectory {kind}")


def generate_manifest(args):
    from mas_cc import config as c
    from mas_cc.topology import CommunicationTopology

    scenarios = []
    modes = ("normal", "low")
    for n, depth, seed in itertools.product(args.scales, args.depth_modes,
                                            range(args.seed_start, args.seed_start + args.num_seeds)):
        # 物理初态与图分流。同一 cell 独立可再生，改变矩阵排列或子集不改变其内容。
        rng = np.random.default_rng([seed, n])
        kind = str(rng.choice(c.LEADER_TRAJECTORY_TYPES))
        params = {key: float(np.float32(rng.uniform(*bounds))) for key, bounds in (
            ("amplitude", c.LEADER_AMPLITUDE_RANGE), ("omega", c.LEADER_OMEGA_RANGE),
            ("phase", c.LEADER_PHASE_RANGE))}
        x, v = trajectory_extrema(kind, params, args.steps * c.DT)
        init = {}
        for key, leader, spread, limit in (("init_pos", x[0], c.FOLLOWER_INIT_POS_STD_RANGE, c.POS_LIMIT),
                                           ("init_vel", v[0], c.FOLLOWER_INIT_VEL_STD_RANGE, c.VEL_LIMIT)):
            followers = leader + rng.normal(0, rng.uniform(*spread), n)
            init[key] = np.r_[leader, np.clip(followers, -limit, limit)].astype("float32").tolist()
        if not np.allclose([init["init_pos"][0], init["init_vel"][0]], [x[0], v[0]]):
            raise ValueError("Frozen leader frame disagrees with trajectory")
        graphs = {mode: CommunicationTopology(
            n, pinned_mode=mode, depth_mode=depth, device="cpu",
            rng=np.random.default_rng([seed, n, int(depth == "deep"), i]))
            for i, mode in enumerate(modes)}
        coincident = graphs["normal"].num_pinned == graphs["low"].num_pinned
        for mode in args.pinned_modes:
            g = graphs[mode]
            s = dict(scenario_id=f"n{n}-{mode}-{depth}-{seed}", N=n, seed=seed, leader_id=0,
                     pinned_mode=mode, depth_mode=depth, adj_matrix=g.adj.numpy().astype("int8").tolist(),
                     pinned_ids=g.pinned_ids, traj_type=kind, traj_params=params, **init,
                     bfs_stats=g.stats, traj_in_bounds=bool((abs(x) <= c.POS_LIMIT).all()
                                                           and (abs(v) <= c.VEL_LIMIT).all()),
                     pinned_ratio_coincident=coincident)
            s["scenario_sha256"] = digest(s)
            scenarios.append(s)
    m = dict(schema_version=1, steps=args.steps, dt=c.DT, pos_limit=c.POS_LIMIT, vel_limit=c.VEL_LIMIT,
             adj_dtype="int8", state_dtype="float32", scenarios=scenarios)
    m["manifest_sha256"] = digest(m)
    write_json(args.generate_manifest, m)


class FrozenGraph:
    """旧版环境只读取该图。返回副本，拒绝可变状态从一个场景泄漏到下一个。"""
    leader_id = 0

    def __init__(self, s, device):
        import torch
        matrix, pins = validate_frozen_graph(s)
        self.num_followers = s["N"]
        self.num_agents = self.num_followers + 1
        self._pins = tuple(pins)
        self._matrix = torch.tensor(matrix, dtype=torch.float32, device=device)

    @property
    def adj_matrix(self):
        return self._matrix.clone()

    adj = adj_matrix

    @property
    def pinned_ids(self):
        return list(self._pins)

    pinned_followers = pinned_ids

    def randomize(self):
        return self.pinned_ids

    def get_receivers(self, sender_id):
        return self._matrix[:, sender_id].nonzero().flatten().tolist()

    def get_neighbors(self, node_id):
        return self._matrix[node_id].nonzero().flatten().tolist()

    def can_receive(self, receiver_id, sender_id):
        return bool(self._matrix[receiver_id, sender_id])


def cold_start(s, version, device):
    import torch
    from mas_cc.environment import BatchedModelFreeEnv

    if version == "v2":
        from mas_cc.topology import CommunicationTopology
        graph = CommunicationTopology.from_frozen(s["adj_matrix"], s["pinned_ids"], device=device)
        env = BatchedModelFreeEnv(graph, num_envs=1, auto_reset=False, randomize_topology_on_reset=False)
        state = env.reset(scenario=dict(s, trajectory_type=s["traj_type"], initial_positions=s["init_pos"],
                                        initial_velocities=s["init_vel"],
                                        **{"leader_" + k: v for k, v in s["traj_params"].items()}))
    else:
        class ScenarioEnv(BatchedModelFreeEnv):
            def _randomize_leader_dynamics(self, env_ids):
                for k, val in s["traj_params"].items():
                    getattr(self, "leader_" + k)[env_ids] = val
                self.trajectory_type_ids[env_ids] = self.type_to_id[s["traj_type"]]

            def _randomize_follower_init(self, env_ids, leader_pos, leader_vel):
                self.positions[env_ids, 1:] = torch.tensor(s["init_pos"][1:], device=device)
                self.velocities[env_ids, 1:] = torch.tensor(s["init_vel"][1:], device=device)

        env = ScenarioEnv(FrozenGraph(s, device), num_envs=1)  # 构造函数已调用原版 reset。
        state = env._get_state_optimized()
    return env, state


def fixed_actions(n, steps, seed):
    rng = np.random.default_rng(seed)
    tape = np.empty((steps, 1, n, 2), dtype=np.float32)
    tape[..., 0] = rng.uniform(-1, 1, (steps, 1, n))
    tape[:, 0, :, 1] = (np.arange(steps)[:, None] // 5 + np.arange(n)) % 2
    return tape


def episode_frames(s, steps, version, device, policy):
    """首帧编号 0，之后的物理帧编号 1..T。调用端在推进下一帧前消费或复制张量。"""
    import torch
    env, state = cold_start(s, version, device)
    if env._cooldown_steps <= 0:
        raise ValueError("Per-node trigger readback needs a positive cooldown")
    tape = None if policy else torch.from_numpy(fixed_actions(s["N"], steps, s["seed"])).to(device)
    def frame(reward, done, triggered, finite):
        return dict(positions=env.positions[0], velocities=env.velocities[0],
                    leader_est_seq=env.leader_est_seq[0], broadcast_leader_est_seq=env.last_broadcast_leader_seq[0],
                    triggered=triggered[0], reward=reward[0], done=done[0], all_finite=finite)

    zeros = env.t.new_zeros(1)
    yield frame(zeros, zeros.bool(), torch.zeros_like(env._cooldown_counter, dtype=torch.bool),
                state.isfinite().all())
    for t in range(steps):
        action = tape[t] if policy is None else policy.select_action(state, deterministic=True)
        state, reward, done, info = env.step(action)
        # 即使广播携带旧序号，触发也会把计数器重置为 cooldown_steps。
        triggered = env._cooldown_counter == env._cooldown_steps
        if not torch.allclose(triggered.float().mean(dim=1), info["comm_rate"], rtol=0, atol=1e-7):
            raise RuntimeError("Per-node trigger readback disagrees with environment comm_rate")
        finite = state.isfinite().all() & action.isfinite().all() & reward.isfinite().all()
        yield frame(reward, done, triggered, finite)


def rollout(s, steps, version, device, policy):
    import torch
    frames = defaultdict(list)
    for frame in episode_frames(s, steps, version, device, policy):
        for k, tensor in frame.items():
            frames[k].append(tensor.clone())
    return {k: torch.stack(v).cpu().numpy() for k, v in frames.items()}


def metrics(trace, s):
    steps = len(trace["reward"]) - 1
    pos, vel = (trace[k][1:].astype("float64") for k in ("positions", "velocities"))
    node_error = abs(pos[:, 1:] - pos[:, :1]) + .01 * abs(vel[:, 1:] - vel[:, :1])
    err = node_error.mean(axis=1)
    deepest = np.array(s["bfs_stats"]["node_depths"][1:]) == s["bfs_stats"]["max_depth"]
    seq = trace["leader_est_seq"][1:, 1:]
    valid = seq >= 0
    ages = (np.arange(1, steps + 1)[:, None] - seq)[valid]
    unpinned = ~np.isin(np.arange(1, s["N"] + 1), s["pinned_ids"])
    # 以第一个动作后的观测为 step 0，未收到估计的节点没有可观测的首次到达时刻。
    first = valid[:, unpinned & valid.any(axis=0)].argmax(axis=0)
    streak = 0
    convergence = None
    for i, error in enumerate(err):
        streak = streak + 1 if error <= .5 else 0
        if streak == 20:
            convergence = i - 19
            break
    reward = float(trace["reward"][1:].sum(dtype="float64"))
    finite = bool(trace["all_finite"].all() and np.isfinite(node_error).all())
    return dict(mean_tracking_error=float(err.mean()), last50_error=float(err[-50:].mean()),
                final_tracking_error=float(err[-1]), max_tracking_error=float(err.max()),
                deepest_layer_error=float(node_error[:, deepest].mean()),
                mean_comm_rate=float(trace["triggered"][1:].mean()), mean_reward=reward,
                reward_per_step=reward / steps, convergence_step=convergence, converged=convergence is not None,
                penalized_convergence_step=steps+1 if convergence is None else convergence,
                stable=bool(finite and err[-1] < 1 and err.max() < 5), all_finite=finite,
                valid_est_coverage=float(valid.mean()), est_age_p95=float(np.percentile(ages, 95)) if ages.size else None,
                high_age_ratio=float((ages > 10).mean()) if ages.size else None,
                first_est_step_mean=float(first.mean()) if first.size else None)


def row_id(s):
    return {**{k: s[k] for k in (*CELL, "scenario_id", "scenario_sha256", "seed", "pinned_ratio_coincident")},
            "actual_pinned": len(s["pinned_ids"]), "actual_H": s["bfs_stats"]["max_depth"]}


def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[k] for k in CELL)].append(row)
    cells = []
    for key, group in sorted(groups.items()):
        ok = [r for r in group if r["status"] == "ok"]
        cell = dict(zip(CELL, key), n_ok=len(ok), n_total=len(group),
                    actual_pinned=sorted({r["actual_pinned"] for r in group}),
                    actual_H=sorted({r["actual_H"] for r in group}),
                    pinned_ratio_coincident=any(r["pinned_ratio_coincident"] for r in group))
        for k in METRICS:
            values = [r[k] for r in ok if r[k] is not None]
            cell[k] = float(np.mean(values)) if values else None
        cell["all_finite"] = len(ok) == len(group) and all(r["all_finite"] for r in ok)
        cell["stability_rate"] = sum(r["stable"] for r in ok) / len(group)
        cell["convergence_rate"] = sum(r["converged"] for r in ok) / len(group)
        cells.append(cell)
    return cells


def worker_command(args, version, repo, mode):
    cmd = [sys.executable, "-B", str(Path(__file__).resolve()), "--worker", mode,
           "--repo-path", str(Path(repo).resolve()), "--worker-version", version,
           "--scenario-manifest", str(Path(args.scenario_manifest).resolve()),
           "--device", args.device or ("cpu" if mode == "trace" else "auto")]
    for key in ("max_scenarios", "top_k", "training_seed"):
        if getattr(args, key) is not None:
            cmd.extend(["--" + key.replace("_", "-"), str(getattr(args, key))])
    if mode == "evaluate":
        cmd.extend(["--checkpoint", str(Path(args.checkpoint).resolve()), "--output", str(Path(args.output).resolve())])
    return cmd


def git(repo, *args):
    return subprocess.check_output(["git", "--no-optional-locks", "-C", str(repo), *args], text=True, stderr=subprocess.PIPE).strip()


def mas_cc_digest(repo):
    root = Path(repo) / "mas_cc"
    if not root.is_dir():
        raise ValueError(f"Missing mas_cc directory under {repo}")
    h = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        if any(part.startswith(".") or part == "__pycache__" for part in path.relative_to(root).parts):
            continue
        rel = path.relative_to(root).as_posix()
        h.update(rel.encode("utf-8") + b"\0" + path.read_bytes() + b"\0")
    return h.hexdigest()


def resolve_repo_identity(repo, worker_version):
    if (Path(repo) / ".git").exists():
        sha = git(repo, "rev-parse", "HEAD")
        if worker_version == "v1" and (sha != BASELINE or git(repo, "status", "--porcelain", "--", "mas_cc")):
            raise ValueError(f"v1 requires the unmodified baseline {BASELINE}")
        return sha
    code_hash = mas_cc_digest(repo)
    if worker_version == "v1" and code_hash != BASELINE_MAS_CC_SHA256:
        raise ValueError(
            f"v1 non-git export mas_cc digest mismatch: got {code_hash}, expected {BASELINE_MAS_CC_SHA256}"
        )
    return f"nongit:{code_hash}"


def error_status(exc):
    import torch
    if isinstance(exc, (MemoryError, torch.cuda.OutOfMemoryError)) or "out of memory" in str(exc).lower():
        return "oom"
    return "error"


def run_worker(args, stream):
    # config 与网络模块在 import 时绑定设备和 K；每个解释器只导入一个 checkout。
    repo = Path(args.repo_path).resolve()
    if args.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    if args.top_k is not None:
        os.environ["MAX_NEIGHBORS"] = str(args.top_k)
    sha = resolve_repo_identity(repo, args.worker_version)
    sys.path.insert(0, str(repo))
    import torch
    from mas_cc import config as c
    from mas_cc.agent import ActorOnlyPolicy
    version = getattr(c, "OBS_VERSION", 1)
    if Path(c.__file__).resolve().parent.parent != repo or f"v{version}" != args.worker_version:
        raise ValueError("Worker imported the wrong checkout or observation version")
    device_unavailable = args.device != "auto" and str(c.DEVICE) != args.device
    torch.set_default_dtype(torch.float32)
    if c.DEVICE.type == "cpu":
        torch.set_num_threads(1)
    m = read_manifest(args.scenario_manifest)
    scenarios = m["scenarios"][:args.max_scenarios]
    start = time.perf_counter()
    if args.worker == "trace":
        if device_unavailable:
            raise ValueError(f"Requested device {args.device} unavailable")
        with torch.inference_mode():
            for s in scenarios:
                record = np.empty(1, dtype=frame_dtype(s["N"]))
                for frame in episode_frames(s, m["steps"], args.worker_version, c.DEVICE, None):
                    for key, tensor in frame.items():
                        record[key][0] = tensor.cpu().numpy()
                    stream.write(record.tobytes())
                    stream.flush()
        return 0

    checkpoint = Path(args.checkpoint)
    rows = []
    load_error = None
    checkpoint_sha = None
    try:
        h = hashlib.sha256()
        with checkpoint.open("rb") as f:
            for block in iter(lambda: f.read(1024 * 1024), b""):
                h.update(block)
        checkpoint_sha = h.hexdigest()
        if device_unavailable:
            raise ValueError(f"Requested device {args.device} unavailable")
        policy = ActorOnlyPolicy.from_checkpoint(str(checkpoint))
    except Exception as exc:
        load_error = (error_status(exc), str(exc))
    for s in scenarios:
        row = row_id(s)
        if load_error:
            row.update(status=load_error[0], error=load_error[1], all_finite=False)
        else:
            try:
                with torch.inference_mode():
                    row.update(metrics(rollout(s, m["steps"], args.worker_version, c.DEVICE, policy), s))
                if not row["all_finite"]:
                    raise ValueError("Non-finite action, state or reward")
                row["status"] = "ok"
            except Exception as exc:
                row = dict(row_id(s), status=error_status(exc), error=str(exc), all_finite=False)
        rows.append(row)
    write_json(args.output, dict(repo_sha=sha, checkpoint_path=str(checkpoint), checkpoint_sha256=checkpoint_sha,
                                 worker_version=args.worker_version, training_seed=args.training_seed,
                                 top_k=c.MAX_NEIGHBORS, obs_version=version, device=str(c.DEVICE), requested_device=args.device,
                                 torch_version=str(torch.__version__), cuda_version=torch.version.cuda, dtype="float32",
                                 manifest_sha256=m["manifest_sha256"], steps=m["steps"], manifest_scenario_count=len(m["scenarios"]),
                                 n_total=len(scenarios),
                                 elapsed_sec=time.perf_counter()-start, scenarios=rows, cells=summarize(rows)))
    return int(any(row["status"] != "ok" for row in rows))


def frame_dtype(n):
    return np.dtype([("positions", "<f4", n + 1), ("velocities", "<f4", n + 1),
                     ("leader_est_seq", "<i8", n + 1), ("broadcast_leader_est_seq", "<i8", n + 1),
                     ("triggered", "?", (n,)), ("reward", "<f4"), ("done", "?"), ("all_finite", "?")])


def read_frame(stream, dtype):
    packet = stream.read(dtype.itemsize)
    if len(packet) != dtype.itemsize:
        raise EOFError("Worker ended before the next complete frame")
    return np.frombuffer(packet, dtype=dtype)[0]


def first_divergence(sid, step, left, right):
    for key in left.dtype.names:
        a, b = left[key], right[key]
        if key in ("positions", "velocities", "reward"):
            match = np.isclose(a, b, rtol=0, atol=1e-5)
        elif key == "all_finite":
            match = a & b
        else:
            match = a == b
        bad = np.flatnonzero(~match)
        if bad.size:
            index = int(bad[0])
            node = index + int(key == "triggered") if a.ndim else None
            values = [(v[index] if a.ndim else v).item() for v in (a, b)]
            return dict(scenario_id=sid, step=step, field=key, node_idx=node,
                        v1_val=values[0] if np.isfinite(values[0]) else str(values[0]),
                        v2_val=values[1] if np.isfinite(values[1]) else str(values[1]))
    return None


def compare_episode(workers, s, steps):
    dtype = frame_dtype(s["N"])
    triggered = 0
    for step in range(steps + 1):
        left, right = (read_frame(proc.stdout, dtype) for proc in workers)
        # 两帧字节相同时省去逐字段容差检查，非有限值仍须立即失败。
        if not left["all_finite"] or left.tobytes() != right.tobytes():
            diff = first_divergence(s["scenario_id"], step, left, right)
            if diff:
                return None, diff
        if step:
            triggered += int(left["triggered"].sum())
    return dict(scenario_id=s["scenario_id"], frames=steps + 1,
                triggered=triggered, not_triggered=s["N"] * steps - triggered), None


def parity(args):
    m = read_manifest(args.scenario_manifest)
    report = dict(manifest_sha256=m["manifest_sha256"], steps=m["steps"], passed=False,
                  scenarios=[], triggered=0, not_triggered=0, first_divergence=None)
    # 有界管道：每个 checkout 常驻一个解释器，父进程逐帧比较，发现分叉即终止两个 worker。
    with ExitStack() as stack:
        workers = []
        try:
            for version, repo in (("v1", args.v1_repo), ("v2", args.v2_repo)):
                proc = stack.enter_context(subprocess.Popen(worker_command(args, version, repo, "trace"),
                                                            stdout=subprocess.PIPE))
                workers.append(proc)
                stack.callback(lambda p=proc: p.terminate() if p.poll() is None else None)
            for s in m["scenarios"][:args.max_scenarios]:
                result, diff = compare_episode(workers, s, m["steps"])
                if diff:
                    report["first_divergence"] = diff
                    break
                report["scenarios"].append(result)
                report["triggered"] += result["triggered"]
                report["not_triggered"] += result["not_triggered"]
            else:
                if any(p.wait() for p in workers):
                    raise ValueError("Trace worker exited unsuccessfully")
                report["passed"] = True
        except (OSError, EOFError, ValueError) as exc:
            report["error"] = str(exc)
    write_json(args.output, report)
    return int(not report["passed"])


def nullable_mean(values, axis):
    valid = np.isfinite(values)
    count = valid.sum(axis=axis)
    total = np.where(valid, values, 0.).sum(axis=axis)
    return np.divide(total, count, out=np.full_like(total, np.nan), where=count > 0)


def number(value):
    return float(value) if np.isfinite(value) else None


def interval(samples):
    valid = samples[np.isfinite(samples)]
    return np.percentile(valid, [2.5, 97.5]).tolist() if valid.size else None


def paired_cell(rows, seeds, seed_draws, rng):
    """先按训练 seed 分层，再整局重采样。每次抽样对两个版本使用同一组索引。"""
    width = len(METRICS)
    shape = (*seed_draws.shape, width)
    bootstrap = np.full(shape, np.nan)
    by_train = []
    for index, seed in enumerate(seeds):
        pairs = [p for p in rows if p["training_seed"] == seed and p["paired_ok"]]
        if not pairs:
            continue
        values = np.array([[[p[v][metric] for metric in METRICS] for v in ("v1", "v2")]
                           for p in pairs], dtype=float)
        # 可空的指标只使用两侧同时可测的场景。
        values = np.where(np.isfinite(values).all(axis=1, keepdims=True), values, np.nan)
        by_train.append(nullable_mean(values, axis=0))
        delta = values[:, 1] - values[:, 0]
        slots = seed_draws == index
        weights = rng.multinomial(len(pairs), np.full(len(pairs), 1 / len(pairs)), size=slots.sum())
        count = weights @ np.isfinite(delta).astype(float)
        total = weights @ np.nan_to_num(delta)
        bootstrap[slots] = np.divide(total, count, out=np.full_like(total, np.nan), where=count > 0)
    avg = nullable_mean(np.stack(by_train), 0) if by_train else np.full((2, width), np.nan)
    draws = nullable_mean(bootstrap, 1)
    stats = {k: dict(v1=number(avg[0, j]), v2=number(avg[1, j]), delta=number(avg[1, j]-avg[0, j]),
                     ci95=interval(draws[:, j])) for j, k in enumerate(METRICS)}
    return stats, draws[:, METRICS.index("mean_tracking_error")]


def read_results(paths, version):
    runs = {}
    for path in paths:
        doc = json.loads(Path(path).read_text())
        seed = doc.get("training_seed")
        if doc["worker_version"] != version or seed in runs or (seed is None and len(paths) > 1):
            raise ValueError(f"{path}: expected {version} and a unique training_seed (use --training-seed)")
        if not doc["scenarios"]:
            raise ValueError(f"{path}: empty result set")
        rows = {s["scenario_id"]: s for s in doc["scenarios"]}
        if len(rows) != len(doc["scenarios"]):
            raise ValueError(f"{path}: duplicate scenario_id")
        runs[seed] = ({k: v for k, v in doc.items() if k not in ("scenarios", "cells")}, rows)
    return runs


def complete_benchmark_run(meta, rows, required_cells):
    per_cell = defaultdict(list)
    for row in rows.values():
        steps = row.get("steps", meta.get("steps"))
        if isinstance(steps, bool) or not isinstance(steps, int) or steps < 300:
            return False
        per_cell[tuple(row[k] for k in CELL)].append(row)
    if per_cell.keys() != required_cells:
        return False
    required_seeds = set(range(1000, 1020))
    for cell in per_cell.values():
        seeds = {row["seed"] for row in cell}
        hashes = {row["scenario_sha256"] for row in cell}
        if not required_seeds <= seeds or len(seeds) != len(cell) or len(hashes) != len(cell):
            return False
    return True


def compare(args):
    paths = [list(args.v1_results or []), list(args.v2_results or [])]
    if args.compare:
        if len(args.compare) != 2:
            raise ValueError("--compare takes a v1/v2 pair or --v1-results and --v2-results")
        for side, path in zip(paths, args.compare):
            side.insert(0, path)
    left, right = (read_results(p, v) for p, v in zip(paths, ("v1", "v2")))
    if not left or left.keys() != right.keys():
        raise ValueError("v1 and v2 must cover the same training seeds")
    seeds = sorted(left)
    hashes = {meta["manifest_sha256"] for runs in (left, right) for meta, _ in runs.values()}
    if len(hashes) != 1 or not all(hashes):
        raise ValueError("Cannot pair different scenario manifests")
    catalog = {}
    identity_fields = (*STRATA, "scenario_sha256", "seed", "pinned_ratio_coincident")
    complete = True
    for runs in (left, right):
        for meta, rows in runs.values():
            complete &= len(rows) == meta.get("manifest_scenario_count", len(rows))
            for sid, row in rows.items():
                sample = catalog.setdefault(sid, row)
                if not sid or not row["scenario_sha256"] or any(sample[k] != row[k] for k in identity_fields):
                    raise ValueError(f"Scenario identity mismatch for {sid}")
    grouped = defaultdict(list)
    pairs = []
    steps_match = True
    for seed in seeds:
        left_meta, left_rows = left[seed]
        right_meta, right_rows = right[seed]
        for sid in sorted(catalog):
            a, b = left_rows.get(sid), right_rows.get(sid)
            sample = catalog[sid]
            steps_a = None if a is None else a.get("steps", left_meta.get("steps"))
            steps_b = None if b is None else b.get("steps", right_meta.get("steps"))
            if (isinstance(steps_a, bool) or isinstance(steps_b, bool) or
                    not isinstance(steps_a, int) or not isinstance(steps_b, int) or
                    steps_a != steps_b or steps_a < 300):
                steps_match = False
            ok = all(r and r["status"] == "ok" and r["all_finite"] and
                     all(r.get(k) is None or np.isfinite(r[k]) for k in METRICS) for r in (a, b))
            pair = dict(training_seed=seed, scenario_id=sid, v1=a, v2=b, paired_ok=ok)
            if ok:
                pair["delta"] = {k: float(b[k])-float(a[k]) if a[k] is not None and b[k] is not None else None
                                 for k in METRICS}
            pairs.append(pair)
            grouped[tuple(sample[k] for k in CELL)].append(pair)

    rng = np.random.default_rng(42)
    seed_draws = rng.integers(len(seeds), size=(2000, len(seeds)))
    cells, hard_draws = [], []
    hard_cells, small_cells = [], []
    for key, group in sorted(grouped.items()):
        stats, draws = paired_cell(group, seeds, seed_draws, rng)
        rows = [p["v2"] or p["v1"] for p in group if p["v2"] or p["v1"]]
        n_ok = sum(p["paired_ok"] for p in group)
        stable = sum(bool(p["v2"] and p["v2"]["status"] == "ok" and p["v2"]["stable"]) for p in group)
        cell = dict(zip(CELL, key), metrics=stats, n_ok=n_ok, n_total=len(group),
                    actual_pinned=sorted({r["actual_pinned"] for r in rows}),
                    actual_H=sorted({r["actual_H"] for r in rows}),
                    pinned_ratio_coincident=any(r["pinned_ratio_coincident"] for r in rows),
                    stability_rate=stable/len(group), all_finite=n_ok == len(group))
        cells.append(cell)
        err, comm = stats["mean_tracking_error"], stats["mean_comm_rate"]
        if key in ((50, "low", "deep"), (100, "low", "deep")):
            hard_cells.append(cell)
            hard_draws.append(draws)
        if key[0] in (5, 10, 20) and key[1:] == ("normal", "shallow"):
            cell["small_passed"] = bool(n_ok == len(group) and
                                        (err["delta"] <= .05 * err["v1"] + 1e-12 or err["delta"] <= .02 + 1e-12)
                                        and comm["delta"] <= .02 + 1e-12)
            small_cells.append(cell)
    hard = dict(error_reduction=None, error_delta_ci95=None, mean_comm_rate=None, comm_delta=None, passed=False)
    if len(hard_cells) == 2 and all(c["n_ok"] for c in hard_cells):
        def hard_mean(metric, field):
            return float(np.mean([c["metrics"][metric][field] for c in hard_cells]))
        base = hard_mean("mean_tracking_error", "v1")
        err_delta = hard_mean("mean_tracking_error", "delta")
        reduction = -err_delta / base if base > 0 else None
        ci = interval(np.mean(hard_draws, axis=0))
        comm = hard_mean("mean_comm_rate", "v2")
        delta = hard_mean("mean_comm_rate", "delta")
        hard.update(error_reduction=reduction, error_delta=err_delta, error_delta_ci95=ci,
                    mean_comm_rate=comm, comm_delta=delta,
                    passed=bool(reduction is not None and reduction >= .15-1e-12 and ci and ci[1] < 0
                                and comm <= .30+1e-12 and delta <= .02+1e-12
                                and all(c["n_ok"] == c["n_total"] for c in hard_cells)))
    required = set(itertools.product((5, 10, 20, 50, 100), ("normal", "low"), ("shallow", "deep")))
    missing = sorted(required - grouped.keys())
    matrix_complete = complete and steps_match and all(
        rows.keys() == catalog.keys() and complete_benchmark_run(meta, rows, required)
        for runs in (left, right) for meta, rows in runs.values())
    stable = complete and not missing and all(c["all_finite"] and c["stability_rate"] >= .95 for c in cells)
    small = len(small_cells) == 3 and all(c["small_passed"] for c in small_cells)
    failures = [p for p in pairs if not p["paired_ok"]]
    if matrix_complete:
        matrix_note = "满足正式验收矩阵。"
    elif not steps_match:
        matrix_note = "探索性对照，两侧每场有效步数不一致或不足 300 步，不满足正式 20×20×300 步验收矩阵。"
    else:
        matrix_note = ("探索性子集，不满足正式 20×20×300 步验收矩阵。每格须覆盖独立场景 "
                       "seed 1000–1019，两侧每场至少运行 300 步且步数一致。")
    report = dict(manifest_sha256=hashes.pop(), training_seeds=seeds, bootstrap_seed=42, bootstrap_samples=len(seed_draws),
                  inputs=[dict(v1=left[s][0], v2=right[s][0]) for s in seeds], hard=hard, cells=cells,
                  acceptance=dict(hard=hard["passed"], small=small, stability=stable,
                                  complete_manifests=complete, steps_match=steps_match, missing_cells=missing,
                                  complete_benchmark_matrix=matrix_complete,
                                  matrix_note=matrix_note,
                                  passed=bool(matrix_complete and not failures and hard["passed"] and small and stable)),
                  failures=failures, pairs=pairs)
    write_json(args.output, report)
    if args.markdown_out:
        Path(args.markdown_out).write_text(markdown(report), encoding="utf-8")


def markdown(report):
    def fmt(x):
        return "—" if x is None else f"{x:.4f}"

    def ci(bounds):
        return "—" if bounds is None else " / ".join(map(fmt, bounds))

    h = report["hard"]
    gates = report["acceptance"]
    lines = ["# 跨版本配对评测", "", f"综合验收 {'通过' if report['acceptance']['passed'] else '未通过'}",
             "", gates["matrix_note"],
             "", "、".join(label + ("通过" if gates[key] else "未通过") for key, label in
                          (("hard", "困难格"), ("small", "小规模常规格"), ("stability", "稳定性"))),
             "", f"困难格等权误差降幅 {fmt(h['error_reduction'])}，误差差值 95% CI {ci(h['error_delta_ci95'])}，"
             f"通信率 {fmt(h['mean_comm_rate'])}，通信率差值 {fmt(h['comm_delta'])}。",
             "", "| N | pinned | 深度 | 实际 P | 实际 H | pinned_ratio_coincident | 成功/全部 | "
             "v1 误差 | v2 误差 | 差值 [95% CI] | v2 通信率 | 累计奖励 | 收敛率 | 惩罚后步数 | 稳定率 |",
             "|" + "---|" * 15]
    for c in report["cells"]:
        m = c["metrics"]
        err = m["mean_tracking_error"]
        lines.append("| " + " | ".join(map(str, [*(c[k] for k in CELL), c["actual_pinned"], c["actual_H"],
                     c["pinned_ratio_coincident"], f"{c['n_ok']}/{c['n_total']}", fmt(err["v1"]), fmt(err["v2"]),
                     f"{fmt(err['delta'])} [{ci(err['ci95'])}]", *(fmt(m[k]["v2"]) for k in
                     ("mean_comm_rate", "mean_reward", "converged", "penalized_convergence_step")),
                     fmt(c["stability_rate"])])) + " |")
    if report["failures"]:
        lines += ["", "## 失败场景", "", "| 训练 seed | 场景 | v1 | v2 |", "|---|---|---|---|"]
        for row in report["failures"]:
            labels = []
            for side in ("v1", "v2"):
                r = row[side] or {"status": "missing"}
                labels.append((r["status"] + " " + r.get("error", "")).replace("|", "/").replace("\n", " "))
            lines.append(f"| {row['training_seed']} | {row['scenario_id']} | {labels[0]} | {labels[1]} |")
    return "\n".join(lines) + "\n"


def positive(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return value


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--generate-manifest", metavar="JSON")
    mode.add_argument("--evaluate", action="store_true")
    mode.add_argument("--parity-check", action="store_true")
    mode.add_argument("--compare", nargs="*", metavar="RESULT")
    mode.add_argument("--worker", choices=("evaluate", "trace"), help=argparse.SUPPRESS)
    ap.add_argument("--scales", type=positive, nargs="+", default=[5, 10, 20, 50, 100])
    ap.add_argument("--pinned-modes", choices=("normal", "low"), nargs="+", default=["normal", "low"])
    ap.add_argument("--depth-modes", choices=("shallow", "deep"), nargs="+", default=["shallow", "deep"])
    ap.add_argument("--seed-start", type=int, default=1000)
    ap.add_argument("--num-seeds", type=positive, default=20)
    ap.add_argument("--steps", type=positive, default=300)
    ap.add_argument("--scenario-manifest")
    ap.add_argument("--worker-version", choices=("v1", "v2"), default="v2")
    ap.add_argument("--repo-path", default=ROOT)
    ap.add_argument("--v2-repo", default=ROOT)
    ap.add_argument("--v1-repo")
    ap.add_argument("--device", choices=("auto", "cpu", "cuda"))
    ap.add_argument("--top-k", type=positive)
    ap.add_argument("--training-seed", type=int, help="多权重配对时为每份结果提供训练 seed")
    ap.add_argument("--checkpoint")
    ap.add_argument("--max-scenarios", type=positive)
    ap.add_argument("--output")
    ap.add_argument("--markdown-out")
    ap.add_argument("--v1-results", nargs="+", action="extend")
    ap.add_argument("--v2-results", nargs="+", action="extend")
    args = ap.parse_args(argv)
    if args.generate_manifest:
        if any(len(set(x)) != len(x) for x in (args.scales, args.pinned_modes, args.depth_modes)):
            ap.error("matrix dimensions must be unique")
        generate_manifest(args)
    elif args.worker:
        stream = sys.stdout.buffer
        with open(os.devnull, "w") as quiet, redirect_stdout(quiet):
            return run_worker(args, stream)
    elif args.evaluate or args.parity_check:
        if not args.scenario_manifest:
            ap.error("--scenario-manifest is required")
        if args.evaluate:
            if not args.output or not args.checkpoint:
                ap.error("--evaluate requires --output and --checkpoint")
            read_manifest(args.scenario_manifest)
            return subprocess.run(worker_command(args, args.worker_version, args.repo_path, "evaluate")).returncode
        if not args.v1_repo:
            ap.error("--parity-check requires --v1-repo")
        return parity(args)
    elif args.compare is not None or args.v1_results or args.v2_results:
        if not args.output:
            ap.error("comparison requires --output")
        compare(args)
    else:
        ap.error("select a benchmark operation")
    return 0


if __name__ == "__main__":
    sys.exit(main())
