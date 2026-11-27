"""短跑瓶颈分析脚本。

在一个很短的 roll-out/update 过程中，分别统计以下阶段的耗时占比：
- actor（选动作）
- env（环境 step）
- store（写入 buffer）
- update（参数更新）

该脚本主要用于粗定位“慢在哪里”，不用于严格基准测试。

用法:
    python profile_bottleneck.py --steps 300 --warmup 10
    python profile_bottleneck.py --algo mappo --device cuda --output mappo.json
    python profile_bottleneck.py --algo mappo --scale-switch-cycles 10 --output switch.json
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import ExitStack, contextmanager
import json
import os
from pathlib import Path
import time
from unittest.mock import patch

import numpy as np
import torch


def parser():
    ap = argparse.ArgumentParser(description="SAC 短跑或真实 MAPPO 整局性能剖析")
    ap.add_argument("--algo", choices=("sac", "mappo"), default="sac")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--warmup", type=int, default=10, help="SAC 热身步数")
    ap.add_argument("--num-followers", type=int, default=20)
    ap.add_argument("--num-envs", type=int, default=48)
    ap.add_argument("--warmup-episodes", type=int, default=1)
    ap.add_argument("--measure-episodes", type=int, default=3)
    ap.add_argument("--depth-mode", choices=("random", "deep"), default="random")
    ap.add_argument("--pinned-mode", choices=("normal", "low"), default="normal")
    ap.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    ap.add_argument("--actor-microbatch-rows", type=int, default=4096)
    ap.add_argument("--scale-switch-cycles", type=int, default=0)
    ap.add_argument("--output", help="MAPPO 的 JSON 结果路径")
    return ap


def run_sac(args):
    from mas_cc import config as C
    from mas_cc.agent import CTDESACAgent
    from mas_cc.environment import BatchedModelFreeEnv
    from mas_cc.topology import CommunicationTopology

    top = CommunicationTopology(C.NUM_FOLLOWERS, num_pinned=C.NUM_PINNED)
    env = BatchedModelFreeEnv(top, num_envs=C.NUM_PARALLEL_ENVS)
    ag = CTDESACAgent(top)

    is_cuda = C.DEVICE.type == "cuda"

    def sync() -> None:
        """在 CUDA 设备上同步，避免异步执行影响计时。"""

        if is_cuda:
            torch.cuda.synchronize()

    def pct(x: float, tot: float) -> float:
        """计算占比（百分数）。"""

        return 0.0 if tot <= 0 else 100.0 * x / tot

    # warmup
    s = env.reset()
    g = env.get_global_state()
    for _ in range(max(0, int(args.warmup))):
        a = ag.select_action(s, deterministic=False)
        ns, r, d, _ = env.step(a)
        ng = env.get_global_state()
        ag.store_transitions_batch(s, g, a, r, ns, ng, d)
        s, g = ns, ng

    # timed run
    N = max(1, int(args.steps))
    at = et = st = ut = 0.0
    uc = 0

    s = env.reset()
    g = env.get_global_state()

    sync()
    t_start = time.perf_counter()

    for k in range(1, N + 1):
        sync()
        t0 = time.perf_counter()
        a = ag.select_action(s, deterministic=False)
        sync()
        at += time.perf_counter() - t0

        sync()
        t0 = time.perf_counter()
        ns, r, d, _ = env.step(a)
        ng = env.get_global_state()
        sync()
        et += time.perf_counter() - t0

        sync()
        t0 = time.perf_counter()
        ag.store_transitions_batch(s, g, a, r, ns, ng, d)
        sync()
        st += time.perf_counter() - t0

        s, g = ns, ng

        if k % C.UPDATE_FREQUENCY == 0 and ag.buffer.is_ready(C.BATCH_SIZE):
            sync()
            t0 = time.perf_counter()
            ag.update(C.BATCH_SIZE, C.GRADIENT_STEPS)
            sync()
            ut += time.perf_counter() - t0
            uc += 1

    sync()
    tot = time.perf_counter() - t_start
    other = max(0.0, tot - (at + et + st + ut))

    print("===== Bottleneck profiling (short run) =====")
    print(
        f"device={C.DEVICE} envs={C.NUM_PARALLEL_ENVS} steps={N} batch={C.BATCH_SIZE} "
        f"upd_freq={C.UPDATE_FREQUENCY} grad_steps={C.GRADIENT_STEPS}"
    )
    print(f"total_s={tot:.3f}")
    print(f"actor_s={at:.3f} actor_pct={pct(at, tot):.1f} actor_ms_per_step={at / N * 1000:.2f}")
    print(f"env_s={et:.3f} env_pct={pct(et, tot):.1f} env_ms_per_step={et / N * 1000:.2f}")
    print(f"store_s={st:.3f} store_pct={pct(st, tot):.1f} store_ms_per_step={st / N * 1000:.2f}")
    print(
        f"update_s={ut:.3f} update_pct={pct(ut, tot):.1f} update_calls={uc} "
        f"update_ms_per_call={ut / max(1, uc) * 1000:.2f}"
    )
    print(f"other_s={other:.3f} other_pct={pct(other, tot):.1f}")
    print("===========================================")


class Timings:
    """CUDA 事件成对记录；每局末统一同步，避免在每个阶段打断队列。"""

    def __init__(self, cuda):
        self.cuda = cuda
        self.events = []
        self.ms = defaultdict(list)

    @contextmanager
    def phase(self, name):
        if self.cuda:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            yield
            end.record()
            self.events.append((name, start, end))
        else:
            start = time.perf_counter()
            yield
            self.ms[name].append((time.perf_counter() - start) * 1000)

    def timed(self, name, method):
        def call(*args, **kwargs):
            with self.phase(name):
                return method(*args, **kwargs)
        return call

    def finish(self):
        if self.cuda:
            torch.cuda.synchronize()
        end_time = time.perf_counter()
        if self.cuda:
            for name, start, end in self.events:
                self.ms[name].append(start.elapsed_time(end))
            self.events.clear()
        return end_time

    def summary(self):
        return {k: dict(count=len(v), p50=float(np.percentile(v, 50)), p95=float(np.percentile(v, 95)))
                for k, v in self.ms.items() if v}


def build_env(args, n):
    from mas_cc.environment import BatchedModelFreeEnv
    from mas_cc.topology import CommunicationTopology
    top = CommunicationTopology(n, pinned_mode=args.pinned_mode, depth_mode=args.depth_mode)
    return BatchedModelFreeEnv(top, num_envs=args.num_envs, max_steps=args.steps,
                               auto_reset=False, randomize_topology_on_reset=False)


def episode(agent, env, timings):
    """计时器装在真实采集器的方法边界，所有权检查、bootstrap 和优化器保持原调用顺序。"""
    timings.finish()
    start = time.perf_counter()
    with timings.phase("hot_reset"):
        env.reset()
    env.randomize_topology = True
    session = agent.start_episode(env)
    lengths, updates = [], []
    with ExitStack() as stack:
        for obj, name, label in ((agent, "_act_for_buffer", "act"), (env, "step", "env_step"),
                                 (session._buffer, "add", "store"), (session, "flush_update", "update")):
            stack.enter_context(patch.object(obj, name, timings.timed(label, getattr(obj, name))))
        for t in range(1, env.max_steps + 1):
            session.step_collect()
            final = t == env.max_steps
            if final or session.buffer.is_full():
                length = len(session.buffer)
                losses = session.flush_update(is_episode_end=final)
                lengths.append(length)
                updates.append(losses["update_minibatches"])
    session.close()
    elapsed = timings.finish() - start
    if sum(lengths) != env.max_steps or min(updates) < 1:
        raise RuntimeError(f"Incomplete rollout update: lengths={lengths}, optimized={updates}")
    return dict(rollout_lengths=lengths, update_minibatches=updates, elapsed_sec=elapsed)


def memory(cuda):
    if not cuda:
        return dict(max_memory_allocated_mb=None, max_memory_reserved_mb=None,
                    total_gpu_memory_mb=None, reserved_pct=None)
    mib = 1024**2
    total = torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory
    reserved = torch.cuda.max_memory_reserved()
    return dict(max_memory_allocated_mb=torch.cuda.max_memory_allocated()/mib,
                max_memory_reserved_mb=reserved/mib, total_gpu_memory_mb=total/mib,
                reserved_pct=100*reserved/total)


def scale_switch(args, agent, cuda, report):
    env_by_n = {}
    cycles = []
    result = report["scale_switch"] = dict(cycles=cycles, env_build_count=0, live_allocated_growth_ratio=None)
    sizes = (10, 20, 30, 10)
    for i in range(args.scale_switch_cycles):
        record = dict(cycle=i+1, scales=[], cache_size=len(env_by_n))
        cycles.append(record)
        for n in sizes:
            report["active_load"] = dict(phase="scale_switch", cycle=i+1, N=n)
            if n not in env_by_n:
                env_by_n[n] = build_env(args, n)
                result["env_build_count"] += 1
            episode(agent, env_by_n[n], Timings(cuda))
            record["scales"].append(n)
            record["cache_size"] = len(env_by_n)
        live = torch.cuda.memory_allocated() if cuda else None
        record["live_allocated_mb"] = live / 1024**2 if cuda else None
        if cuda:
            base = cycles[0]["live_allocated_mb"]
            record["growth_ratio"] = record["live_allocated_mb"]/base - 1
    result["cache_ok"] = result["env_build_count"] == 3 and all(c["cache_size"] == 3 for c in cycles)
    if cuda:
        result["live_allocated_growth_ratio"] = max(c["growth_ratio"] for c in cycles)
        result["growth_within_5pct"] = result["live_allocated_growth_ratio"] <= .05


def profile_mappo(args, report):
    from mas_cc import config as c
    from mas_cc.agent import CTDEMAPPOAgent
    if args.device != "auto" and args.device != str(c.DEVICE):
        raise RuntimeError(f"Requested device {args.device} unavailable")
    cuda = c.DEVICE.type == "cuda"
    if not cuda and args.num_envs <= 2:
        torch.set_num_threads(1)
    c.set_seed()
    report.update(device=str(c.DEVICE), torch_version=str(torch.__version__), cuda_version=torch.version.cuda,
                  top_k=c.MAX_NEIGHBORS, episodes=[], active_load=dict(phase="cold_build", N=args.num_followers))
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    timings = Timings(cuda)
    with timings.phase("cold_build"):
        env = build_env(args, args.num_followers)
    timings.finish()
    agent = CTDEMAPPOAgent(env.topology, actor_microbatch_max_rows=args.actor_microbatch_rows)
    for i in range(args.warmup_episodes):
        report["active_load"] = dict(phase="warmup", episode=i+1, N=args.num_followers)
        episode(agent, env, Timings(cuda))
    steps = dict(actor=0, value=0)

    def counter(key):
        def increment(*_):
            steps[key] += 1
        return increment

    with ExitStack() as stack:
        for name in steps:
            opt = getattr(agent, name+"_optimizer")
            stack.callback(opt.register_step_post_hook(counter(name)).remove)
        for i in range(args.measure_episodes):
            report["active_load"] = dict(phase="measure", episode=i+1, N=args.num_followers)
            report["episodes"].append(episode(agent, env, timings))
    report.update(timings_ms=timings.summary(), optimizer_steps=steps,
                  seconds_per_episode=float(np.mean([e["elapsed_sec"] for e in report["episodes"]])),
                  **memory(cuda))
    expected = sum(sum(e["update_minibatches"]) for e in report["episodes"])
    if steps["actor"] != expected or steps["value"] != expected:
        raise RuntimeError("Observed Adam step counts disagree with flushed minibatches")
    if args.scale_switch_cycles:
        del env
        scale_switch(args, agent, cuda, report)
    report.pop("active_load")
    report["status"] = "ok"


def main(argv=None):
    ap = parser()
    args = ap.parse_args(argv)
    if args.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    if args.algo == "sac":
        run_sac(args)
        return 0
    if min(args.steps, args.num_followers, args.num_envs, args.actor_microbatch_rows, args.measure_episodes) < 1:
        ap.error("MAPPO steps, dimensions and measurement episodes must be positive")
    if min(args.warmup_episodes, args.scale_switch_cycles) < 0:
        ap.error("Warmup episodes and switch cycles cannot be negative")
    report = {k: v for k, v in vars(args).items() if k not in ("output", "warmup")}
    try:
        profile_mappo(args, report)
    except (RuntimeError, MemoryError, ValueError) as exc:
        oom = isinstance(exc, (MemoryError, torch.cuda.OutOfMemoryError)) or "out of memory" in str(exc).lower()
        report.update(status="oom" if oom else "error", error=str(exc))
    data = json.dumps(report, indent=2, allow_nan=False)
    if args.output:
        Path(args.output).write_text(data+"\n", encoding="utf-8")
    else:
        print(data)
    return int(report["status"] != "ok")


if __name__ == "__main__":
    raise SystemExit(main())
