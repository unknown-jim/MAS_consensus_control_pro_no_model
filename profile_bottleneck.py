import argparse
import time

import torch

import config as C
from agent import CTDESACAgent
from environment import BatchedModelFreeEnv
from topology import CommunicationTopology


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--warmup", type=int, default=10)
    args = ap.parse_args()

    top = CommunicationTopology(C.NUM_FOLLOWERS, num_pinned=C.NUM_PINNED)
    env = BatchedModelFreeEnv(top, num_envs=C.NUM_PARALLEL_ENVS)
    ag = CTDESACAgent(top)

    is_cuda = C.DEVICE.type == "cuda"

    def sync():
        if is_cuda:
            torch.cuda.synchronize()

    def pct(x, tot):
        return 0.0 if tot <= 0 else 100.0 * x / tot

    # warmup
    s = env.reset()
    g = env.get_global_state()
    for _ in range(max(0, args.warmup)):
        a = ag.select_action(s, deterministic=False)
        ns, r, d, _ = env.step(a)
        ng = env.get_global_state()
        ag.store_transitions_batch(s, g, a, r, ns, ng, d)
        s, g = ns, ng

    # timed run
    N = max(1, args.steps)
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
    print(f"device={C.DEVICE} envs={C.NUM_PARALLEL_ENVS} steps={N} batch={C.BATCH_SIZE} upd_freq={C.UPDATE_FREQUENCY} grad_steps={C.GRADIENT_STEPS}")
    print(f"total_s={tot:.3f}")
    print(f"actor_s={at:.3f} actor_pct={pct(at, tot):.1f} actor_ms_per_step={at / N * 1000:.2f}")
    print(f"env_s={et:.3f} env_pct={pct(et, tot):.1f} env_ms_per_step={et / N * 1000:.2f}")
    print(f"store_s={st:.3f} store_pct={pct(st, tot):.1f} store_ms_per_step={st / N * 1000:.2f}")
    print(f"update_s={ut:.3f} update_pct={pct(ut, tot):.1f} update_calls={uc} update_ms_per_call={ut / max(1, uc) * 1000:.2f}")
    print(f"other_s={other:.3f} other_pct={pct(other, tot):.1f}")
    print("===========================================")


if __name__ == "__main__":
    main()
