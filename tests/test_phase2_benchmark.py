"""通过 CLI 观测真实的跨版本轨迹与训练更新。"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import networkx as nx
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
V1 = Path(os.getenv("MAS_V1_REPO", ROOT.parent / "MAS_consensus_control_pro_no_model"))
PYTHON = sys.executable


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


class BenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)
        self.manifest = self.dir / "manifest.json"
        self.out = self.dir / "result.json"

    def run_cli(self, *args, code=0):
        run = subprocess.run([PYTHON, "-B", *map(str, args)], cwd=ROOT,
                             capture_output=True, text=True, env=dict(os.environ,
                             PYTHONDONTWRITEBYTECODE="1", MPLCONFIGDIR=os.environ.get("MPLCONFIGDIR", str(self.dir))))
        self.assertEqual(run.returncode, code, run.stderr + run.stdout)
        return run.stdout

    def generate(self, *args):
        self.run_cli("evaluate_scaling.py", "--generate-manifest", self.manifest,
                     "--scales", "5", "--num-seeds", "1", *args)
        return json.loads(self.manifest.read_text())

    def test_manifest(self):
        manifest = self.generate()
        self.assertEqual(manifest, self.generate())
        self.assertEqual(manifest.pop("manifest_sha256"), digest(manifest))
        self.assertEqual(len(manifest["scenarios"]), 4)
        for s in manifest["scenarios"]:
            self.assertEqual(s.pop("scenario_sha256"), digest(s))
            adj = np.array(s["adj_matrix"])
            g = nx.from_numpy_array(adj.T, create_using=nx.DiGraph)
            depth = nx.single_source_shortest_path_length(g, 0)
            stats = s["bfs_stats"]
            self.assertEqual(stats["node_depths"], [depth[i] for i in range(6)])
            self.assertEqual(stats["layer_counts"], np.bincount(list(depth.values())).tolist())
            self.assertEqual(stats["reachable_count"], 6)
            self.assertEqual(stats["max_depth"], max(depth.values()))
            self.assertEqual(stats["mean_in_degree"], adj[1:].sum(axis=1).mean())
            self.assertEqual(stats["max_in_degree"], adj[1:].sum(axis=1).max())
            self.assertEqual(np.flatnonzero(adj[:, 0]).tolist(), s["pinned_ids"])
            self.assertTrue(s["pinned_ratio_coincident"])
            a, w, p = (s["traj_params"][k] for k in ("amplitude", "omega", "phase"))
            t = np.arange(301) * manifest["dt"]
            angle = w * t + p + (np.pi / 2 if s["traj_type"] == "cosine" else 0)
            x, v = a * np.sin(angle), a * w * np.cos(angle)
            self.assertAlmostEqual(s["init_pos"][0], x[0], places=5)
            self.assertAlmostEqual(s["init_vel"][0], v[0], places=5)
            self.assertEqual(s["traj_in_bounds"], bool(np.all(abs(x) <= manifest["pos_limit"])
                                                     and np.all(abs(v) <= manifest["vel_limit"])))

    def test_trajectory_bound_between_frames(self):
        from unittest.mock import patch
        import evaluate_scaling
        from mas_cc import config
        # 两端位置接近零，中间超过位置限幅。越界要记录，不能靠裁剪或只验首帧掩盖。
        with patch.multiple(config, LEADER_TRAJECTORY_TYPES=["sine"], LEADER_AMPLITUDE_RANGE=(2., 2.),
                            LEADER_OMEGA_RANGE=(np.pi, np.pi), LEADER_PHASE_RANGE=(0., 0.),
                            POS_LIMIT=1., VEL_LIMIT=10., DT=1.):
            evaluate_scaling.main(["--generate-manifest", str(self.manifest), "--scales", "5",
                                   "--num-seeds", "1", "--steps", "1"])
        self.assertTrue(all(not s["traj_in_bounds"] for s in json.loads(self.manifest.read_text())["scenarios"]))

    def test_manifest_rejects_tampering(self):
        from evaluate_scaling import read_manifest
        m = self.generate("--pinned-modes", "low", "--depth-modes", "deep")
        m["scenarios"][0]["init_pos"][2] += 1.
        self.manifest.write_text(json.dumps(m))
        with self.assertRaisesRegex(ValueError, "manifest_sha256"):
            read_manifest(self.manifest)
        m.pop("manifest_sha256")
        m["manifest_sha256"] = digest(m)
        self.manifest.write_text(json.dumps(m))
        with self.assertRaisesRegex(ValueError, "scenario_sha256"):
            read_manifest(self.manifest)

    def test_frozen_graph_integrity_before_workers(self):
        from unittest.mock import patch
        import evaluate_scaling as bench
        from mas_cc.topology import CommunicationTopology

        chain = [[0, 0, 0], [1, 0, 0], [0, 1, 0]]
        base = dict(N=2, leader_id=0, pinned_ids=[1], adj_matrix=chain,
                    bfs_stats=dict(node_depths=[0, 1, 2], max_depth=2), scenario_id="chain")
        self.assertEqual(bench.FrozenGraph(base, "cpu").get_receivers(1), [2])
        disconnected = dict(base, adj_matrix=chain[:2] + [[0, 0, 0]])
        with self.assertRaises(ValueError):
            CommunicationTopology.from_frozen(disconnected["adj_matrix"], [1], device="cpu")
        for change in [disconnected, dict(base, pinned_ids=[2]), dict(base, pinned_ids=[True]),
                       dict(base, pinned_ids=[1.0]),
                       dict(base, adj_matrix=chain[:2] + [[0, 1, 1]]),
                       dict(base, bfs_stats=dict(node_depths=[0, 2, 1], max_depth=2)),
                       dict(base, bfs_stats=dict(node_depths=[0, 1, 2], max_depth=1))]:
            with self.subTest(change=change):
                row = dict(change, scenario_sha256=digest(change))
                manifest = dict(schema_version=1, steps=300, scenarios=[row])
                manifest["manifest_sha256"] = digest(manifest)
                self.manifest.write_text(json.dumps(manifest))
                with self.assertRaises(ValueError):
                    bench.FrozenGraph(change, "cpu")
                with self.assertRaises(ValueError):
                    bench.read_manifest(self.manifest)
                with patch.object(bench, "worker_command") as command:
                    for v in ("v1", "v2"):
                        with self.assertRaises(ValueError):
                            bench.main(["--evaluate", "--worker-version", v, "--scenario-manifest",
                                        str(self.manifest), "--checkpoint", "unused", "--output", str(self.out)])
                    command.assert_not_called()

    def test_python310_checkpoint_digest(self):
        self.generate("--steps", "2", "--depth-modes", "deep", "--pinned-modes", "low")
        checkpoint = self.dir / "checkpoint.bin"
        # 即使权重内容无法加载，仍能核对实际 worker 读取的多块 SHA。
        data = bytes(range(256)) * 10000
        checkpoint.write_bytes(data)
        worker = """
import hashlib, sys
import evaluate_scaling
if hasattr(hashlib, 'file_digest'):
    del hashlib.file_digest
raise SystemExit(evaluate_scaling.main(sys.argv[1:]))
"""
        self.run_cli("-c", worker, "--worker", "evaluate", "--scenario-manifest", self.manifest,
                     "--checkpoint", checkpoint, "--output", self.out, "--device", "cpu", code=1)
        result = json.loads(self.out.read_text())
        self.assertEqual(result["checkpoint_sha256"], hashlib.sha256(data).hexdigest())

    def test_metric_denominators_and_raw_age(self):
        from evaluate_scaling import metrics
        # 第 0 帧只用于门禁。两名 follower 的第 1 个动作后误差分别为 0、2。
        steps = 60
        pos = np.zeros((steps+1, 3))
        pos[:2, 2] = 2
        vel = np.zeros_like(pos)
        vel[1:, 2] = 10
        seq = np.zeros_like(pos, dtype=int)
        seq[:4, 2] = -1
        trace = dict(positions=pos, velocities=vel, leader_est_seq=seq,
                     reward=np.full(steps+1, 2.), triggered=np.zeros((steps+1, 2), dtype=bool),
                     all_finite=np.ones(steps+1, dtype=bool))
        trace["triggered"][::2, 0] = True
        s = dict(N=2, pinned_ids=[1], bfs_stats=dict(node_depths=[0, 1, 2], max_depth=2))
        r = metrics(trace, s)
        self.assertAlmostEqual(r["mean_tracking_error"], .05 + 1/60)
        self.assertAlmostEqual(r["last50_error"], .05)
        self.assertAlmostEqual(r["deepest_layer_error"], .1 + 2/60)
        self.assertEqual(r["convergence_step"], 1)
        self.assertTrue(r["stable"])
        self.assertEqual(r["mean_reward"], 120)
        self.assertEqual(r["mean_comm_rate"], .25)
        self.assertAlmostEqual(r["valid_est_coverage"], 117/120)
        self.assertAlmostEqual(r["high_age_ratio"], 100/117)
        self.assertAlmostEqual(r["est_age_p95"], np.percentile([*range(1, 61), *range(4, 61)], 95))
        self.assertEqual(r["first_est_step_mean"], 3)
        seq[:] = -1
        r = metrics(trace, s)
        for k in ("est_age_p95", "high_age_ratio", "first_est_step_mean"):
            self.assertIsNone(r[k])

    def test_real_parity_and_shifted_initial_position(self):
        m = self.generate("--pinned-modes", "normal")
        flags = ["--parity-check", "--scenario-manifest", self.manifest,
                 "--v1-repo", V1, "--v2-repo", ROOT, "--output", self.out]
        self.run_cli("evaluate_scaling.py", *flags)
        report = json.loads(self.out.read_text())
        self.assertTrue(report["passed"])
        self.assertEqual(len(report["scenarios"]), 2)
        self.assertEqual(report["triggered"] + report["not_triggered"], 300 * 5 * 2)
        for s in report["scenarios"]:
            self.assertEqual(s["frames"], 301)
            self.assertGreater(min(s["triggered"], s["not_triggered"]), 0)
        # 只改变送给 v2 子进程的场景，保留两份输入各自合法的内容摘要。
        altered = self.dir / "shifted.json"
        scenario = m["scenarios"][0]
        scenario["init_pos"][2] += .25
        del scenario["scenario_sha256"]
        scenario["scenario_sha256"] = digest(scenario)
        del m["manifest_sha256"]
        m["manifest_sha256"] = digest(m)
        altered.write_text(json.dumps(m))
        # 首帧已经有差异。即使 v2 的下一步不能执行，仍须报告第 0 帧的读数。
        worker = """
import sys
import evaluate_scaling as benchmark
original = benchmark.cold_start
def cold(*args):
    env, state = original(*args)
    def unavailable(*args, **kwargs):
        raise RuntimeError("unavailable after the divergent initial frame")
    env.step = unavailable
    return env, state
benchmark.cold_start = cold
raise SystemExit(benchmark.main())
"""
        harness = """
import sys
import evaluate_scaling as benchmark
original = benchmark.worker_command
def shifted(args, version, repo, mode):
    cmd = original(args, version, repo, mode)
    if version == 'v2':
        cmd[cmd.index('--scenario-manifest') + 1] = sys.argv[1]
        cmd[2:3] = ['-c', sys.argv[2]]
    return cmd
benchmark.worker_command = shifted
raise SystemExit(benchmark.main(sys.argv[3:]))
"""
        self.run_cli("-c", harness, altered, worker, *flags, code=1)
        diff = json.loads(self.out.read_text())["first_divergence"]
        self.assertEqual((diff["scenario_id"], diff["step"], diff["field"], diff["node_idx"]),
                         (scenario["scenario_id"], 0, "positions", 2))
        self.assertAlmostEqual(diff["v2_val"] - diff["v1_val"], .25, places=5)
        print("300-step parity:", report["scenarios"], "negative control:", diff)

    def test_checkpoint(self):
        from mas_cc.agent import CTDEMAPPOAgent
        from mas_cc.topology import CommunicationTopology
        m = self.generate("--steps", "8", "--pinned-modes", "low", "--scales", "5", "10")
        ckpt = self.dir / "actor.pt"
        CTDEMAPPOAgent(CommunicationTopology(5)).save_actor(ckpt)
        flags = ["evaluate_scaling.py", "--evaluate", "--scenario-manifest", self.manifest,
                 "--worker-version", "v2", "--repo-path", ROOT, "--device", "cpu",
                 "--checkpoint", ckpt, "--output", self.out]
        self.run_cli(*flags)
        result = json.loads(self.out.read_text())
        self.assertEqual(result["checkpoint_sha256"], hashlib.sha256(ckpt.read_bytes()).hexdigest())
        self.assertEqual(result["manifest_sha256"], m["manifest_sha256"])
        self.assertEqual(len(result["scenarios"]), 4)
        for row in result["scenarios"]:
            self.assertEqual(row["status"], "ok", row)
            self.assertIsNone(row["convergence_step"])
            self.assertFalse(row["converged"])
            self.assertEqual(row["penalized_convergence_step"], 9)
            self.assertEqual(row["mean_tracking_error"], row["last50_error"])
            self.assertGreater(row["deepest_layer_error"], 0)
            self.assertTrue(0 < row["valid_est_coverage"] <= 1)
            self.assertGreaterEqual(row["est_age_p95"], 0)
            self.assertAlmostEqual(row["mean_reward"], 8 * row["reward_per_step"])
        ckpt.write_bytes(b"bad model")
        self.run_cli(*flags, code=1)
        self.assertEqual([r["status"] for r in json.loads(self.out.read_text())["scenarios"]],
                         ["error"] * 4)

    def test_v1_actor_loader(self):
        ckpt = self.dir / "v1.pt"
        self.generate("--steps", "3", "--depth-modes", "shallow", "--pinned-modes", "normal")
        # 旧 actor 也在独立解释器构造，测试进程不混导两个 mas_cc。
        self.run_cli("-c", """
import sys, torch
sys.path.insert(0, sys.argv[1])
from mas_cc.networks import DecentralizedActor
torch.save(DecentralizedActor().state_dict(), sys.argv[2])
""", V1, ckpt)
        self.run_cli("evaluate_scaling.py", "--evaluate", "--scenario-manifest", self.manifest,
                     "--checkpoint", ckpt, "--worker-version", "v1", "--repo-path", V1,
                     "--output", self.out, "--device", "cpu", "--training-seed", "37")
        result = json.loads(self.out.read_text())
        self.assertEqual(result["repo_sha"], "4115d00412d14cc2600235aa0e99af5e90f131f9")
        self.assertEqual((result["training_seed"], result["obs_version"]), (37, 1))
        self.assertEqual(result["scenarios"][0]["status"], "ok")

    def test_compare(self):
        from evaluate_scaling import METRICS
        # 训练 seed 和文件顺序独立。大格样本不均衡，验收须给每格相同权重。
        paths = {"v1": [], "v2": []}
        for version in paths:
            for train_seed in [11, 23]:
                rows = []
                for n in [5, 10, 20, 50, 100]:
                    hard = n >= 50
                    for i in range(22 if n == 50 else 20):
                        err = (1 if n == 50 else 3) if hard else .2
                        if version == "v2":
                            err *= (.9 if n == 50 else .8) if hard else 1.075
                        for pin in ("normal", "low"):
                            for depth in ("shallow", "deep"):
                                rows.append(dict(
                                    dict.fromkeys(METRICS, 1.), N=n, seed=1000+i, status="ok",
                                    scenario_id=f"{n}-{pin}-{depth}-{i}", scenario_sha256=f"h{n}-{pin}-{depth}-{i}",
                                    pinned_ratio_coincident=n == 5, pinned_mode=pin, depth_mode=depth,
                                    actual_pinned=1, actual_H=4 if depth == "deep" else 2,
                                    stable=True, all_finite=True, converged=True,
                                    mean_tracking_error=err, mean_comm_rate=.21))
                path = self.dir / f"{version}-{train_seed}.json"
                path.write_text(json.dumps(dict(worker_version=version, training_seed=train_seed,
                                                manifest_sha256="same", steps=300, scenarios=rows)))
                paths[version].append(path)
        md = self.dir / "report.md"
        flags = ["evaluate_scaling.py", "--compare", "--v1-results", *paths["v1"],
                 "--v2-results", *reversed(paths["v2"]), "--output", self.out, "--markdown-out", md]
        self.run_cli(*flags)
        result = json.loads(self.out.read_text())
        self.assertTrue(result["acceptance"]["passed"], result["acceptance"])
        self.assertAlmostEqual(result["hard"]["error_reduction"], .175)
        self.assertTrue(np.allclose(result["hard"]["error_delta_ci95"], [-.35, -.35]))
        self.run_cli(*flags)
        self.assertEqual(json.loads(self.out.read_text()), result)
        # 每次只改变覆盖条件。所有配对仍完整，性能门槛保持通过。
        for case in ("one_per_cell", "short_row", "short_run", "unknown_steps", "wrong_seeds", "duplicate_seed"):
            with self.subTest(case=case):
                variant = []
                for path in (paths["v1"][0], paths["v2"][0]):
                    doc = json.loads(path.read_text())
                    rows = doc["scenarios"]
                    if case == "one_per_cell":
                        doc["scenarios"] = [r for r in rows if r["seed"] == 1000]
                    elif case == "short_row" and doc["worker_version"] == "v2":
                        rows[0]["steps"] = 299
                    elif case == "short_run":
                        doc["steps"] = 299
                    elif case == "unknown_steps":
                        del doc["steps"]
                    elif case == "wrong_seeds":
                        rows[0]["seed"] = 999
                    elif case == "duplicate_seed":
                        rows[0]["seed"] = 1001
                    doc["manifest_scenario_count"] = len(doc["scenarios"])
                    dst = self.dir / ("variant-" + path.name)
                    dst.write_text(json.dumps(doc))
                    variant.append(dst)
                self.run_cli("evaluate_scaling.py", "--compare", *variant, "--output", self.out, "--markdown-out", md)
                report = json.loads(self.out.read_text())
                self.assertFalse(report["acceptance"]["passed"])
                self.assertFalse(report["acceptance"]["complete_benchmark_matrix"])
                self.assertIn("探索性", md.read_text())
                self.assertTrue(report["hard"]["passed"])
                self.assertEqual(len(report["cells"]), 20)
        self.assertTrue(result["acceptance"]["complete_benchmark_matrix"])
        doc = json.loads(paths["v2"][0].read_text())
        doc["scenarios"][0].update(status="oom", error="out of memory")
        paths["v2"][0].write_text(json.dumps(doc))
        self.run_cli(*flags)
        failed = json.loads(self.out.read_text())
        self.assertFalse(failed["acceptance"]["passed"])
        self.assertEqual(failed["failures"][0]["v2"]["status"], "oom")
        self.assertIn("oom", md.read_text())
        self.assertIn("pinned_ratio_coincident", md.read_text())
        self.run_cli("evaluate_scaling.py", "--compare", paths["v1"][0], paths["v2"][0],
                     "--output", self.out)
        self.assertFalse(json.loads(self.out.read_text())["acceptance"]["passed"])

        # 每个反例只改一条指标，排除“失败仅因为存在 OOM 行”的假阳性。
        pair_paths = [paths[v][1] for v in ("v1", "v2")]
        originals = [p.read_text() for p in pair_paths]
        for metric, target, value, gate in (
            ("mean_tracking_error", (5, "normal", "shallow"), .23, "small"),
            ("mean_comm_rate", (10, "normal", "shallow"), .231, "small"),
            ("mean_tracking_error", (100, "low", "deep"), 3., "hard"),
            ("mean_comm_rate", (50, "low", "deep"), .26, "hard"),
            ("stable", (20, "low", "deep"), False, "stability"),
        ):
            with self.subTest(metric=metric, target=target):
                altered = json.loads(originals[1])
                for row in altered["scenarios"]:
                    if (row["N"], row["pinned_mode"], row["depth_mode"]) == target:
                        row[metric] = value
                pair_paths[1].write_text(json.dumps(altered))
                self.run_cli("evaluate_scaling.py", "--compare", *pair_paths, "--output", self.out)
                acceptance = json.loads(self.out.read_text())["acceptance"]
                self.assertFalse(acceptance[gate])
                self.assertFalse(acceptance["passed"])

        # 达到误差降幅且通信率相对不变，仍须应用 30% 上限。
        for path, original in zip(pair_paths, originals):
            altered = json.loads(original)
            for row in altered["scenarios"]:
                row["mean_comm_rate"] = .31
            path.write_text(json.dumps(altered))
        self.run_cli("evaluate_scaling.py", "--compare", *pair_paths, "--output", self.out)
        self.assertFalse(json.loads(self.out.read_text())["hard"]["passed"])

        # 均值下降足够，但两粒训练 seed 的方向相反，置信上界应否决。
        for path in paths["v1"] + paths["v2"]:
            side, seed = path.stem.split("-")
            doc = json.loads(originals[int(side == "v2")])
            doc["training_seed"] = int(seed)
            for row in doc["scenarios"]:
                if side == "v2" and row["N"] >= 50:
                    row["mean_tracking_error"] = (1 if row["N"] == 50 else 3) * (.4 if seed == "11" else 1.1)
            path.write_text(json.dumps(doc))
        self.run_cli(*flags)
        hard = json.loads(self.out.read_text())["hard"]
        self.assertGreater(hard["error_reduction"], .15)
        self.assertGreater(hard["error_delta_ci95"][1], 0)
        self.assertFalse(hard["passed"])

    def test_profile(self):
        flags = ["profile_bottleneck.py", "--algo", "mappo", "--num-followers", "5",
                 "--num-envs", "1", "--warmup-episodes", "0", "--measure-episodes", "1",
                 "--device", "cpu", "--output", self.out]
        self.run_cli(*flags, "--steps", "300")
        profile = json.loads(self.out.read_text())
        self.assertEqual(profile["episodes"][0]["rollout_lengths"], [128, 128, 44])
        self.assertGreaterEqual(profile["optimizer_steps"]["actor"], 3)
        print("CPU profile:", profile["episodes"], profile["optimizer_steps"])
        for key in ["cold_build", "hot_reset", "act", "env_step", "store", "update"]:
            self.assertGreater(profile["timings_ms"][key]["p95"], 0)
        self.run_cli(*flags, "--steps", "2", "--scale-switch-cycles", "2")
        cycles = json.loads(self.out.read_text())["scale_switch"]["cycles"]
        self.assertEqual([c["cache_size"] for c in cycles], [3, 3])
        print("CPU switch cycles:", cycles)
        self.assertEqual([c["scales"] for c in cycles], [[10, 20, 30, 10]] * 2)

    def test_profile_oom(self):
        self.run_cli("-c", """
import sys
from unittest.mock import patch
import profile_bottleneck
with patch.object(profile_bottleneck, 'build_env', side_effect=RuntimeError('CUDA out of memory')):
    raise SystemExit(profile_bottleneck.main(sys.argv[1:]))
""", "--algo", "mappo", "--device", "cpu", "--output", self.out, code=1)
        result = json.loads(self.out.read_text())
        self.assertEqual(result["status"], "oom")
        self.assertEqual(result["active_load"]["phase"], "cold_build")

    def test_sac_cli(self):
        import profile_bottleneck
        self.assertEqual(profile_bottleneck.parser().parse_args([]).algo, "sac")
        cmd = profile_bottleneck.parser().parse_args(["--algo", "sac", "--steps", "1", "--warmup", "0"])
        self.assertEqual((cmd.algo, cmd.steps, cmd.warmup), ("sac", 1, 0))

    def test_non_git_export_identity(self):
        import shutil
        import evaluate_scaling as es
        export_v1 = self.dir / "export_v1"
        shutil.copytree(V1 / "mas_cc", export_v1 / "mas_cc", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        # macOS tar 产生的 AppleDouble 隐藏文件不得破坏校验
        (export_v1 / "mas_cc" / "._agent.py").write_bytes(b"appledouble")
        ident = es.resolve_repo_identity(export_v1, "v1")
        self.assertEqual(ident, f"nongit:{es.BASELINE_MAS_CC_SHA256}")
        # 篡改 v1 源码必须被拒绝
        (export_v1 / "mas_cc" / "config.py").write_text("# tampered\n", encoding="utf-8")
        with self.assertRaises(ValueError):
            es.resolve_repo_identity(export_v1, "v1")

    def test_compare_rejects_mismatched_steps(self):
        from evaluate_scaling import METRICS
        paths = {}
        for version, steps in (("v1", 300), ("v2", 600)):
            rows = []
            for n in (5, 10, 20, 50, 100):
                hard = n >= 50
                for seed in range(1000, 1020):
                    for pin in ("normal", "low"):
                        for depth in ("shallow", "deep"):
                            err = (1.0 if n == 50 else 2.0) if hard else 0.2
                            if version == "v2":
                                err *= 0.5 if hard else 1.0
                            rows.append(dict(
                                dict.fromkeys(METRICS, 1.0), N=n, seed=seed, status="ok",
                                steps=steps, scenario_id=f"{n}-{pin}-{depth}-{seed}",
                                scenario_sha256=f"h-{n}-{pin}-{depth}-{seed}",
                                pinned_ratio_coincident=n == 5, pinned_mode=pin, depth_mode=depth,
                                actual_pinned=1, actual_H=4 if depth == "deep" else 2,
                                stable=True, all_finite=True, converged=True,
                                mean_tracking_error=err, mean_comm_rate=0.20))
            p = self.dir / f"mismatch-{version}.json"
            p.write_text(json.dumps(dict(
                worker_version=version, training_seed=11, steps=steps,
                manifest_sha256="same", manifest_scenario_count=len(rows), scenarios=rows)))
            paths[version] = p
        self.run_cli("evaluate_scaling.py", "--compare", paths["v1"], paths["v2"], "--output", self.out)
        report = json.loads(self.out.read_text())
        self.assertFalse(report["acceptance"]["steps_match"])
        self.assertFalse(report["acceptance"]["complete_benchmark_matrix"])
        self.assertFalse(report["acceptance"]["passed"])
        self.assertEqual(len(report["cells"]), 20)
