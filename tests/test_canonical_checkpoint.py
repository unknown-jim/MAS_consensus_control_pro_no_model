"""单算法唯一最优权重（models/best_<algo>.pt）回归测试。"""

from __future__ import annotations

import os
import subprocess
from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import torch

from mas_cc import agent as agents
from mas_cc.agent import ActorOnlyPolicy, save_canonical_best_actor
from mas_cc.config import OBS_LAYOUT, canonical_best_model_path
from mas_cc.networks import DecentralizedActor
from mas_cc.utils import evaluate_canonical_score
import train as training


REPO = Path(__file__).resolve().parent.parent


class CanonicalCheckpointTests(unittest.TestCase):
    def test_gitignore_allows_only_canonical_models(self):
        def ignored(rel):
            rc = subprocess.run(
                ["git", "-C", str(REPO), "check-ignore", "-q", rel],
                check=False,
            ).returncode
            return rc == 0

        self.assertFalse(ignored("models/best_mappo.pt"))
        self.assertFalse(ignored("models/best_masac.pt"))
        self.assertTrue(ignored("models/best_mappo_copy.pt"))
        self.assertTrue(ignored("models/best_mappo.pt.lock"))
        self.assertTrue(ignored("results/run/metrics.json"))
        self.assertTrue(ignored("results/mappo/20261127/models/best_model.pt"))
        self.assertTrue(ignored("scratch_actor.pt"))

    def test_canonical_best_model_path_anchored_to_repo_root(self):
        with patch.dict(os.environ, {"CANONICAL_MODELS_DIR": ""}):
            p = canonical_best_model_path("ppo")
            self.assertEqual(Path(p).resolve(), (REPO / "models" / "best_mappo.pt").resolve())

    def test_r1_audit_regressions(self):
        import threading
        import numpy as np
        from mas_cc import environment as env_mod

        with TemporaryDirectory() as tmp:
            mappo_path = Path(tmp) / "best_mappo.pt"
            a1 = DecentralizedActor(hidden_dim=16)
            a2 = DecentralizedActor(hidden_dim=16)
            a3 = DecentralizedActor(hidden_dim=16)
            with torch.no_grad():
                for a, v in ((a1, 0.1), (a2, 0.2), (a3, 0.3)):
                    for p in a.parameters():
                        p.fill_(v)

            # 1) 缺失签名(None) 不得覆盖已有非空签名
            save_canonical_best_actor(mappo_path, a1, "MAPPO", score=10.0, eval_signature="prod")
            r_none = save_canonical_best_actor(mappo_path, a2, "MAPPO", score=99.0, eval_signature=None)
            self.assertFalse(r_none["updated"])
            self.assertEqual(r_none["reason"], "eval_signature_mismatch")

            # 2) NumPy 标量（含 np.str_）metrics 必须转为内建类型，weights_only=True 可加载
            r_np = save_canonical_best_actor(
                mappo_path, a2, "MAPPO", score=20.0, eval_signature="prod",
                metrics={"np_f": np.float64(1.25), "np_i": np.int64(7), "np_b": np.bool_(True), "np_s": np.str_("peak")},
            )
            self.assertTrue(r_np["updated"])
            loaded_ckpt = torch.load(mappo_path, map_location="cpu", weights_only=True)
            self.assertEqual(loaded_ckpt["canonical_metrics"], {"np_f": 1.25, "np_i": 7, "np_b": True, "np_s": "peak"})

            # 3) 损坏权重（缺参数键/截断字节 IndexError）可恢复；临时 PermissionError/OOM 必须抛错并保留原件
            bad_ckpt = torch.load(mappo_path, map_location="cpu", weights_only=True)
            del bad_ckpt["actor"]["v_mean.bias"]
            bad_ckpt["canonical_score"] = 500.0
            torch.save(bad_ckpt, mappo_path)
            r_recover = save_canonical_best_actor(mappo_path, a3, "MAPPO", score=40.0, eval_signature="prod")
            self.assertTrue(r_recover["updated"])
            self.assertEqual(r_recover["reason"], "schema_changed_or_unreadable")
            good_bytes = mappo_path.read_bytes()
            for transient_exc in (
                PermissionError("denied"),
                MemoryError("oom"),
                RuntimeError("CUDA out of memory"),
                RuntimeError("DefaultCPUAllocator: can't allocate memory"),
            ):
                with patch.object(agents, "_read_checkpoint", side_effect=transient_exc):
                    with self.assertRaises(type(transient_exc)):
                        save_canonical_best_actor(mappo_path, a1, "MAPPO", score=1.0, eval_signature="prod")
                self.assertEqual(mappo_path.read_bytes(), good_bytes)
            mappo_path.write_bytes(bytes.fromhex("80024b"))
            r_trunc = save_canonical_best_actor(mappo_path, a3, "MAPPO", score=40.0, eval_signature="prod")
            self.assertTrue(r_trunc["updated"])
            self.assertEqual(r_trunc["reason"], "schema_changed_or_unreadable")

            # 4) KeyboardInterrupt 在 torch.save 中断不得遗留临时文件（NamedTemporaryFile 托管无重复关闭）
            with patch.object(torch, "save", side_effect=KeyboardInterrupt):
                with self.assertRaises(KeyboardInterrupt):
                    save_canonical_best_actor(mappo_path, a2, "MAPPO", score=999.0, eval_signature="prod")
            self.assertEqual(list(Path(tmp).glob(".tmp_ckpt_*.pt")), [])

            # 5) 并发保存（20 vs 30 竞争已有 10）最终必须保留 30
            conc_path = Path(tmp) / "conc_mappo.pt"
            save_canonical_best_actor(conc_path, a1, "MAPPO", score=10.0, eval_signature="prod")
            errors = []
            def worker(actor, sc):
                try:
                    save_canonical_best_actor(conc_path, actor, "MAPPO", score=sc, eval_signature="prod")
                except BaseException as exc:
                    errors.append(exc)
            t1 = threading.Thread(target=worker, args=(a2, 20.0))
            t2 = threading.Thread(target=worker, args=(a3, 30.0))
            t1.start(); t2.start(); t1.join(); t2.join()
            self.assertEqual(errors, [])
            final_ckpt = torch.load(conc_path, map_location="cpu", weights_only=True)
            self.assertEqual(final_ckpt["canonical_score"], 30.0)

            # 6) 奖励、边界与随机初态范围配置（含 LEADER_AMPLITUDE_RANGE / FOLLOWER_INIT_POS_STD_RANGE）必须体现在 eval_signature 中
            policy = ActorOnlyPolicy(a1)
            sig1 = evaluate_canonical_score(policy, max_steps=2, cases=((5, "normal", "shallow", 1000),))["eval_signature"]
            for attr, val in (
                ("TRACKING_PENALTY_MAX", 0.0),
                ("POS_LIMIT", 5.0),
                ("VEL_LIMIT", 5.0),
                ("COMM_PENALTY", 0.15000001),
                ("LEADER_AMPLITUDE_RANGE", (0.5, 1.0)),
                ("FOLLOWER_INIT_POS_STD_RANGE", (2.0, 3.0)),
            ):
                with patch.object(env_mod, attr, val):
                    sig_alt = evaluate_canonical_score(policy, max_steps=2, cases=((5, "normal", "shallow", 1000),))["eval_signature"]
                self.assertNotEqual(sig1, sig_alt, f"{attr} change must alter eval_signature")

    def test_canonical_best_model_path_normalization(self):
        self.assertEqual(canonical_best_model_path("MAPPO", models_dir="models"), os.path.join("models", "best_mappo.pt"))
        self.assertEqual(canonical_best_model_path("sac", models_dir="models"), os.path.join("models", "best_masac.pt"))
        with self.assertRaises(ValueError):
            canonical_best_model_path("unknown_algo")

    def test_save_canonical_best_actor_single_file_semantics(self):
        with TemporaryDirectory() as tmp:
            mappo_path = Path(tmp) / "best_mappo.pt"
            masac_path = Path(tmp) / "best_masac.pt"

            actor1 = DecentralizedActor(hidden_dim=16)
            actor2 = DecentralizedActor(hidden_dim=16)
            with torch.no_grad():
                for p in actor1.parameters():
                    p.fill_(0.1)
                for p in actor2.parameters():
                    p.fill_(0.25)

            # 1) 首次创建
            r1 = save_canonical_best_actor(
                mappo_path, actor1, "MAPPO", score=-20.0, eval_signature="sig_v1",
                metrics={"mean_tracking_error": 0.4},
            )
            self.assertTrue(r1["updated"])
            self.assertEqual(r1["reason"], "created")
            self.assertEqual(sorted(p.name for p in Path(tmp).glob("*.pt")), ["best_mappo.pt"])
            loaded1 = ActorOnlyPolicy.from_checkpoint(str(mappo_path))
            self.assertTrue(torch.allclose(loaded1.actor.v_mean.bias.cpu(), actor1.v_mean.bias.cpu()))
            bytes_v1 = mappo_path.read_bytes()

            # 2) 同算法同协议分数更差或持平 -> 不覆盖、不生成第二份文件
            for worse in (-25.0, -20.0):
                r_skip = save_canonical_best_actor(
                    mappo_path, actor2, "MAPPO", score=worse, eval_signature="sig_v1",
                )
                self.assertFalse(r_skip["updated"])
                self.assertEqual(r_skip["reason"], "existing_better_or_equal")
                self.assertEqual(mappo_path.read_bytes(), bytes_v1)

            # 3) 同算法同协议但评估口径不同（如短步数 smoke）-> 默认不覆盖
            r_sig = save_canonical_best_actor(
                mappo_path, actor2, "MAPPO", score=100.0, eval_signature="sig_smoke_2steps",
            )
            self.assertFalse(r_sig["updated"])
            self.assertEqual(r_sig["reason"], "eval_signature_mismatch")
            self.assertEqual(mappo_path.read_bytes(), bytes_v1)

            # 4) 同算法同协议且分数严格更优 -> 原子替换，仍只有 1 份文件
            r_imp = save_canonical_best_actor(
                mappo_path, actor2, "MAPPO", score=-12.5, eval_signature="sig_v1",
                metrics={"mean_tracking_error": 0.15},
            )
            self.assertTrue(r_imp["updated"])
            self.assertEqual(r_imp["reason"], "improved")
            self.assertEqual(r_imp["previous_score"], -20.0)
            self.assertEqual(sorted(p.name for p in Path(tmp).glob("*.pt")), ["best_mappo.pt"])
            loaded2 = ActorOnlyPolicy.from_checkpoint(str(mappo_path))
            self.assertTrue(torch.allclose(loaded2.actor.v_mean.bias.cpu(), actor2.v_mean.bias.cpu()))

            # 5) 观测协议变化 -> 直接覆盖
            r_schema = save_canonical_best_actor(
                mappo_path, actor1, "MAPPO", score=-99.0, obs_layout=OBS_LAYOUT + "_new",
                eval_signature="sig_v2",
            )
            self.assertTrue(r_schema["updated"])
            self.assertEqual(r_schema["reason"], "schema_changed_or_unreadable")

            # 6) 不同算法保存各自唯一文件
            r_sac = save_canonical_best_actor(
                masac_path, actor1, "MASAC", score=-15.0, eval_signature="sig_v1",
            )
            self.assertTrue(r_sac["updated"])
            self.assertEqual(sorted(p.name for p in Path(tmp).glob("*.pt")), ["best_mappo.pt", "best_masac.pt"])

            # 7) 非有限分数或布尔值拒绝
            for bad in (float("nan"), float("inf"), True):
                with self.assertRaises(ValueError):
                    save_canonical_best_actor(mappo_path, actor1, "MAPPO", score=bad)

    def test_evaluate_canonical_score_is_deterministic_and_isolated(self):
        actor = DecentralizedActor(hidden_dim=16)
        policy = ActorOnlyPolicy(actor)
        torch.manual_seed(123)
        r1 = evaluate_canonical_score(policy, max_steps=4, cases=((5, "normal", "shallow", 1000),))
        after_r1 = torch.rand(4)
        torch.manual_seed(123)
        r2 = evaluate_canonical_score(policy, max_steps=4, cases=((5, "normal", "shallow", 1000),))
        after_r2 = torch.rand(4)
        self.assertEqual(r1, r2)
        self.assertTrue(torch.equal(after_r1, after_r2))

    def test_train_updates_canonical_only_when_improved(self):
        with TemporaryDirectory() as tmp, ExitStack() as stack:
            canon = Path(tmp) / "models" / "best_mappo.pt"
            run_best = Path(tmp) / "run" / "best.pt"
            stack.enter_context(patch.object(training, "SAVE_MODEL_PATH", str(run_best)))
            stack.enter_context(patch.object(training, "NUM_PARALLEL_ENVS", 2))
            stack.enter_context(patch.object(training, "MAX_STEPS", 2))
            stack.enter_context(patch.object(agents, "PPO_EPOCHS", 1))
            stack.enter_context(patch.object(agents, "HIDDEN_DIM", 16))

            training.train(
                num_episodes=1, train_follower_scales=(10,), seed=7,
                show_dashboard=False, canonical_model_path=str(canon),
            )
            self.assertTrue(canon.is_file())
            first_bytes = canon.read_bytes()
            ckpt = torch.load(canon, map_location="cpu", weights_only=True)
            self.assertIn("canonical_score", ckpt)
            self.assertEqual(ckpt["algorithm"], "MAPPO")
            ActorOnlyPolicy.from_checkpoint(str(canon))

            # 人为写入更高的已有分数，再次运行训练时不得覆盖
            ckpt["canonical_score"] = 1e6
            torch.save(ckpt, canon)
            protected_bytes = canon.read_bytes()
            training.train(
                num_episodes=1, train_follower_scales=(10,), seed=8,
                show_dashboard=False, canonical_model_path=str(canon),
            )
            self.assertEqual(canon.read_bytes(), protected_bytes)
            self.assertNotEqual(protected_bytes, first_bytes)

    def test_r4_signature_short_read_and_unloadable_weights(self):
        from mas_cc import environment as env_mod

        actor = DecentralizedActor(hidden_dim=16)
        policy = ActorOnlyPolicy(actor)

        # 1) OBS_AGE_CAP 和随机化区间变更必须改变 eval_signature
        base = evaluate_canonical_score(policy, max_steps=4, cases=((5, "normal", "shallow", 1000),))
        with patch.object(env_mod, "OBS_AGE_CAP", 2):
            age_changed = evaluate_canonical_score(policy, max_steps=4, cases=((5, "normal", "shallow", 1000),))
        self.assertNotEqual(base["eval_signature"], age_changed["eval_signature"])
        with patch.object(env_mod, "LEADER_AMPLITUDE_RANGE", (1.5581355839967728, 1.8081355839967728)):
            amp_changed = evaluate_canonical_score(policy, max_steps=4, cases=((5, "normal", "shallow", 1000),))
        self.assertNotEqual(base["eval_signature"], amp_changed["eval_signature"])

        # 2) 单次短读不得把健康高分原件误判为损坏而覆盖
        with TemporaryDirectory() as tmp:
            ckpt_path = Path(tmp) / "best_mappo.pt"
            save_canonical_best_actor(ckpt_path, actor, "MAPPO", score=100.0, eval_signature="sig_v1")
            healthy_bytes = ckpt_path.read_bytes()

            real_load = torch.load
            calls = {"n": 0}

            def flaky_load(f, *args, **kwargs):
                calls["n"] += 1
                if calls["n"] == 1:
                    raise RuntimeError("PytorchStreamReader failed reading zip archive: failed finding central directory")
                return real_load(f, *args, **kwargs)

            with patch.object(agents.torch, "load", side_effect=flaky_load):
                res = save_canonical_best_actor(ckpt_path, actor, "MAPPO", score=1.0, eval_signature="sig_v1")
            self.assertFalse(res["updated"])
            self.assertEqual(res["reason"], "existing_better_or_equal")
            self.assertEqual(ckpt_path.read_bytes(), healthy_bytes)

            # 3) sparse / meta 等无法被 ActorOnlyPolicy 加载的同 shape 参数须判为不完整并恢复
            for bad_kind in ("sparse", "meta"):
                bad_ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
                orig_bias = bad_ckpt["actor"]["v_mean.bias"]
                if bad_kind == "sparse":
                    bad_ckpt["actor"]["v_mean.bias"] = orig_bias.to_sparse()
                else:
                    bad_ckpt["actor"]["v_mean.bias"] = torch.empty_like(orig_bias, device="meta")
                with patch.object(agents, "_read_checkpoint", return_value=(bad_ckpt, 16)):
                    res_bad = save_canonical_best_actor(ckpt_path, actor, "MAPPO", score=1.0, eval_signature="sig_v1")
                self.assertTrue(res_bad["updated"])
                self.assertEqual(res_bad["reason"], "schema_changed_or_unreadable")
                ActorOnlyPolicy.from_checkpoint(str(ckpt_path))

    def test_r5_fstat_eio_oom_struct_meta_and_rng_isolation(self):
        import errno
        import struct

        actor = DecentralizedActor(hidden_dim=16)
        with TemporaryDirectory() as tmp:
            ckpt_path = Path(tmp) / "best_mappo.pt"
            save_canonical_best_actor(ckpt_path, actor, "MAPPO", score=100.0, eval_signature="sig_v1")
            healthy_bytes = ckpt_path.read_bytes()

            # 1) 复核时 fstat 报 EIO 必须上抛 OSError，不得降级跳过尺寸校验而覆盖高分原件
            with patch.object(agents.torch, "load", side_effect=RuntimeError("failed finding central directory")), \
                 patch.object(agents.os, "fstat", side_effect=OSError(errno.EIO, "I/O error")):
                with self.assertRaises(OSError):
                    save_canonical_best_actor(ckpt_path, actor, "MAPPO", score=1.0, eval_signature="sig_v1")
            self.assertEqual(ckpt_path.read_bytes(), healthy_bytes)

            # 2) Actor 校验阶段遇到 typed torch.OutOfMemoryError("OOM") 必须原样上抛，不覆盖原件
            oom_cls = getattr(torch, "OutOfMemoryError", MemoryError)
            with patch.object(agents, "DecentralizedActor", side_effect=oom_cls("OOM")):
                with self.assertRaises(oom_cls):
                    save_canonical_best_actor(ckpt_path, actor, "MAPPO", score=1.0, eval_signature="sig_v1")
            self.assertEqual(ckpt_path.read_bytes(), healthy_bytes)

            # 3) 拒绝低分候选时不污染全局 CPU Torch RNG 状态
            torch.manual_seed(7101)
            expected_next = torch.rand(1).item()
            torch.manual_seed(7101)
            res_keep = save_canonical_best_actor(ckpt_path, actor, "MAPPO", score=1.0, eval_signature="sig_v1")
            actual_next = torch.rand(1).item()
            self.assertFalse(res_keep["updated"])
            self.assertEqual(actual_next, expected_next)

            # 4) 协议标记为 meta 标量时走 ValueError 恢复，不因 .item() 崩溃
            bad_meta = torch.load(ckpt_path, map_location="cpu", weights_only=True)
            bad_meta["actor"]["_obs_version"] = torch.empty((), dtype=torch.int64, device="meta")
            with patch.object(agents.torch, "load", return_value=bad_meta):
                res_meta = save_canonical_best_actor(ckpt_path, actor, "MAPPO", score=1.0, eval_signature="sig_v1")
            self.assertTrue(res_meta["updated"])
            self.assertEqual(res_meta["reason"], "schema_changed_or_unreadable")

            # 5) 截断旧格式 pickle 抛 struct.error 时可恢复
            legacy_path = Path(tmp) / "legacy.pt"
            torch.save({"actor": actor.state_dict()}, legacy_path, _use_new_zipfile_serialization=False)
            raw_legacy = legacy_path.read_bytes()
            # 确认截断/struct.error 均可恢复
            with patch.object(agents.torch, "load", side_effect=struct.error("unpack requires a buffer of 2 bytes")):
                res_struct = save_canonical_best_actor(ckpt_path, actor, "MAPPO", score=2.0, eval_signature="sig_v1")
            self.assertTrue(res_struct["updated"])
            self.assertEqual(res_struct["reason"], "schema_changed_or_unreadable")
            self.assertGreater(len(raw_legacy), 18)


if __name__ == "__main__":
    unittest.main()
