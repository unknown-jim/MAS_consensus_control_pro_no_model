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
        self.assertTrue(ignored("results/mappo/20261127/models/best_model.pt"))
        self.assertTrue(ignored("scratch_actor.pt"))

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


if __name__ == "__main__":
    unittest.main()
