"""Phase 2 外围接线检查：dashboard 动态规模、Notebook AST、README CLI。

运行：
    PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg TOP_K=4 python -B -m unittest tests.test_phase2_wiring
"""
from __future__ import annotations

import ast
import json
from pathlib import Path
import re
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from mas_cc.dashboard import TrainingDashboard

REPO = Path(__file__).resolve().parents[1]


def _trajectory(n: int, pinned_ids, seed: int = 0) -> dict:
    """构造 utils.collect_trajectory 风格的轨迹字典。"""
    rng = np.random.default_rng(seed)
    steps = 6
    return {
        "num_followers": n,
        "pinned_ids": list(pinned_ids),
        "topology_stats": {},
        "times": np.linspace(0.0, 1.0, steps + 1),
        "leader_pos": rng.normal(size=steps + 1),
        "leader_vel": rng.normal(size=steps + 1),
        "follower_pos": rng.normal(size=(steps + 1, n)),
        "follower_vel": rng.normal(size=(steps + 1, n)),
        "comm_rates": rng.uniform(size=steps),
        "theta_norms": rng.uniform(size=(steps, n)),
    }


def _legacy_trajectory(n: int, seed: int = 1) -> dict:
    """旧格式：没有 num_followers/pinned_ids，只有含 leader 列的 positions。"""
    rng = np.random.default_rng(seed)
    steps = 4
    return {
        "times": np.linspace(0.0, 1.0, steps + 1),
        "leader_pos": rng.normal(size=steps + 1),
        "leader_vel": rng.normal(size=steps + 1),
        "positions": rng.normal(size=(steps + 1, n + 1)),
        "velocities": rng.normal(size=(steps + 1, n + 1)),
        "comm_rates": rng.uniform(size=steps),
        "theta_norms": rng.uniform(size=(steps, n)),
    }


def _colored_counts(ax, n: int, pinned_ids) -> tuple[int, int]:
    """按 colormap 匹配 ax1 中的 pinned/normal 曲线数量。"""
    pinned = {matplotlib.colors.to_rgba(c) for c in plt.cm.Greens(np.linspace(0.5, 0.9, len(pinned_ids)))}
    normal = {matplotlib.colors.to_rgba(c) for c in plt.cm.Blues(np.linspace(0.4, 0.8, n - len(pinned_ids)))}
    n_pinned = n_normal = 0
    for line in ax.get_lines():
        color = matplotlib.colors.to_rgba(line.get_color())
        if color in pinned:
            n_pinned += 1
        elif color in normal:
            n_normal += 1
    return n_pinned, n_normal


class DashboardWiringTest(unittest.TestCase):
    def setUp(self):
        self.topology = SimpleNamespace(pinned_ids=[1, 3], num_followers=10)
        self.dashboard = TrainingDashboard(2, vis_interval=1, topology=self.topology)

    def tearDown(self):
        plt.close("all")

    def test_scale_switch_updates_title_and_colors(self):
        dash = self.dashboard
        self.assertEqual(dash.current_num_followers, 10)
        self.assertIn("10 followers", dash.title_html.value)

        first = _trajectory(10, (1, 3), seed=10)
        dash.update_episode(1, 1.0, 0.2, 0.3, {"policy": 0.1, "entropy": 1.5}, first)
        self.assertEqual(dash.current_num_followers, 10)
        self.assertIn("N=10", dash._last_progress_fig.axes[0].get_title())
        first_colors = [line.get_color() for line in dash._last_progress_fig.axes[0].get_lines()]

        # 切换到 N=30 且 pinned_ids 完全不同：标题/着色/计数都跟随新轨迹。
        second = _trajectory(30, (2, 5, 7, 30), seed=30)
        dash.update_episode(2, 2.0, 0.1, 0.2, {"policy": 0.05, "entropy_joint": 1.2}, second)
        self.assertEqual(dash.current_num_followers, 30)
        self.assertIn("30 followers", dash.title_html.value)
        self.assertIn("N=30", dash._last_progress_fig.axes[0].get_title())
        self.assertIn("N=30", dash._last_progress_fig.axes[1].get_title())
        self.assertIn("N=30", dash._last_progress_fig._suptitle.get_text())

        pinned_count, normal_count = _colored_counts(dash._last_progress_fig.axes[0], 30, (2, 5, 7, 30))
        self.assertEqual((pinned_count, normal_count), (4, 26))
        second_colors = [line.get_color() for line in dash._last_progress_fig.axes[0].get_lines()]
        self.assertNotEqual(first_colors, second_colors)

    def test_legacy_trajectory_fallback(self):
        dash = self.dashboard
        legacy = _legacy_trajectory(12, seed=2)
        self.assertEqual(dash._trajectory_num_followers(legacy), 12)
        self.assertEqual(dash._trajectory_pinned_ids(legacy), {1, 3})
        self.assertEqual(dash._follower_array(legacy, "follower_pos").shape[1], 12)

        dash.best_trajectory = legacy
        dash._update_plots()
        self.assertEqual(dash.current_num_followers, 12)
        pinned_count, normal_count = _colored_counts(dash._last_progress_fig.axes[0], 12, (1, 3))
        self.assertEqual((pinned_count, normal_count), (2, 10))

    def test_entropy_legacy_fallback(self):
        self.assertEqual(self.dashboard._entropy_from_losses({"entropy": 0.25, "entropy_joint": 9.0}), 0.25)
        self.assertEqual(self.dashboard._entropy_from_losses({"entropy_joint": 1.75}), 1.75)

    def test_export_uses_current_scale(self):
        dash = self.dashboard
        dash.best_trajectory = _trajectory(30, (2, 5, 7, 30), seed=4)
        dash._update_plots()

        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch("mas_cc.dashboard.FIGS_DIR", tmp):
                saved = dash.save_training_progress()
            self.assertIsNotNone(saved)
            self.assertTrue(Path(saved).exists(), saved)
            self.assertIn("_30f_", Path(saved).name)


def _strip_magics(code: str) -> str:
    kept = []
    for line in code.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("%") or stripped.startswith("!"):
            continue
        kept.append(line)
    return "\n".join(kept)


def _read_notebook(name: str) -> dict:
    path = REPO / name
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


class NotebookWiringTest(unittest.TestCase):
    def test_notebooks_are_valid_json_and_compile(self):
        for name in ("main.ipynb", "evaluate_results_models.ipynb"):
            with self.subTest(notebook=name):
                nb = _read_notebook(name)
                self.assertEqual(nb["nbformat"], 4)
                code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
                self.assertTrue(code_cells)
                for idx, cell in enumerate(code_cells):
                    source = cell["source"]
                    code = "".join(source) if isinstance(source, list) else source
                    with self.subTest(notebook=name, cell=idx):
                        compile(_strip_magics(code), f"{name}:cell{idx}", "exec")

    def test_main_notebook_phase2_tokens(self):
        code = "".join(
            "".join(c["source"]) if isinstance(c["source"], list) else c["source"]
            for c in _read_notebook("main.ipynb")["cells"] if c["cell_type"] == "code"
        )
        for token in (
            "ActorOnlyPolicy.from_checkpoint",
            "train_follower_scales=TRAIN_FOLLOWER_SCALES",
            "10 + 7",
            "has_leader_est",
            "valid",
            "DeepSets",
            "reset(scenario=scenario)",
        ):
            with self.subTest(token=token):
                self.assertIn(token, code)

    def test_eval_notebook_phase2_tokens(self):
        code = "\n".join(
            "".join(c["source"]) if isinstance(c["source"], list) else c["source"]
            for c in _read_notebook("evaluate_results_models.ipynb")["cells"]
        )
        for token in (
            "v2",
            "ActorOnlyPolicy.from_checkpoint",
            "final_actor.pt",
            "evaluate_scaling.py --worker-version v1",
            "SAVE_OUTPUTS = False",
            "reset(scenario=fixed_scenario)",
            "avg_error",
            "mean_tracking_error",
        ):
            with self.subTest(token=token):
                self.assertIn(token, code)


def _argparse_flags(script: str) -> set[str]:
    tree = ast.parse((REPO / script).read_text(encoding="utf-8"))
    flags: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "add_argument":
            for arg in node.args:
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str) and arg.value.startswith("--"):
                    flags.add(arg.value)
    return flags


class ReadmeCliTest(unittest.TestCase):
    def test_documented_flags_exist_in_argparse(self):
        readme = (REPO / "README.md").read_text(encoding="utf-8")
        eval_flags = _argparse_flags("evaluate_scaling.py")
        prof_flags = _argparse_flags("profile_bottleneck.py")

        self.assertTrue({"--generate-manifest", "--parity-check", "--evaluate", "--compare"} <= eval_flags)
        self.assertTrue({"--algo", "--num-followers", "--num-envs", "--steps", "--scale-switch-cycles"} <= prof_flags)

        for block in re.findall(r"```(?:bash|text)\n(.*?)```", readme, flags=re.S):
            flags = set(re.findall(r"--[a-z][a-z0-9-]*", block))
            if not flags:
                continue
            if "evaluate_scaling.py" in block:
                with self.subTest(block="evaluate_scaling"):
                    self.assertFalse(flags - eval_flags, f"README 中的未知标志: {sorted(flags - eval_flags)}")
            if "profile_bottleneck.py" in block:
                with self.subTest(block="profile_bottleneck"):
                    self.assertFalse(flags - prof_flags, f"README 中的未知标志: {sorted(flags - prof_flags)}")

    def test_readme_mentions_required_workflows(self):
        readme = (REPO / "README.md").read_text(encoding="utf-8")
        for token in (
            "10 + 7K", "38", "valid", "has_leader_est", "置换不变", "BFS",
            "TRAIN_FOLLOWER_SCALES = (10, 20, 30)", "train_follower_scales=(20,)",
            "--worker-version v1", "--algo mappo", "--scale-switch-cycles 10",
        ):
            with self.subTest(token=token):
                self.assertIn(token, readme)


if __name__ == "__main__":
    unittest.main()
