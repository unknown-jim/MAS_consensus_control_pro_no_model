"""Phase 2 外围接线检查：dashboard 动态规模、Notebook AST、README CLI。

运行：
    PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg TOP_K=4 python -B -m unittest tests.test_phase2_wiring
"""
from __future__ import annotations

import ast
from contextlib import ExitStack
import json
import os
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
import torch

from mas_cc.config import TOP_K
from mas_cc.dashboard import TrainingDashboard
from mas_cc.topology import CommunicationTopology

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

        dash.update_episode(1, 1.0, 0.2, 0.3, {}, legacy)
        self.assertEqual(dash.current_num_followers, 12)
        self.assertIn("N=12", dash._last_progress_fig.axes[0].get_title())
        pinned_count, normal_count = _colored_counts(dash._last_progress_fig.axes[0], 12, (1, 3))
        self.assertEqual((pinned_count, normal_count), (2, 10))

    def test_training_title_tracks_each_episode_while_preview_keeps_its_scale(self):
        import train as training
        import mas_cc.agent as agents

        training_scales, title_scales = [], []
        start = agents.CTDEMAPPOAgent.start_episode
        update = TrainingDashboard.update_episode

        def record_start(agent, env):
            training_scales.append(env.num_followers)
            return start(agent, env)

        def record_update(dashboard, *args, **kwargs):
            update(dashboard, *args, **kwargs)
            title_scales.append(int(re.search(r"(\d+) followers", dashboard.title_html.value)[1]))
            fig = dashboard._last_progress_fig
            self.assertIn(f"N={title_scales[-1]}", fig._suptitle.get_text())
            preview = dashboard.best_trajectory
            n, pins = preview["num_followers"], preview["pinned_ids"]
            self.assertIn(f"N={n}", fig.axes[0].get_title())
            self.assertEqual(_colored_counts(fig.axes[0], n, pins), (len(pins), n - len(pins)))

        with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
            stack.enter_context(mock.patch.object(training, "SAVE_MODEL_PATH", os.path.join(tmp, "best.pt")))
            stack.enter_context(mock.patch("mas_cc.dashboard.FIGS_DIR", tmp))
            stack.enter_context(mock.patch.object(training, "MAX_STEPS", 2))
            stack.enter_context(mock.patch.object(training, "NUM_PARALLEL_ENVS", 2))
            stack.enter_context(mock.patch.object(agents, "PPO_EPOCHS", 1))
            stack.enter_context(mock.patch.object(agents, "HIDDEN_DIM", 16))
            stack.enter_context(mock.patch.object(agents.CTDEMAPPOAgent, "start_episode", record_start))
            stack.enter_context(mock.patch.object(TrainingDashboard, "update_episode", record_update))
            _, _, dashboard = training.train(num_episodes=3, train_follower_scales=(10, 20, 30),
                                              seed=11, vis_interval=1, show_dashboard=True, canonical_model_path="")
        self.assertEqual(list(zip(training_scales, title_scales)), [(10, 10), (20, 20), (30, 30)])
        self.assertIn("30 followers", dashboard.title_html.value)

    def test_entropy_legacy_fallback(self):
        self.assertEqual(self.dashboard._entropy_from_losses({"entropy": 0.25, "entropy_joint": 9.0}), 0.25)
        self.assertEqual(self.dashboard._entropy_from_losses({"entropy_joint": 1.75}), 1.75)

    def test_export_uses_current_scale(self):
        dash = self.dashboard
        preview = _trajectory(30, (2, 5, 7, 30), seed=4)
        dash.update_episode(1, 1.0, 0.2, 0.3, {}, preview, current_num_followers=10)
        progress = dash._last_progress_fig

        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch("mas_cc.dashboard.FIGS_DIR", tmp):
                saved = dash.save_training_progress()
            self.assertIsNotNone(saved)
            self.assertTrue(Path(saved).exists(), saved)
            self.assertIn("_30f_", Path(saved).name)
            self.assertIn("N=10", progress._suptitle.get_text())
            self.assertIn("N=30", progress.axes[0].get_title())


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
    def code_cell(self, notebook, marker):
        return next("".join(c["source"]) for c in _read_notebook(notebook)["cells"]
                    if c["cell_type"] == "code" and marker in "".join(c["source"]))

    def test_specific_dynamics_keeps_cpu_position_history(self):
        code = self.code_cell("main.ipynb", "def test_specific_dynamics")
        func = next(node for node in ast.parse(_strip_magics(code)).body
                    if isinstance(node, ast.FunctionDef) and node.name == "test_specific_dynamics")
        scope = {"np": np}
        exec(compile(ast.Module([func], type_ignores=[]), "main.ipynb", "exec"), scope)
        graph = CommunicationTopology(2, seed=5, device="cpu")
        action = torch.tensor([[0.5, 0.0], [0.5, 0.0]])
        policy = SimpleNamespace(select_action=lambda state, deterministic: action)
        with mock.patch("mas_cc.environment.DEVICE", torch.device("cpu")):
            traj = scope["test_specific_dynamics"](policy, graph, 1, 1, num_steps=3)
        expected = np.array([[0.0, 0.025, 0.075, 0.15]]).T.repeat(2, axis=1)
        np.testing.assert_allclose(traj["follower_pos"], expected)

    def test_fixed_scenario_graph_tensors_are_device_safe(self):
        for name in ("main.ipynb", "evaluate_results_models.ipynb"):
            with self.subTest(name=name):
                code = "\n".join("".join(c["source"]) for c in _read_notebook(name)["cells"]
                                 if c["cell_type"] == "code")
                self.assertNotRegex(code, r"np\.asarray\(\s*[\w.]+\.(?:adj|adj_matrix)\s*[,)]")

    def test_fixed_plot_consumes_same_trajectory_as_mse(self):
        code = self.code_cell("evaluate_results_models.ipynb", "env_fixed =")
        traj = _trajectory(3, [2])
        collect = mock.Mock(return_value=traj)
        scope = dict(np=np, plt=plt, CommunicationTopology=CommunicationTopology,
                     topology=CommunicationTopology(3), LEADER_TRAJECTORY_TYPES=["sine"],
                     LEADER_AMPLITUDE=1, LEADER_OMEGA=1, LEADER_PHASE=0, NUM_FOLLOWERS=3,
                     ModelFreeEnv=mock.Mock(), evaluate_agent=lambda *a, **k: {"mean_tracking_error": 0.},
                     collect_trajectory=collect, policy=object(), MAX_PLOT_FOLLOWERS=1,
                     plot_evaluation=mock.Mock(side_effect=AssertionError("resampled the fixed trajectory")))
        try:
            exec(code, scope)
            collect.assert_called_once_with(scope["policy"], scope["env_fixed"])
            self.assertEqual(scope["pos_mse"], np.mean((traj["follower_pos"] - traj["leader_pos"][:, None])**2))
            for ax, kind in zip(plt.gcf().axes, ("pos", "vel"), strict=True):
                np.testing.assert_array_equal(ax.lines[0].get_ydata(), traj[f"leader_{kind}"])
                np.testing.assert_array_equal(ax.lines[1].get_ydata(), traj[f"follower_{kind}"][:, 0])
                self.assertEqual(len(ax.lines), 2)
        finally:
            plt.close("all")

    def test_discover_and_identify_final_actor(self):
        code = self.code_cell("evaluate_results_models.ipynb", "def find_latest_model_in_results")
        scope = {"Path": Path, "torch": torch, "TOP_K": TOP_K}
        exec(code, scope)
        discover, guess = scope["find_latest_model_in_results"], scope["guess_algo_from_model_path"]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "results"
            models = root / "mappo" / "run" / "models"
            models.mkdir(parents=True)
            final = models / "final_actor.pt"
            weights = {"_obs_version": torch.tensor(2), "_top_k": torch.tensor(TOP_K)}
            torch.save({"algorithm": "MAPPO", "actor": weights}, final)
            self.assertEqual(Path(discover("mappo", root)), final)
            self.assertEqual(guess(str(final)), "mappo")
            best = models / "best_model_mappo.pt"
            torch.save({"algorithm": "mappo", "actor": weights, "value": {}}, best)
            os.utime(best, (1, 1))
            self.assertEqual(Path(discover("mappo", root)), final)
            torch.save({"algorithm": "masac", "actor": weights}, final)
            self.assertEqual(guess(str(final)), "masac")
            self.assertEqual(Path(discover("mappo", root)), best)
            self.assertEqual(Path(discover("masac", root)), final)
            torch.save(weights, final)
            self.assertEqual(guess(str(final)), "mappo")
            selection = self.code_cell("evaluate_results_models.ipynb", "if MODEL_PATH:")
            selection = selection.replace("MODEL_PATH = ''", f"MODEL_PATH = {str(final)!r}", 1)
            exec(selection, scope)
            self.assertEqual((scope["algo_tag"], scope["ckpt_kind"]), ("mappo", "v2-actor-only"))
            torch.save({"pos_embed": torch.zeros(1)}, final)
            with self.assertRaisesRegex(RuntimeError, r"evaluate_scaling\.py --evaluate --worker-version v1"):
                exec(selection, scope)

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
            "evaluate_scaling.py --evaluate --worker-version v1",
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
            "额外入边期望 ≤2", "1 条骨架入边", "总期望入度约为 3",
            "TRAIN_FOLLOWER_SCALES = (10, 20, 30)", "train_follower_scales=(20,)",
            "--worker-version v1", "--algo mappo", "--scale-switch-cycles 10",
        ):
            with self.subTest(token=token):
                self.assertIn(token, readme)


if __name__ == "__main__":
    unittest.main()
