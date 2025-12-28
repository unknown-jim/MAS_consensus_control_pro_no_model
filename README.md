# MAS_consensus_control_pro_no_model

本项目是一个 **多智能体 leader-follower 一致性控制（event-triggered）** 的强化学习实现，采用 **CTDE**（集中训练、分散执行）架构，支持 `MASAC(CTDE-SAC)` 与 `MAPPO` 两种训练路径。

## 代码结构

- `mas_cc/`: 核心可导入包
  - `config.py`: 全部超参/路径/维度定义
  - `environment.py`: 单环境与并行环境（batch）
  - `topology.py`: 通信拓扑（含 pinned followers）
  - `networks.py`: Actor/Critic 网络
  - `buffer.py`: replay / rollout buffer
  - `agent.py`: `CTDESACAgent` / `CTDEMAPPOAgent`
  - `utils.py`: 评估、轨迹采样、绘图
  - `dashboard.py`: Jupyter 仪表盘（可选使用，需额外依赖）
- `train.py`: 训练入口脚本（默认不启用 dashboard）
- `test_env.py`: 环境/阈值的快速 sanity check
- `diagnose.py`: 设备与性能诊断
- `profile_bottleneck.py`: 训练关键路径 profile
- `results/`: 输出目录（模型、图像、评估结果等）

## 环境要求

- Python 3.10+
- PyTorch **2.x**（项目已移除旧版本兼容分支）

如果你想在 Notebook 中启用仪表盘：需要 `matplotlib`、`ipywidgets`、`IPython`。

## 快速开始

- **训练（默认关闭 dashboard）**：运行 `python train.py`
- **只做环境自检**：运行 `python test_env.py`
- **诊断**：运行 `python diagnose.py`

## Notebook

`main.ipynb` / `evaluate_results_models.ipynb` 已统一从 `mas_cc` 包导入；如需在 Notebook 中启用 `TrainingDashboard`，请在调用 `train()` 时显式传入 `show_dashboard=True`。
