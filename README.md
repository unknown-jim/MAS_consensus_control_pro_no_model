## MAS_consensus_control_pro_no_model

本项目实现了一个 **多智能体 leader-follower 一致性控制（event-triggered）** 的强化学习训练框架，采用 **CTDE**（Centralized Training, Decentralized Execution：集中训练、分散执行）架构。

- **环境**：leader 轨迹 + follower 动力学 + 事件触发通信（阈值控制）
- **算法**：支持 `MASAC(CTDE-SAC)` 与 `MAPPO(CTDE-MAPPO)` 两种训练路径
- **输出**：自动保存 best checkpoint 与评估图（统一落到 `results/` 下）

---

### 代码结构

- `mas_cc/`: 核心可导入包
  - `config.py`: 全部超参/路径/维度定义（含输出目录规则）
  - `environment.py`: 单环境与并行环境（batch），含事件触发通信逻辑
  - `topology.py`: 通信拓扑（含 pinned followers）
  - `networks.py`: Actor/Critic/Value 网络
  - `buffer.py`: replay buffer（SAC）
  - `agent.py`: `CTDESACAgent` / `CTDEMAPPOAgent` 与 `ActorOnlyPolicy`
  - `utils.py`: 评估、轨迹采样、绘图
  - `dashboard.py`: Jupyter 训练仪表盘（可选，需额外依赖）
- 根目录脚本
  - `train.py`: 训练入口（可选启用仪表盘），训练结束会保存 best 模型并绘制评估图
  - `test_env.py`: 环境/阈值的快速 sanity check（训练前推荐先跑）
  - `diagnose.py`: CUDA / PyTorch 状态与粗粒度 GPU 性能诊断
  - `profile_bottleneck.py`: 训练关键路径短跑 profiling（actor/env/store/update）
- `results/`: 输出目录（模型、图像、评估结果等；默认会自动创建）

---

### 环境要求

- Python 3.10+
- PyTorch 2.x

核心依赖（按源码导入统计，实际以你的环境为准）：
- `torch`
- `numpy`

可视化/拓扑可视化相关（按需安装）：
- `matplotlib`
- `networkx`（仅 `topology.visualize()` 需要）

Notebook 仪表盘相关（按需安装）：
- `ipywidgets`
- `IPython`

---

### 快速开始

建议先确认你在仓库根目录运行。

#### 1) 环境自检（强烈推荐先跑）

```bash
python test_env.py
```

该脚本会做：
- 零控制 + 固定阈值下是否稳定
- 粗搜一个可用阈值（环境阈值，会映射到策略动作 raw 值）
- 轨迹跟踪相关性（leader 与 follower 均值位置）

#### 2) 开始训练

```bash
python train.py
```

训练算法由 `mas_cc/config.py` 中的 `ALGO` 控制：
- `ALGO = "MAPPO"`：on-policy（rollout + PPO 更新）
- `ALGO = "MASAC"`：off-policy（replay buffer + SAC 更新）

训练会：
- 创建 `CommunicationTopology`、`BatchedModelFreeEnv` 并进行训练
- 记录 best reward 的模型参数
- 将 best checkpoint 保存到 `mas_cc.config.SAVE_MODEL_PATH`
- 在训练结束后调用 `mas_cc.utils.plot_evaluation(...)` 保存评估图到 `mas_cc.config.EVAL_SAVE_PATH`

#### 3) 启用 Notebook 仪表盘（可选）

`train.py` 里 `train(show_dashboard=True)` 会在运行时导入 `mas_cc/dashboard.py`。
该模式依赖 `ipywidgets` 与 `IPython`，建议在 Jupyter 环境中使用。

#### 4) 设备/性能诊断

```bash
python diagnose.py
```

用于快速确认 CUDA 是否可用，并通过两组矩阵乘法粗测 GPU 性能。

#### 5) 训练瓶颈粗定位（短跑 profiling）

```bash
python profile_bottleneck.py --steps 300 --warmup 10
```

该脚本会统计 actor/env/store/update 各阶段耗时占比，帮助定位“慢在哪里”。

---

### 常用配置入口（`mas_cc/config.py`）

- **算法选择**：`ALGO`（`"MAPPO"` 或 `"MASAC"`）
- **规模相关**：`NUM_FOLLOWERS`、`NUM_PINNED`、`NUM_PARALLEL_ENVS`、`MAX_STEPS`
- **事件触发通信**：`THRESHOLD_MIN`/`THRESHOLD_MAX`、`AGE_MAX_STEPS`、`COOLDOWN_STEPS`、`COMM_PENALTY`
- **输出目录**：
  - 默认根目录：`OUTPUT_ROOT`（默认 `results`）
  - 运行目录：`RUN_DIR`（默认 `results/<algo>/YYYYMMDD/HHMMSS/`）
  - 模型：`MODELS_DIR`、`SAVE_MODEL_PATH`
  - 图片：`FIGS_DIR`、`EVAL_SAVE_PATH`

支持环境变量覆盖：
- `RUN_DIR`：直接指定本次运行输出目录（最高优先级）
- `OUTPUT_ROOT`：指定输出根目录（默认 `results`）

示例：
```bash
RUN_DIR=/tmp/ctde_run_001 python train.py
# 或
OUTPUT_ROOT=/tmp/results python train.py
```

---

### 输出目录结构（默认）

运行一次训练后会生成类似结构：

```text
results/
  <algo>/
    YYYYMMDD/
      HHMMSS/
        models/
          best_model_*.pt
        figs/
          final_evaluation_*.png
          training_progress_*.png  (仅 dashboard 保存时)
```

---

### 评估说明

- 训练脚本末尾会自动调用 `plot_evaluation(...)` 绘制位置/速度曲线，并保存到 `EVAL_SAVE_PATH`。
- Notebook：
  - `main.ipynb`、`evaluate_results_models.ipynb` 已统一从 `mas_cc` 包导入。
  - `evaluate_results_models.ipynb` 会从 `results/` 下查找 checkpoint 并做评估/可视化。

---

### 常见问题（Troubleshooting）

- **训练很慢**：先运行 `python diagnose.py` 确认 CUDA 可用；再用 `python profile_bottleneck.py ...` 看是 env/actor/update 哪块占比高。
- **评估 follower 数量变大**：可参考 `mas_cc/agent.py` 中的 `ActorOnlyPolicy`，它允许只加载 actor 做部署/泛化评估（无需 critic/buffer）。
- **输出目录太大**：建议把 `OUTPUT_ROOT` 指到仓库外的磁盘路径，或用 `RUN_DIR` 指定单次运行目录。
