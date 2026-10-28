# CartPole PPO Practice

使用 **PyTorch** 与 **Gymnasium**，从基础张量操作逐步实现 CartPole-v1 上的 PPO-Clip。项目保留分阶段练习和学习记录；实现过程包含 AI 提示、框架与局部代码示范，具体贡献与证据见 [学习记录](learning/project-notes.md)。

当前已实现固定步数采样、GAE、小批量更新、持续训练、模型和日志保存、固定模型开发评估及训练曲线。已开展单训练种子的裁剪范围对照；三训练种子和保留评估种子的最终验收尚未完成。

## 当前功能

- CartPole-v1 随机策略基线，以及张量、自动求导和优化器练习。
- 独立 Actor / Critic 网络：两层 64 单元 Tanh 隐藏层。
- 固定步数跨回合采样，跨批次保留观测和未完成回合回报。
- GAE：真正终止时停止自举，终止或时间截断时切断跨回合递推。
- PPO 裁剪目标、价值损失、熵奖励及两类裁剪诊断。
- 每批优势统一标准化，再打乱索引执行多轮小批量更新。
- 命令行设置训练种子、迭代数和裁剪范围。
- 保存模型参数、配置、计数及 CSV 日志，加载固定模型评估并绘图。

## 环境与安装

开发环境为 Windows / PowerShell、Python **3.12.10** 和 CPU 版 PyTorch。`requirements.txt` 锁定依赖，包括 PyTorch `2.14.1+cpu`、Gymnasium `1.4.0`、NumPy `2.5.3` 和 Matplotlib `3.11.2`。

在项目根目录执行：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe check_environment.py
```

确认输出的解释器位于当前项目 `.venv`。迁移目录后应重新创建虚拟环境。Linux / macOS 将解释器路径改为 `.venv/bin/python`；当前锁定版本的跨平台安装尚未验证。

## 训练、评估与绘图

以下命令在项目根目录执行。先用一次迭代检查完整流程：

```powershell
.\.venv\Scripts\python.exe train.py --seed 42 --num-updates 1 --clip-epsilon 0.1
.\.venv\Scripts\python.exe evaluate.py --run-dir runs/seed42_clip0.1_updates1
.\.venv\Scripts\python.exe plot_training.py --run-dir runs/seed42_clip0.1_updates1
```

100 次迭代的开发实验：

```powershell
.\.venv\Scripts\python.exe train.py --seed 42 --num-updates 100 --clip-epsilon 0.1
.\.venv\Scripts\python.exe evaluate.py --run-dir runs/seed42_clip0.1_updates100
.\.venv\Scripts\python.exe plot_training.py --run-dir runs/seed42_clip0.1_updates100
```

训练参数默认值为 `--seed 42 --num-updates 100 --clip-epsilon 0.1`。输出写入脚本所在目录的 `runs/seed{seed}_clip{clip_epsilon}_updates{num_updates}/`：

- `model.pt`：模型参数、配置、环境步数和优化器更新计数。
- `training.csv`：`update,steps,episodes,mean20,optimizer_steps`。
- `training_curve.png`：由绘图命令生成；横轴为环境步数，纵轴为最近最多 20 个训练回合的平均回报。

重复使用相同训练参数会覆盖同名目录中的模型和日志。checkpoint 未保存优化器与随机数状态，不支持精确续训。实验产物位于被 Git 忽略的 `runs/`，克隆仓库后需自行运行训练。`.venv`、Python 缓存和 `.vs` 等本地编辑器配置也不上传。

评估与绘图时请始终显式提供 `--run-dir`。当前评估入口省略该参数会报错，绘图入口则默认使用 `runs/seed42_clip0.1_updates100`；统一为必填参数仍是待完成任务。

## 文件说明

| 文件 | 用途 |
| --- | --- |
| `check_environment.py` | 检查解释器及依赖导入 |
| `random_agent.py` | 10 回合随机策略基线 |
| `torch_basics.py` / `optimizer_basics.py` | 张量、自动求导及 SGD 练习 |
| `actor_critic.py` | Actor–Critic 网络与前向计算示例 |
| `policy_episode.py` | 单回合采样、GAE 和整批 10 轮 PPO 更新练习 |
| `gae.py` | 优势估计与价值目标 |
| `ppo_update.py` | PPO 损失、裁剪诊断、打乱与小批量更新 |
| `rollout.py` | 固定步数采样及采样、更新检查入口 |
| `train.py` | 参数化持续训练、模型与 CSV 保存 |
| `evaluate.py` | 加载固定模型，随机采样动作评估 20 回合 |
| `plot_training.py` | 从 CSV 绘图，标题读取 checkpoint 配置 |
| `learning/project-notes.md` | 学习过程、历史运行反馈与待完成事项 |

基础练习可依次运行 `random_agent.py`、`torch_basics.py`、`optimizer_basics.py`、`actor_critic.py`、`policy_episode.py` 和 `rollout.py`。这些独立入口不会复用已训练模型。

## PPO 实现与评估协议

| 项目 | 训练入口当前设置 |
| --- | --- |
| 环境 | `CartPole-v1`，4 维观测、2 个离散动作 |
| Actor / Critic | `4 → 64 → 64 → 2` logits / `4 → 64 → 64 → 1` 价值 |
| 每批环境步数 | `1024` |
| 每批更新 | `4` epochs，minibatch `64`，共 `64` 次优化器更新 |
| 优化器 / 学习率 | Adam / `3e-4` |
| 折扣因子 / GAE 系数 | `0.99` / `0.95` |
| PPO 裁剪范围 | 默认 `0.1`，可由命令行修改 |
| 价值损失 / 熵系数 | `0.5` / `0.01` |

旧策略对数概率、GAE 优势和价值目标在每批更新期间固定。Actor 使用标准化优势，Critic 使用原始价值目标。`ratio_outside_fraction` 记录概率比越界比例，`objective_clipped_fraction` 记录裁剪目标严格生效的比例；PPO 裁剪不强制概率比始终处于区间内。

`evaluate.py` 加载固定模型，不更新参数；动作 RNG 种子为 `2026`，环境初始化种子为 `1000` 至 `1019`，动作按策略分布随机采样。`model.eval()` 不会将动作选择改成 argmax。当前这组评估种子已用于开发与参数比较。

## 已有开发结果与限制

以下为学习记录中的用户运行结果。训练种子均为 `42`，每批 `1024` 环境步、`64` 次优化器更新，评估协议如上：

| clip | 迭代数 | 环境步数 | 训练末 mean20 | 固定模型评估均值 | 总体标准差 | 最小 / 最大 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `0.2` | 10 | 10240 | 143.75 | 178.95 | 83.45 | 56 / 432 |
| `0.2` | 100 | 102400 | 500 | 459.50 | 123.02 | 34 / 500 |
| `0.1` | 100 | 102400 | 500 | 471.85 | 99.93 | 48 / 500 |

训练 `mean20` 来自变化中的策略，不能替代固定模型评估。单训练种子的结果不足以判断某个裁剪范围普遍更优；当前较高开发均值 `471.85` 仍低于目标 `475`。

2026-10-28 发布前检查由 AI 执行：依赖一致性与全部 Python 文件语法检查通过；独立输出目录中完成一迭代训练、1024 环境步、64 次优化器更新，验证命令行参数、CSV、checkpoint、20 回合评估与曲线生成。重新评估本地两份 100 迭代 checkpoint，分别复现 `471.85` 和 `459.50`。本次未重新训练这两份模型，也未执行多种子最终验收。历史学习记录中的“AI 未运行”描述对应当时教学轮次。

## 后续计划

- [ ] 将评估与绘图入口的 `--run-dir` 统一设为必填，并验证缺参提示。
- [ ] 锁定训练配置，开展至少 3 个训练种子的实验。
- [ ] 使用预留环境种子 `2000` 至 `2019`，对各固定模型进行最终评估。
- [ ] 检验每个模型独立评估 20 回合的平均未折扣回报是否达到 `475`。
- [ ] 补充多种子的复现结果和稳定性分析。

## 参考资料

- [Gymnasium 基本用法](https://gymnasium.farama.org/introduction/basic_usage/)
- [CartPole 环境定义](https://gymnasium.farama.org/environments/classic_control/cart_pole/)
- [PyTorch 自动求导](https://docs.pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html)
- [Gymnasium 时间限制与自举](https://gymnasium.farama.org/tutorials/gymnasium_basics/handling_time_limits/)
- [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
- [Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438)
