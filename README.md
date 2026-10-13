# CartPole PPO Practice

使用 **PyTorch** 与 **Gymnasium**，从基础张量操作逐步实现 CartPole-v1 上的 Proximal Policy Optimization（PPO）。项目保留了分阶段练习脚本和学习记录，便于理解策略交互、Actor–Critic、广义优势估计（GAE）与 PPO 裁剪目标之间的关系。

> 当前版本是学习中的实现：已完成单回合采样、GAE 和同一批经验上的 10 轮参数更新，尚未实现持续采样的完整训练循环、模型保存和独立评估。单回合回报不能代表训练后的策略性能。

## 当前功能

- CartPole-v1 随机策略基线：固定种子运行 10 个回合。
- PyTorch 张量、自动求导和 SGD 参数更新练习。
- 独立的 Actor / Critic 网络：均采用两层 64 单元的 Tanh 隐藏层。
- 按策略分布采样动作，记录观测、动作、奖励、旧策略对数概率和价值等轨迹信息。
- GAE：区分真正终止与时间截断的自举逻辑，在回合结束处切断优势递推。
- PPO：裁剪策略目标、价值均方误差与熵奖励；标准化 Actor 使用的优势。
- 对固定的一批经验执行 10 轮 Adam 更新，并输出损失、熵和新旧策略概率比范围。

## 环境与安装

当前开发环境为 Windows / PowerShell、Python **3.12.10**，使用 CPU 版 PyTorch。`requirements.txt` 保存了当前环境的依赖版本，包括 PyTorch `2.14.1+cpu`、Gymnasium `1.4.0` 和 NumPy `2.5.3`。

在项目根目录执行：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe check_environment.py
```

确认 `Python executable` 指向当前项目的 `.venv`，并能输出 PyTorch 与 Gymnasium 的版本。上面的命令直接使用虚拟环境解释器，无需先激活环境。

Linux / macOS 使用 `python3 -m venv .venv` 创建环境，并将后续命令中的 `.\.venv\Scripts\python.exe` 替换为 `.venv/bin/python`。当前锁定版本的跨平台安装尚未验证。

如果使用 VS Code，请通过 `Python: Select Interpreter` 选择本项目 `.venv` 内的解释器。迁移项目后，在新位置重新创建虚拟环境。

## 文件说明

```text
RL_practice_cartpole/
├── README.md                 # 项目说明与运行方式
├── requirements.txt          # 当前环境的依赖版本
├── check_environment.py      # 检查解释器与依赖导入
├── random_agent.py           # 10 个回合的随机策略基线
├── torch_basics.py           # 张量与自动求导练习
├── optimizer_basics.py       # SGD 更新练习
├── actor_critic.py           # Actor–Critic 网络与前向计算示例
├── policy_episode.py         # 单回合采样、GAE 与 10 轮 PPO 更新
├── gae.py                    # 优势估计与价值目标
├── ppo_update.py             # PPO 损失与诊断指标
└── learning/
    └── project-notes.md      # 学习进度、运行反馈和待完成事项
```

`.venv`、Python 缓存和本地配置不纳入版本管理。

## 建议运行顺序

在项目根目录依次运行：

```powershell
.\.venv\Scripts\python.exe check_environment.py
.\.venv\Scripts\python.exe random_agent.py
.\.venv\Scripts\python.exe torch_basics.py
.\.venv\Scripts\python.exe optimizer_basics.py
.\.venv\Scripts\python.exe actor_critic.py
.\.venv\Scripts\python.exe policy_episode.py
```

`gae.py` 和 `ppo_update.py` 提供函数，由 `policy_episode.py` 调用，无需单独执行。

`policy_episode.py` 每次启动都会新建未训练的网络，采集一个回合，再反复使用这批经验更新 10 次。它会输出轨迹长度、采样回报、优势和价值目标，以及每轮更新后的指标；程序结束时不会保存参数。

## PPO 实现要点

| 项目 | 当前设置 |
| --- | --- |
| 环境 | `CartPole-v1`，4 维观测、2 个离散动作 |
| Actor | `4 → 64 → 64 → 2`，输出 logits，构造 `Categorical` 分布 |
| Critic | `4 → 64 → 64 → 1`，输出状态价值 |
| 随机种子 | 策略示例为 `42`；随机基线环境种子为 `42` 至 `51` |
| 折扣因子 / GAE 系数 | `gamma=0.99` / `gae_lambda=0.95` |
| PPO 裁剪范围 | `clip_epsilon=0.2` |
| 价值损失 / 熵系数 | `value_coef=0.5` / `entropy_coef=0.01` |
| 优化器 / 学习率 | Adam / `3e-4` |
| 每批更新轮数 | `10`，使用整批经验 |

采样时记录的 `old_log_prob` 固定作为比较基准；每轮更新重新计算当前策略的对数概率。GAE 得到的优势和价值目标在更新期间保持固定，Critic 使用未标准化的价值目标。

PPO 裁剪作用于优化目标，并不强制概率比始终位于 `[0.8, 1.2]`。当前打印的 `ratio_min` / `ratio_max` 可用于观察策略变化；判断某个样本是否实际使用了裁剪分支，还需要结合优势的符号。

## 已有运行记录与限制

[学习记录](learning/project-notes.md) 中保存了此前运行反馈：

- 随机策略基线的 10 回合平均回报为 **29.20**。
- 未训练策略的一次采样得到 **19 条转移、回报 19**，随后完成同一批经验的 10 轮更新。

这些数字来自学习过程中的运行记录，并非本次发布重新执行的基准，也不构成训练后策略性能的证据。学习记录中的旧目录路径保留为历史环境信息，当前解释器以 `check_environment.py` 输出为准。

## 后续计划

- [ ] 增加概率比越界比例与裁剪目标实际生效比例的诊断。
- [ ] 实现固定步数、多回合的轨迹收集。
- [ ] 加入打乱与小批量 PPO 更新。
- [ ] 实现持续采样和更新的完整训练循环。
- [ ] 保存 / 加载模型，绘制训练曲线。
- [ ] 使用多个训练种子和独立评估回合检验性能。
- [ ] 对比不同裁剪范围，并补充复现结果。

计划中的学习目标为：3 个训练种子，每个模型独立评估 20 回合，平均未折扣回报达到 475；当前尚未验证该目标。

## 参考资料

- [Gymnasium 基本用法](https://gymnasium.farama.org/introduction/basic_usage/)
- [CartPole 环境定义](https://gymnasium.farama.org/environments/classic_control/cart_pole/)
- [PyTorch 自动求导](https://docs.pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html)
- [Gymnasium 时间限制与自举](https://gymnasium.farama.org/tutorials/gymnasium_basics/handling_time_limits/)
- [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
- [Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438)
