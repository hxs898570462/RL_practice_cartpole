# CartPole PPO 项目学习记录

## 选题与最小范围

- 学习者自述已完成 David Silver 强化学习课程，尚无 PyTorch 使用经验。
- 已选择 Gymnasium CartPole-v1，以 PyTorch 从零实现 PPO；先采用单环境和 CPU。
- 最小成果：随机策略基线、Actor–Critic、轨迹采集与 GAE、PPO 更新、保存模型、训练曲线、独立评估及裁剪范围对照实验。
- 约定的教学目标：3 个训练种子，每个模型评估 20 回合，平均未折扣回报达到 475。已完成单批 19 条经验的 10 次整批参数更新检查；尚未完成完整训练或独立评估。

## 实际环境与证据

- 操作系统：Windows；工具：VS Code、PowerShell。
- 基础 Python：3.12.10，路径 D:\python3.12.10\python.exe。
- 项目解释器：D:\XJTLU\RL_PPO_Practice\.venv\Scripts\python.exe。
- 用户反馈：Virtual environment 为 True，pip 25.0.1 位于项目 .venv。
- 用户导入验证：PyTorch 2.14.1+cpu、Gymnasium 1.4.0、NumPy 2.5.3、Matplotlib 3.11.2。
- 用户已确认 VS Code 使用正确环境；安装与配置由用户执行。

## 理论与实践映射

| 编号与主题 | 实现位置 | 关键决策及证据 | 学习状态 |
|---|---|---|---|
| RL-PPO-01：环境交互与回合回报 | random_agent.py | 随机采样动作，累计即时奖励；terminated 或 truncated 均结束当前回合 | 提示下完成；用户运行输出与代码检查支持 |
| RL-PPO-02：张量与自动求导 | torch_basics.py | 对平方损失调用 backward，读取 w.grad；用户报告 w=2 时 loss=9、grad=-6，w=8 时 loss=9、grad=6，参数均未改变 | 提示下完成；用户正确解释梯度下降时应分别增大与减小参数 |
| RL-PPO-03：优化器与参数更新 | optimizer_basics.py | 用户完成 zero_grad、backward、step；SGD 学习率 0.1，10 次更新后 w 从 2 到 4.6779，损失从 9 到 0.1038 | 提示下完成；已只读检查代码，实际输出由用户提供 |
| RL-PPO-04：PPO 总体流程 | 聊天概念说明；ppo_update.py；policy_episode.py | 已讲解 Actor、Critic、优势、新旧策略概率比、裁剪目标以及按批采样和多轮更新；已在提示下实现一次整批更新 | 总体流程已讲解；概率比比较同状态同动作有用户独立解释证据，尚不能推断独立掌握完整 PPO |
| RL-PPO-05：Actor–Critic 前向计算 | actor_critic.py | 已检查两个独立的 4→64→64 网络、Actor 输出 2 个 logits、Critic 输出 1 个值和 forward；用户输出概率 [0.5451,0.4549]，概率和 1，动作 1，log_prob=-0.7875924110，价值形状 (1,)，价值 0.1107748002 | 提示下完成；用户运行证据与只读代码检查支持，网络尚未训练 |
| RL-PPO-06：策略交互与轨迹记录 | policy_episode.py | 已检查网络前向、分布采样、整数动作交互和八字段轨迹记录；用户回报 19.0，首条动作 1、奖励 1、old_log_prob=-0.7875924110、value=0.1107748002，最后标志 True/False；本轮用户明确提供轨迹长度 19 | 提示下完成；用户输出与只读代码检查支持 |
| RL-PPO-07：GAE 与价值目标 | gae.py；policy_episode.py 中调用 | 用户补全 TD 残差、逆序优势递推和未归一化优势加旧价值的目标；只读确认真正终止不自举，两种回合结束均切断递推。用户输出两个形状均为 (19,)，优势首项 11.4893093109、末项 0.9619714618，目标首项 11.6000843048、末项 1.0 | 提示下完成；真正终止回合有运行证据，仅截断分支和跨回合递推边界尚无运行验证；AI 未执行练习 |
| RL-PPO-08：PPO 损失与一次参数更新 | ppo_update.py；policy_episode.py | 2026-10-09 只读确认完整损失函数和 Adam 更新已保存；用户输出更新前概率比约 1，Actor 损失 -6.58788e-8、Critic 损失 61.206955、熵 0.69086975，总损失 30.596569；loss.requires_grad=True，Actor/Critic 第一层权重梯度范数为 0.03032805/1.43437934；更新后 Actor 损失 -0.00496056、Critic 损失 60.883331、熵 0.69190013，概率比范围 0.98068672～1.02061248 | 提示下完成，有用户运行证据和代码检查；用户解释同状态同动作才能观察新旧策略差异，支持该局部概念的独立解释；AI 未运行练习，回报 19 是更新前采样结果，不代表更新后性能 |
| RL-PPO-09：同一批经验多轮更新 | policy_episode.py | 用户已修正遗留单次更新与日志缩进；只读确认每轮重算损失，优化器与 GAE 在循环外。用户 10 行输出无报错，第一轮复现单次更新，末轮 Actor 损失 -0.04960499、Critic 损失 58.001659、熵 0.68651319、概率比范围 0.80912310～1.20460308；熵在第 4 轮约 0.69302672 达到本次峰值后降低 | 提示下完成；有代码检查与用户运行证据，尚无独立训练或评估，AI 未运行练习 |
| RL-PPO-10：概率比越界与目标裁剪诊断 | 本轮布置 ppo_update.py 的 metrics | 区分概率比落在 [1-epsilon,1+epsilon] 外的样本比例与裁剪分支严格降低 surrogate 的样本比例；后者取决于优势符号。只增加诊断，不改变损失与训练参数 | 当前任务；等待用户补全两个布尔掩码并反馈每轮比例；不从 ratio_max 单独推断裁剪目标已生效 |

已讲解 reset 的两个返回值、step 的五个返回值、元组解包，以及外层回合循环与内层步循环。GAE 已讲解并实现终止与截断的不同自举规则；后续多回合采样时仍需检查边界处理。

## 随机策略基线

- 文件：../random_agent.py。
- 动作空间种子：42；10 个回合的环境初始化种子依次为 42 至 51。
- 用户报告的未折扣回报：30、46、30、24、28、39、20、19、24、32。
- 合计 292，平均 29.20，最小值 19，最大值 46。
- 已只读检查代码：动作采样、五项返回值接收、奖励累计、terminated or truncated、每回合 reset 和 finally 中关闭环境均正确。
- 这是用户运行的结果，AI 未重新执行程序；框架和局部示范由 AI 提供，不能记为独立完成。
- 10 个回合仅作为初始基线，不能推断总体平均表现；随机策略没有参数学习或 PPO 训练。

## 当前任务与后续反馈

- 环境配置和随机策略最小交互流程已完成。
- 已完成标量自动求导练习；只读检查 torch_basics.py 确认 requires_grad=True、loss.backward() 和 w.grad.item() 使用正确。运行结果由用户提供，AI 未重新运行。
- 已完成 SGD 标量参数更新练习，用户输出与平方损失下的梯度下降一致。AI 未代用户运行练习。
- 已回答 PPO 总体流程与项目对应关系；用户要求进入下一步。
- ActorCritic 前向计算已完成，尚未训练；AI 未代用户运行。
- 一个未训练策略回合的采样与记录已完成；用户报告回报 19、最后真正终止。AI 未代用户运行采样。
- GAE 已通过只读代码检查和用户运行反馈：gamma=0.99、gae_lambda=0.95，形状匹配，最后真正终止转移的价值目标为 1.0。未将提示下实现记为独立掌握。
- 单次 PPO 更新已通过用户运行反馈和只读检查；已解释标准化优势使初始 Actor 损失近零但梯度可非零，Critic 损失是目标误差而非回报，熵以自然对数计且两动作最大值为 ln(2)，概率比取采样时策略为固定参照。记录梯度范数仅来自两网络第一层权重。
- 多轮更新调试已解决：循环后遗留的原单次更新区块重复对最后一个 loss 执行 backward；用户删除遗留区块并修正日志缩进后完成 10 次更新。每轮前向建立新计算图，不需要 retain_graph=True。AI 未执行或修改练习。
- 本轮输出解读：Actor 损失逐轮下降仅说明固定批次上的 surrogate 改善；Critic 对固定目标的 MSE 降低；熵先升后降且未超过 ln(2)；第 10 轮最大概率比略超 1.2，与 PPO 的目标裁剪并非硬约束一致，但是否选中裁剪分支还依赖该样本的优势符号。回报 19 仍来自更新前采样。
- 当前任务：在 ppo_update.py 增加 ratio_outside_fraction 与 objective_clipped_fraction 两个诊断指标，由用户补全越界掩码和 clipped_surrogate < surrogate 的比较逻辑；保持参数和损失不变，重新运行并反馈第 1～10 轮两个比例。框架由 AI 提供，尚无用户实现证据。
- 尚待完成：裁剪诊断、小批量 PPO 更新、多回合固定步数采样、完整训练、保存模型、曲线、独立评估、裁剪范围对照实验与复现说明。当前单回合输出不构成更新后策略性能证据。

## 参考资料

- [Gymnasium 基本用法](https://gymnasium.farama.org/introduction/basic_usage/)
- [CartPole 环境定义](https://gymnasium.farama.org/environments/classic_control/cart_pole/)
- [PyTorch 自动求导](https://docs.pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html)
- [PyTorch 参数优化](https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html)
- [PPO 原论文](https://arxiv.org/abs/1707.06347)
- [GAE 原论文](https://arxiv.org/abs/1506.02438)
- [Gymnasium 时间限制与自举](https://gymnasium.farama.org/tutorials/gymnasium_basics/handling_time_limits/)
