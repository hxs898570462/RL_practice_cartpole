import gymnasium as gym
import torch

from actor_critic import ActorCritic
from gae import compute_gae
from ppo_update import compute_ppo_loss


torch.manual_seed(42)
model = ActorCritic()
env = gym.make("CartPole-v1")
trajectory = []

try:
    obs, info = env.reset(seed=42)
    done = False

    while not done:
        obs_tensor = torch.as_tensor(
            obs, dtype=torch.float32
        ).unsqueeze(0)

        with torch.no_grad():
            # TODO 1：把观测输入 model
            distribution, value = model(obs_tensor)

            # TODO 2：按照 distribution 采样动作
            action = distribution.sample()

            old_log_probability = distribution.log_prob(action)

        # TODO 3：把 action 转为 Python 整数，再交给环境
        next_obs, reward, terminated, truncated, info = env.step(action.item())

        trajectory.append({
            "obs": obs.copy(),
            "action": action.item(),
            "reward": reward,
            "next_obs": next_obs.copy(),
            "old_log_prob": old_log_probability.item(),
            "value": value.item(),
            "terminated": terminated,
            "truncated": truncated,
        })

        obs = next_obs
        done = terminated or truncated

    total_reward = sum(item["reward"] for item in trajectory)

    print("Trajectory length:", len(trajectory))
    print("Episode return:", total_reward)
    print("First transition:", trajectory[0])
    print(
        "Final flags:",
        trajectory[-1]["terminated"],
        trajectory[-1]["truncated"],
    )

    advantages, value_targets = compute_gae(trajectory, model)

    print("Advantages shape:", advantages.shape)
    print("First advantage:", advantages[0].item())
    print("Last advantage:", advantages[-1].item())
    print("Value targets shape:", value_targets.shape)
    print("First value target:", value_targets[0].item())
    print("Last value target:", value_targets[-1].item())

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    epochs = 10

    for epoch in range(epochs):
        # TODO 1：调用 compute_ppo_loss，接收 loss 和 metrics
        loss, metrics = compute_ppo_loss(
            trajectory, model, advantages, value_targets
        )

        # TODO 2：清空梯度
        optimizer.zero_grad()

        # TODO 3：反向传播
        loss.backward()

        # TODO 4：更新参数
        optimizer.step()

        # 查看本轮更新后的指标
        with torch.no_grad():
            _, after_metrics = compute_ppo_loss(
                trajectory, model, advantages, value_targets
            )

        print(f"Epoch {epoch + 1:02d} after update:", after_metrics)


finally:
    env.close()