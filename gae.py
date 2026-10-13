import numpy as np
import torch


def compute_gae(trajectory, model, gamma=0.99, gae_lambda=0.95):
    rewards = torch.tensor(
        [item["reward"] for item in trajectory],
        dtype=torch.float32,
    )
    values = torch.tensor(
        [item["value"] for item in trajectory],
        dtype=torch.float32,
    )

    next_observations = torch.as_tensor(
        np.stack([item["next_obs"] for item in trajectory]),
        dtype=torch.float32,
    )

    with torch.no_grad():
        _, next_values = model(next_observations)

    advantages = torch.zeros(len(trajectory), dtype=torch.float32)
    next_advantage = 0.0

    for t in reversed(range(len(trajectory))):
        item = trajectory[t]

        bootstrap_mask = 0.0 if item["terminated"] else 1.0
        episode_ended = item["terminated"] or item["truncated"]
        continuation_mask = 0.0 if episode_ended else 1.0

        # TODO 1：本步奖励 + 折扣后的下一状态价值 - 当前价值
        delta = rewards[t] + gamma * next_values[t] * bootstrap_mask - values[t]

        # TODO 2：本步 TD 误差 + 加权的下一步优势
        next_advantage = delta + gamma * gae_lambda * next_advantage * continuation_mask

        advantages[t] = next_advantage

    # TODO 3：优势 + 采样时的价值估计
    value_targets = advantages + values

    return advantages, value_targets