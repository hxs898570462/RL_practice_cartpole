import numpy as np
import torch


def compute_ppo_loss(
    trajectory,
    model,
    advantages,
    value_targets,
    clip_epsilon=0.2,
    value_coef=0.5,
    entropy_coef=0.01,
):
    observations = torch.as_tensor(
        np.stack([item["obs"] for item in trajectory]),
        dtype=torch.float32,
    )
    actions = torch.tensor(
        [item["action"] for item in trajectory],
        dtype=torch.long,
    )
    old_log_probs = torch.tensor(
        [item["old_log_prob"] for item in trajectory],
        dtype=torch.float32,
    )

    # 这批经验的优势与目标固定，不参与求导
    advantages = advantages.detach()
    value_targets = value_targets.detach()

    # 只标准化 Actor 使用的优势，Critic 目标保持原值
    normalized_advantages = (
        advantages - advantages.mean()
    ) / (advantages.std(correction=0) + 1e-8)

    # 此处需要保留计算图，不能使用 torch.no_grad()
    distribution, values = model(observations)
    new_log_probs = distribution.log_prob(actions)

    # TODO 1：根据新旧对数概率计算概率比
    ratios = torch.exp(new_log_probs - old_log_probs)

    surrogate = ratios * normalized_advantages
    clipped_surrogate = torch.clamp(
        ratios, 1.0 - clip_epsilon, 1.0 + clip_epsilon
    ) * normalized_advantages

    # TODO 2：逐元素取两个 surrogate 的较小值，再取平均和负号
    actor_loss = -torch.min(surrogate, clipped_surrogate).mean()

    # TODO 3：当前 values 与固定 value_targets 的均方误差
    critic_loss = (values - value_targets).pow(2).mean()

    entropy = distribution.entropy().mean()

    # TODO 4：组合 Actor、Critic 与熵三项
    loss = actor_loss + value_coef * critic_loss - entropy_coef * entropy

    metrics = {
        "actor_loss": actor_loss.detach().item(),
        "critic_loss": critic_loss.detach().item(),
        "entropy": entropy.detach().item(),
        "ratio_min": ratios.detach().min().item(),
        "ratio_max": ratios.detach().max().item(),
    }
    return loss, metrics