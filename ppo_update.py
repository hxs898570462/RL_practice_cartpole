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
    normalize_advantages=True,
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
    if normalize_advantages:
        normalized_advantages = (
            advantages - advantages.mean()
        ) / (advantages.std(correction=0) + 1e-8)
    else:
        normalized_advantages = advantages

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


    # TODO 1：概率比低于下界，或者高于上界
    outside_mask = (ratios < 1.0 - clip_epsilon) | (ratios > 1.0 + clip_epsilon)

    # TODO 2：裁剪项严格小于未裁剪项
    objective_clipped_mask = clipped_surrogate < surrogate


    metrics = {
        "actor_loss": actor_loss.detach().item(),
        "critic_loss": critic_loss.detach().item(),
        "entropy": entropy.detach().item(),
        "ratio_min": ratios.detach().min().item(),
        "ratio_max": ratios.detach().max().item(),
        "ratio_outside_fraction": outside_mask.float().mean().item(),
        "objective_clipped_fraction": (
            objective_clipped_mask.float().mean().item()
        ),
    }
    return loss, metrics


def ppo_update(
    trajectory,
    model,
    optimizer,
    advantages,
    value_targets,
    epochs=4,
    minibatch_size=64,
    clip_epsilon=0.2,
    verbose=True,
):
    advantages = advantages.detach()
    value_targets = value_targets.detach()

    normalized_advantages = (
        advantages - advantages.mean()
    ) / (advantages.std(correction=0) + 1e-8)

    sample_count = len(trajectory)
    update_steps = 0

    for epoch in range(epochs):
        # TODO 1：生成 0 到 sample_count-1 的随机排列
        indices = torch.randperm(sample_count)

        for start in range(0, sample_count, minibatch_size):
            minibatch_indices = indices[
                start:start + minibatch_size
            ]

            minibatch_trajectory = [
                trajectory[i]
                for i in minibatch_indices.tolist()
            ]

            # TODO 2：用相同索引选取已经标准化的优势
            minibatch_advantages = normalized_advantages[minibatch_indices]

            # TODO 3：用相同索引选取原始价值目标
            minibatch_targets = value_targets[minibatch_indices]

            loss, _ = compute_ppo_loss(
                minibatch_trajectory,
                model,
                minibatch_advantages,
                minibatch_targets,
                clip_epsilon=clip_epsilon,
                normalize_advantages=False,
            )

            # TODO 4：清梯度、反向传播、更新参数
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            update_steps += 1

        # 每轮结束后，查看完整 256 条经验上的指标
        with torch.no_grad():
            _, metrics = compute_ppo_loss(
                trajectory,
                model,
                advantages,
                value_targets,
                clip_epsilon=clip_epsilon,
            )

        if verbose:
            print(f"Epoch {epoch + 1:02d} full batch:", metrics)
    return update_steps