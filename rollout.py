import torch
from ppo_update import ppo_update

def collect_rollout(
    env,
    model,
    obs,
    rollout_steps=256,
    episode_return=0.0,
):
    trajectory = []
    completed_returns = []

    for _ in range(rollout_steps):
        # TODO 1：
        # 复用原来的观测转张量、no_grad 前向、
        # 动作采样、旧对数概率计算，以及 env.step
        with torch.no_grad():
            obs_tensor = torch.as_tensor(
                obs, dtype=torch.float32
            ).unsqueeze(0)

            distribution, value = model(obs_tensor)
            action = distribution.sample()
            old_log_probability = distribution.log_prob(action)



        # TODO 2：
        # 复用原来的 trajectory.append，保留八个字段
        # 注意：这里保存的 next_obs 必须是 env.step 的结果
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
        episode_return += reward

        if terminated or truncated:
            completed_returns.append(episode_return)

            # TODO 3：
            # 重置环境，将新观测赋给 obs
            # 并将 episode_return 归零
            obs, _ = env.reset()
            episode_return = 0.0
        else:
            obs = next_obs

    return trajectory, obs, episode_return, completed_returns


if __name__ == "__main__":
    import gymnasium as gym

    from actor_critic import ActorCritic
    from gae import compute_gae

    torch.manual_seed(42)
    model = ActorCritic()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    env = gym.make("CartPole-v1")

    try:
        obs, _ = env.reset(seed=42)
        episode_return = 0.0

        for batch in range(1):
            trajectory, obs, episode_return, completed_returns = (
                collect_rollout(
                    env,
                    model,
                    obs,
                    episode_return=episode_return,
                )
            )

            advantages, value_targets = compute_gae(
                trajectory, model
            )

            print("Batch:", batch + 1)
            print("Trajectory length:", len(trajectory))
            print("Completed returns:", completed_returns)
            print("Unfinished episode return:", episode_return)
            print("Advantages shape:", advantages.shape)
            print("Value targets shape:", value_targets.shape)

            update_steps = ppo_update(
                trajectory,
                model,
                optimizer,
                advantages,
                value_targets,
            )
            print("Optimizer steps:", update_steps)
    finally:
        env.close()