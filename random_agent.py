import gymnasium as gym

env = gym.make("CartPole-v1")
env.action_space.seed(42)

episode_returns = []

try:
    for episode in range(10):
        obs, info = env.reset(seed=42 + episode)
        total_reward = 0.0
        done = False

        while not done:
            # TODO 1：从动作空间中随机采样一个动作
            action = env.action_space.sample()

            # TODO 2：执行动作，接收环境返回的五个结果
            obs, reward, terminated, truncated, info = env.step(action)

            # TODO 3：累计本步奖励
            total_reward += reward

            # TODO 4：判断是否应该结束当前回合
            done = terminated or truncated

        episode_returns.append(total_reward)
        print(f"Episode {episode + 1}: return = {total_reward:.0f}")

    mean_return = sum(episode_returns) / len(episode_returns)
    print(f"Mean return: {mean_return:.2f}")

finally:
    env.close()