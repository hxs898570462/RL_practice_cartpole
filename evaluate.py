from pathlib import Path
from statistics import mean, pstdev

import gymnasium as gym
import torch
import argparse

from actor_critic import ActorCritic


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=None)
    args = parser.parse_args()
    run_dir = args.run_dir


    checkpoint = torch.load(
        run_dir / "model.pt",
        map_location="cpu",
        weights_only=True,
    )

    model = ActorCritic()

    # TODO 1：传入 checkpoint 中的模型参数字典
    load_result = model.load_state_dict(checkpoint["model_state_dict"])

    model.eval()
    torch.manual_seed(2026)

    print("Loaded weights:", load_result)
    print("Checkpoint steps:", checkpoint["total_steps"])
    print("Action mode: stochastic")

    env = gym.make(checkpoint["config"]["env_id"])
    returns = []

    try:
        for episode in range(20):
            env_seed = 1000 + episode
            obs, _ = env.reset(seed=env_seed)
            episode_return = 0.0
            done = False

            while not done:
                obs_tensor = torch.as_tensor(
                    obs, dtype=torch.float32
                ).unsqueeze(0)

                with torch.no_grad():
                    distribution, _ = model(obs_tensor)

                    # TODO 2：按照动作分布采样
                    action = distribution.sample()

                obs, reward, terminated, truncated, _ = (
                    env.step(action.item())
                )

                # TODO 3：累计本回合奖励
                episode_return += reward

                # TODO 4：判断回合是否结束
                done = terminated or truncated

            returns.append(episode_return)
            print(
                f"Episode {episode + 1:02d} | "
                f"seed={env_seed} | return={episode_return} | "
                f"terminated={terminated} | truncated={truncated}"
            )
    finally:
        env.close()

    print("Evaluation episodes:", len(returns))
    print(f"Mean return: {mean(returns):.2f}")
    print(f"Return std: {pstdev(returns):.2f}")
    print("Min return:", min(returns))
    print("Max return:", max(returns))