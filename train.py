import gymnasium as gym
import torch
import csv
import argparse

from pathlib import Path
from actor_critic import ActorCritic
from gae import compute_gae
from ppo_update import ppo_update
from rollout import collect_rollout


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-updates", type=int, default=100)
    parser.add_argument("--clip-epsilon", type=float, default=0.1)
    args = parser.parse_args()

    # TODO：从 args 取得三个参数
    seed = args.seed
    num_updates = args.num_updates
    clip_epsilon = args.clip_epsilon

    rollout_steps = 1024
    history = []

    run_dir = (
        Path(__file__).resolve().parent
        / "runs"
        / f"seed{seed}_clip{clip_epsilon}_updates{num_updates}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(seed)
    model = ActorCritic()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    env = gym.make("CartPole-v1")

    total_steps = 0
    optimizer_steps_total = 0
    episode_returns = []

    try:
        obs, _ = env.reset(seed=seed)
        episode_return = 0.0

        for update in range(num_updates):
            # TODO 1：采集新经验，接收四个返回值
            # 指定 rollout_steps，并传入 episode_return
            trajectory, obs, episode_return, completed_returns = collect_rollout(
                env,
                model,
                obs,
                rollout_steps=rollout_steps,
                episode_return=episode_return,
            )

            # TODO 2：在参数更新前计算本批 GAE 和价值目标
            advantages, value_targets = compute_gae(
                trajectory,
                model
            )

            # TODO 3：更新模型，接收参数更新次数
            # epochs=4，minibatch_size=64
            # clip_epsilon=0.2，verbose=False
            update_steps = ppo_update(
                trajectory,
                model,
                optimizer,
                advantages,
                value_targets,
                epochs=4,
                minibatch_size=64,
                clip_epsilon=clip_epsilon,
                verbose=False
            )

            total_steps += len(trajectory)
            optimizer_steps_total += update_steps
            episode_returns.extend(completed_returns)

            recent_returns = episode_returns[-20:]
            mean20 = (
                sum(recent_returns) / len(recent_returns)
                if recent_returns else None
            )

            # TODO 1：把当前迭代的五项日志存成字典
            history.append({
                "update": update + 1,
                "steps": total_steps,
                "episodes": len(episode_returns),
                "mean20": mean20,
                "optimizer_steps": optimizer_steps_total,
            })

            print(
                f"Update {update + 1:02d} | "
                f"steps={total_steps} | "
                f"episodes={len(episode_returns)} | "
                f"mean20={mean20} | "
                f"optimizer_steps={optimizer_steps_total}"
            )

        model_path = run_dir / "model.pt"
        log_path = run_dir / "training.csv"

        checkpoint = {
            # TODO 2：取得模型参数字典
            "model_state_dict": model.state_dict(),
            "seed": seed,
            "total_steps": total_steps,
            "optimizer_steps": optimizer_steps_total,
            "config": {
                "env_id": "CartPole-v1",
                "num_updates": num_updates,
                "rollout_steps": rollout_steps,
                "learning_rate": 3e-4,
                "epochs": 4,
                "minibatch_size": 64,
                "clip_epsilon": clip_epsilon,
                "gamma": 0.99,
                "gae_lambda": 0.95,
                "value_coef": 0.5,
                "entropy_coef": 0.01,
            },
        }

        # TODO 3：用 torch.save 将 checkpoint 保存到 model_path
        torch.save(checkpoint, model_path)

        with log_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(
                file,
                fieldnames=[
                    "update",
                    "steps",
                    "episodes",
                    "mean20",
                    "optimizer_steps",
                ],
            )
            writer.writeheader()
            writer.writerows(history)

        print("Model saved:", model_path)
        print("Training log saved:", log_path)

        loaded = torch.load(
            model_path,
            map_location="cpu",
            weights_only=True,
        )
        print("Checkpoint steps:", loaded["total_steps"])
    finally:
        env.close()