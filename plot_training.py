import csv
from pathlib import Path
import torch

import matplotlib.pyplot as plt
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=Path("runs") / "seed42_clip0.1_updates100")
    args = parser.parse_args()

    run_dir = args.run_dir

    checkpoint = torch.load(
        run_dir / "model.pt",
        map_location="cpu",
        weights_only=True,
    )
    config = checkpoint["config"]

    with (run_dir / "training.csv").open(
        newline="", encoding="utf-8"
    ) as file:
        rows = list(csv.DictReader(file))

    # TODO 1：提取每行 steps，转换为整数列表
    steps = [int(row["steps"]) for row in rows]

    # TODO 2：提取每行 mean20，转换为浮点数列表
    mean_returns = [float(row["mean20"]) for row in rows]

    fig, ax = plt.subplots(
        figsize=(7, 4), layout="constrained"
    )

    # TODO 3：横轴 steps，纵轴 mean_returns
    ax.plot(
        steps,
        mean_returns,
        marker="o",
        label="Training mean (last 20 episodes)",
    )

    ax.set_xlabel("Environment steps")
    ax.set_ylabel("Mean undiscounted return")
    ax.set_title(
        f"CartPole PPO | seed {checkpoint['seed']} | "
        f"clip {config['clip_epsilon']}"
    )
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.3)
    ax.legend()

    output_path = run_dir / "training_curve.png"

    # TODO 4：把 fig 保存到 output_path，设置 dpi=150
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    print("Data points:", len(steps))
    print("First point:", steps[0], mean_returns[0])
    print("Last point:", steps[-1], mean_returns[-1])
    print("Plot saved:", output_path)