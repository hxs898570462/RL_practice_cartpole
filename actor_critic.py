import gymnasium as gym
import torch
from torch import nn
from torch.distributions import Categorical


class ActorCritic(nn.Module):
    def __init__(self):
        super().__init__()

        self.actor = nn.Sequential(
            nn.Linear(4, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 2),  # TODO 1：Actor 输出多少个得分？
        )

        self.critic = nn.Sequential(
            nn.Linear(4, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1),  # TODO 2：Critic 输出多少个值？
        )

    def forward(self, obs):
        logits = self.actor(obs)
        value = self.critic(obs)

        distribution = Categorical(logits=logits)
        return distribution, value.squeeze(-1)


if __name__ == "__main__":
    torch.manual_seed(42)
    env = gym.make("CartPole-v1")

    try:
        obs, info = env.reset(seed=42)
        model = ActorCritic()

        obs_tensor = torch.as_tensor(
            obs, dtype=torch.float32
        ).unsqueeze(0)

        with torch.no_grad():
            distribution, value = model(obs_tensor)
            action = distribution.sample()
            log_probability = distribution.log_prob(action)

        print("Observation shape:", obs_tensor.shape)
        print("Action probabilities:", distribution.probs)
        print("Probability sum:", distribution.probs.sum(dim=-1))
        print("Sampled action:", action.item())
        print("Log probability:", log_probability.item())
        print("Value shape:", value.shape)
        print("Value estimate:", value.item())

    finally:
        env.close()