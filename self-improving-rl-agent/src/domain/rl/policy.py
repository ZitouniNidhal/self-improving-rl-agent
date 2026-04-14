"""Policy network — actor-critic for PPO."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from typing import Tuple


class PolicyNetwork(nn.Module):
    """
    Actor-Critic network for PPO.
    Shared backbone → separate actor (policy) and critic (value) heads.
    """

    def __init__(self, state_dim: int, action_dim: int, hidden: int = 128):
        super().__init__()

        # Shared encoder
        self.backbone = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )

        # Actor head — outputs action logits
        self.actor = nn.Linear(hidden, action_dim)

        # Critic head — outputs state value V(s)
        self.critic = nn.Linear(hidden, 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.01)
                nn.init.zeros_(m.bias)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (action_logits, state_value)."""
        x = self.backbone(state)
        return self.actor(x), self.critic(x).squeeze(-1)

    def act(self, state: torch.Tensor) -> Tuple[int, torch.Tensor, torch.Tensor]:
        """
        Sample an action from the policy.
        Returns (action_index, log_prob, value).
        """
        logits, value = self(state)
        dist = Categorical(logits=logits)
        action = dist.sample()
        return action.item(), dist.log_prob(action), value

    def evaluate(
        self, states: torch.Tensor, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions for PPO update.
        Returns (log_probs, values, entropy).
        """
        logits, values = self(states)
        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_probs, values, entropy