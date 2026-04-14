"""Rollout buffer for on-policy PPO training."""
from dataclasses import dataclass, field
from typing import List
import torch


@dataclass
class Transition:
    state:      torch.Tensor
    action:     int
    log_prob:   torch.Tensor
    value:      torch.Tensor
    reward:     float
    done:       bool


class RolloutBuffer:
    """
    Stores a full episode rollout for PPO update.
    On-policy: cleared after each update.
    """

    def __init__(self):
        self.transitions: List[Transition] = []

    def add(
        self,
        state: torch.Tensor,
        action: int,
        log_prob: torch.Tensor,
        value: torch.Tensor,
        reward: float,
        done: bool,
    ):
        self.transitions.append(
            Transition(state, action, log_prob, value, reward, done)
        )

    def compute_returns(self, gamma: float = 0.99) -> List[float]:
        """Compute discounted returns (Monte-Carlo, no bootstrapping)."""
        returns, R = [], 0.0
        for t in reversed(self.transitions):
            R = t.reward + gamma * R * (1 - t.done)
            returns.insert(0, R)
        return returns

    def to_tensors(self, gamma: float = 0.99):
        returns = self.compute_returns(gamma)
        states   = torch.stack([t.state    for t in self.transitions])
        actions  = torch.tensor([t.action  for t in self.transitions], dtype=torch.long)
        log_probs= torch.stack([t.log_prob for t in self.transitions])
        values   = torch.stack([t.value    for t in self.transitions])
        returns_t= torch.tensor(returns, dtype=torch.float32)
        advantages = (returns_t - values.detach())
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return states, actions, log_probs, returns_t, advantages

    def clear(self):
        self.transitions.clear()

    def __len__(self):
        return len(self.transitions)