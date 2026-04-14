"""PPO Agent — orchestrates policy, memory, and updates."""
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Tuple

from src.domain.rl.policy import PolicyNetwork
from src.domain.rl.memory import RolloutBuffer
from src.domain.entities.state import State
from src.domain.entities.action import Action, ACTION_SPACE_SIZE


class PPOAgent:
    """
    Proximal Policy Optimization agent for AutoML search.

    Hyperparameters:
        clip_eps   – PPO clipping parameter (ε)
        vf_coef    – value function loss coefficient
        ent_coef   – entropy bonus coefficient
        ppo_epochs – number of gradient steps per update
        gamma      – discount factor
        lr         – learning rate
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int = ACTION_SPACE_SIZE,
        hidden: int = 128,
        lr: float = 3e-4,
        clip_eps: float = 0.2,
        vf_coef: float = 0.5,
        ent_coef: float = 0.01,
        ppo_epochs: int = 4,
        gamma: float = 0.99,
        device: str = "cpu",
    ):
        self.device     = torch.device(device)
        self.clip_eps   = clip_eps
        self.vf_coef    = vf_coef
        self.ent_coef   = ent_coef
        self.ppo_epochs = ppo_epochs
        self.gamma      = gamma

        self.policy  = PolicyNetwork(state_dim, action_dim, hidden).to(self.device)
        self.buffer  = RolloutBuffer()
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

        self.total_updates = 0

    # ── Acting ────────────────────────────────────────────────────────────────

    def act(self, state: State) -> Tuple[Action, torch.Tensor, torch.Tensor]:
        """Select an action given the current state."""
        state_vec = torch.tensor(state.to_vector(), dtype=torch.float32).to(self.device)
        action_idx, log_prob, value = self.policy.act(state_vec)
        return Action.from_index(action_idx), log_prob, value

    def store(self, state: State, action: Action, log_prob, value, reward: float, done: bool):
        state_vec = torch.tensor(state.to_vector(), dtype=torch.float32)
        self.buffer.add(state_vec, action.action_type.value, log_prob, value, reward, done)

    # ── Updating ──────────────────────────────────────────────────────────────

    def update(self) -> dict:
        """Run PPO update on the collected rollout."""
        if len(self.buffer) == 0:
            return {}

        states, actions, old_log_probs, returns, advantages = self.buffer.to_tensors(self.gamma)
        states      = states.to(self.device)
        actions     = actions.to(self.device)
        old_log_probs = old_log_probs.detach().to(self.device)
        returns     = returns.to(self.device)
        advantages  = advantages.to(self.device)

        metrics = {"policy_loss": 0, "value_loss": 0, "entropy": 0}

        for _ in range(self.ppo_epochs):
            log_probs, values, entropy = self.policy.evaluate(states, actions)

            # Policy loss (clipped surrogate)
            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            # Value loss
            value_loss = nn.MSELoss()(values, returns)

            # Total loss
            loss = policy_loss + self.vf_coef * value_loss - self.ent_coef * entropy.mean()

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            self.optimizer.step()

            metrics["policy_loss"] += policy_loss.item()
            metrics["value_loss"]  += value_loss.item()
            metrics["entropy"]     += entropy.mean().item()

        self.buffer.clear()
        self.total_updates += 1
        n = self.ppo_epochs
        return {k: v / n for k, v in metrics.items()}

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str):
        torch.save({"policy": self.policy.state_dict(), "updates": self.total_updates}, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(ckpt["policy"])
        self.total_updates = ckpt.get("updates", 0)