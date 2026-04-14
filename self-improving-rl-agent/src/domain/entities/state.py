"""RL State Definition — describes the current ML task context."""
from dataclasses import dataclass, field
from typing import Dict, Any, List


@dataclass
class State:
    """
    Represents the environment state visible to the RL agent.
    Encodes dataset properties + history of past actions/rewards.
    """
    # Dataset characteristics
    n_samples: int = 0
    n_features: int = 0
    n_classes: int = 2
    task_type: str = "classification"   # "classification" | "regression"

    # Current episode context
    episode_step: int = 0
    best_val_acc: float = 0.0
    last_val_acc: float = 0.0
    last_train_loss: float = 1.0

    # History of last N actions (encoded as ints)
    action_history: List[int] = field(default_factory=list)

    # Optional: similarity vector retrieved from FAISS memory
    memory_context: List[float] = field(default_factory=list)

    def to_vector(self) -> List[float]:
        """Flatten state to a numeric vector for the policy network."""
        history_padded = (self.action_history + [0] * 10)[:10]
        return [
            self.n_samples / 100_000,
            self.n_features / 1_000,
            self.n_classes / 100,
            float(self.task_type == "classification"),
            self.episode_step / 50,
            self.best_val_acc,
            self.last_val_acc,
            self.last_train_loss,
            *history_padded,
            *self.memory_context[:16],   # clip / pad memory context
        ]

    @property
    def dim(self) -> int:
        return len(self.to_vector())