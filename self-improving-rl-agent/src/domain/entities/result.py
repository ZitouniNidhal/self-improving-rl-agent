"""Result entity — output of an ML training run."""
from dataclasses import dataclass, field
from typing import Dict, Any


@dataclass
class TrainingResult:
    val_accuracy: float = 0.0
    val_loss: float = 999.0
    train_accuracy: float = 0.0
    train_loss: float = 999.0
    epochs_run: int = 0
    training_time_sec: float = 0.0
    model_params: int = 0
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def reward(self) -> float:
        """
        Composite reward:
          - high accuracy is good
          - fewer parameters is slightly rewarded (efficiency)
          - fast training is slightly rewarded
        """
        acc_reward = self.val_accuracy
        efficiency  = max(0.0, 1.0 - self.model_params / 5_000_000) * 0.05
        speed       = max(0.0, 1.0 - self.training_time_sec / 300)  * 0.05
        return round(acc_reward + efficiency + speed, 4)

    def __repr__(self):
        return (
            f"TrainingResult(val_acc={self.val_accuracy:.4f}, "
            f"val_loss={self.val_loss:.4f}, reward={self.reward:.4f})"
        )