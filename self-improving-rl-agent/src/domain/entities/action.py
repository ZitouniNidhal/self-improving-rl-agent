"""Action space definition for the AutoML RL agent."""
from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Dict


class ActionType(IntEnum):
    # Architecture choices
    ADD_LAYER      = 0
    REMOVE_LAYER   = 1
    CHANGE_ACT_FN  = 2
    # Optimizer choices
    SET_LR_HIGH    = 3
    SET_LR_MED     = 4
    SET_LR_LOW     = 5
    CHANGE_OPT     = 6
    # Regularization
    ADD_DROPOUT    = 7
    REMOVE_DROPOUT = 8
    ADD_BN         = 9
    # Training
    INCREASE_BATCH = 10
    DECREASE_BATCH = 11
    TRAIN_MORE     = 12
    # Terminal
    SUBMIT         = 13


ACTION_SPACE_SIZE = len(ActionType)


@dataclass
class Action:
    action_type: ActionType
    params: Dict[str, Any] = None

    def __post_init__(self):
        if self.params is None:
            self.params = {}

    @classmethod
    def from_index(cls, idx: int) -> "Action":
        return cls(action_type=ActionType(idx))

    def __repr__(self):
        return f"Action({self.action_type.name}, params={self.params})"