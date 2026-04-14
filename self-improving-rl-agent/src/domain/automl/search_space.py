"""Search space — maps discrete actions to concrete architecture configs."""
from typing import List, Dict, Any
from src.domain.entities.action import ActionType


# ── Default architecture config ─────────────────────────────────────────────

DEFAULT_CONFIG: Dict[str, Any] = {
    "hidden_layers": [128, 64],
    "activation": "relu",
    "dropout": 0.0,
    "batch_norm": False,
    "optimizer": "adam",
    "lr": 1e-3,
    "batch_size": 64,
    "epochs": 10,
}

ACTIVATIONS      = ["relu", "tanh", "gelu", "silu"]
OPTIMIZERS       = ["adam", "sgd", "rmsprop", "adamw"]
LAYER_SIZES      = [32, 64, 128, 256, 512]


def apply_action(config: Dict[str, Any], action_type: ActionType) -> Dict[str, Any]:
    """
    Apply an action to a config dict and return the *new* config.
    Pure function — does not mutate the input.
    """
    cfg = {**config, "hidden_layers": list(config["hidden_layers"])}

    match action_type:
        case ActionType.ADD_LAYER:
            cfg["hidden_layers"].append(64)

        case ActionType.REMOVE_LAYER:
            if len(cfg["hidden_layers"]) > 1:
                cfg["hidden_layers"].pop()

        case ActionType.CHANGE_ACT_FN:
            idx = (ACTIVATIONS.index(cfg["activation"]) + 1) % len(ACTIVATIONS)
            cfg["activation"] = ACTIVATIONS[idx]

        case ActionType.SET_LR_HIGH:
            cfg["lr"] = 1e-2
        case ActionType.SET_LR_MED:
            cfg["lr"] = 1e-3
        case ActionType.SET_LR_LOW:
            cfg["lr"] = 1e-4

        case ActionType.CHANGE_OPT:
            idx = (OPTIMIZERS.index(cfg["optimizer"]) + 1) % len(OPTIMIZERS)
            cfg["optimizer"] = OPTIMIZERS[idx]

        case ActionType.ADD_DROPOUT:
            cfg["dropout"] = min(cfg["dropout"] + 0.1, 0.5)
        case ActionType.REMOVE_DROPOUT:
            cfg["dropout"] = max(cfg["dropout"] - 0.1, 0.0)

        case ActionType.ADD_BN:
            cfg["batch_norm"] = True

        case ActionType.INCREASE_BATCH:
            cfg["batch_size"] = min(cfg["batch_size"] * 2, 512)
        case ActionType.DECREASE_BATCH:
            cfg["batch_size"] = max(cfg["batch_size"] // 2, 16)

        case ActionType.TRAIN_MORE:
            cfg["epochs"] = cfg["epochs"] + 5

        case ActionType.SUBMIT:
            pass   # terminal — no change

    return cfg