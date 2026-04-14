"""Dynamic neural network builder from a config dict."""
import torch
import torch.nn as nn
from typing import Dict, Any, List


def build_model(
    input_dim: int,
    output_dim: int,
    config: Dict[str, Any],
) -> nn.Sequential:
    """
    Build a fully-connected network from a config dict.

    Args:
        input_dim:  Number of input features.
        output_dim: Number of output classes (or 1 for regression).
        config:     Architecture config from search_space.

    Returns:
        An nn.Sequential model ready for training.
    """
    layers: List[nn.Module] = []
    prev_dim = input_dim
    activation_fn = _get_activation(config.get("activation", "relu"))
    use_bn = config.get("batch_norm", False)
    dropout_p = config.get("dropout", 0.0)

    for hidden_dim in config.get("hidden_layers", [128, 64]):
        layers.append(nn.Linear(prev_dim, hidden_dim))
        if use_bn:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(activation_fn)
        if dropout_p > 0:
            layers.append(nn.Dropout(dropout_p))
        prev_dim = hidden_dim

    layers.append(nn.Linear(prev_dim, output_dim))
    return nn.Sequential(*layers)


def _get_activation(name: str) -> nn.Module:
    return {
        "relu":  nn.ReLU(),
        "tanh":  nn.Tanh(),
        "gelu":  nn.GELU(),
        "silu":  nn.SiLU(),
    }.get(name, nn.ReLU())


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)