import torch
from torch import nn


def linear_channel_first(x: torch.Tensor, linear: nn.Linear) -> torch.Tensor:
    """
    Apply nn.Linear over the channel axis of a [B, C, T] tensor without transpose.
    """
    y = torch.matmul(linear.weight, x)
    if linear.bias is not None:
        y = y + linear.bias.view(1, -1, 1)
    return y
