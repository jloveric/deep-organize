import torch
from typing import Callable


def equidistance2d_loss(x: torch.Tensor, target: float = 1.0):
    dist = torch.cdist(x, x, p=2)
    return torch.sqrt(torch.sum(torch.pow(dist - target, 2))/len(dist))


def within_region_loss(
    x: torch.Tensor, distance_function: Callable[[torch.Tensor], float]
):
    pass
