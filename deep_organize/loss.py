import torch
from typing import Callable


def equidistance2d_loss(x: torch.Tensor, input: torch.Tensor, target: float = 1.0):
    dist = torch.cdist(x, x, p=2)
    return torch.sum(torch.pow(dist - target, 2)) / len(dist)


def check_point_inside(a, b, dim):
    """
    Compute the minmums, and compute the distance to the nearest
    edge if interior. If exterior then return 0 as the minimum.
    TODO: Search could stop early the first 0 or negative observed
    in each dimension.
    """
    res = True
    mins = []
    for d in range(dim):
        am = b[:, :, d] - a[:,:,d]
        ap = a[:, :, d] + a[:, :, d + dim] - b[:,:,d]
        dist = torch.clamp(torch.minimum(am,ap), min=0.0)
        mins.append(dist)

    minimums = torch.cat(mins,dim=2)
    res = torch.clamp(torch.min(minimums, dim=2), min=0.0)

    return res


def check_overlap2d(a, b, dim):
    b0 = b

    b1 = b
    b1[:, :, 0] = b[:, :, 0] + b[:, :, dim + 0]

    b2 = b
    b2[:, :, 1] = b[:, :, 1] + b[:, :, dim + 1]

    b3 = b
    b3[:, :, 0] = b1[:, :, dim + 0]
    b3[:, :, 1] = b2[:, :, dim + 1]

    overlap0 = check_point_inside(a, b0)
    overlap1 = check_point_inside(a, b1)
    overlap2 = check_point_inside(a, b2)
    overlap3 = check_point_inside(a, b3)

    minimums = torch.cat([overlap0, overlap1, overlap2, overlap3])
    res = torch.clamp(torch.min(minimums, dim=2.0), min=0.0)


def overlap_loss(
    x: torch.Tensor,
    input: torch.Tensor,
    dim
):
    final_tensor = input
    final_tensor[:,:,0:dim] = x

    
