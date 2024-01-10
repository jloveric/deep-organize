import torch
from typing import Callable


def equidistance2d_loss(x: torch.Tensor, input: torch.Tensor, target: float = 1.0):
    dist = torch.cdist(x, x, p=2)
    return torch.sum(torch.pow(dist - target, 2)) / len(dist)


def check_point_inside(a, b, dim, i):
    """
    Compute the minmums, and compute the distance to the nearest
    edge if interior. If exterior then return 0 as the minimum.
    TODO: Search could stop early the first 0 or negative observed
    in each dimension.
    """
    res = True
    mins = []
    for d in range(dim):
        am = b[:, i, d] - a[:, :, d]
        ap = a[:, :, d] + a[:, :, d + dim] - b[:, i, d]
        dist = torch.clamp(torch.minimum(am, ap), min=0.0)
        mins.append(dist)

    minimums = torch.cat(mins, dim=0)
    low = torch.min(minimums, dim=0, keepdim=True)[0]
    return low


def check_overlap_2d(a, b, dim):
    b0 = b

    b1 = b
    b1[:, :, 0] = b[:, :, 0] + b[:, :, dim + 0]

    b2 = b
    b2[:, :, 1] = b[:, :, 1] + b[:, :, dim + 1]

    b3 = b
    b3[:, :, 0] = b1[:, :, dim + 0]
    b3[:, :, 1] = b2[:, :, dim + 1]

    res = 0
    for i in range(b.shape[1]):
        overlap0 = check_point_inside(a, b0, 2, i)
        overlap1 = check_point_inside(a, b1, 2, i)
        overlap2 = check_point_inside(a, b2, 2, i)
        overlap3 = check_point_inside(a, b3, 2, i)

        minimums = torch.cat([overlap0, overlap1, overlap2, overlap3])
        res += torch.sum(minimums)

    return res


def overlap_loss(x: torch.Tensor, y: torch.Tensor, dim: int):
    final_tensor = x
    final_tensor[:, :, 0:dim] = y

    return check_overlap_2d(final_tensor, final_tensor, dim=2)
