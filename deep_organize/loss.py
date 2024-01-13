import torch
from typing import Callable


def rectangle_center(x, dims: int = 2):
    y = x[:, :, :dims]
    y[:, :, 0:dims] = x[:, :, 0:dims] + 0.5 * x[:, :, dims : (2 * dims)]
    return y


class DistanceLoss:
    def __init__(self, target: float = 1):
        self.target = target

    def __call__(self, y):
        dist = torch.cdist(y, y, p=2)
        return torch.pow(torch.sum(dist),0.5) / len(dist)


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
        bd = b[:, i, d].unsqueeze(1)
        am = bd - a[:, :, d]
        ap = (a[:, :, d] + a[:, :, d + dim]) - bd

        tmin = torch.minimum(am, ap)

        dist = torch.clamp(tmin, min=0.0)
        mins.append(dist)

    minimums = torch.stack(mins, dim=2)
    low = torch.min(minimums, dim=2, keepdim=False)[0]

    return low


def check_overlap_2d(a, b, dim):
    b0 = b  # (x,y)

    b1 = b  # (x+w,y)
    b1[:, :, 0] = b[:, :, 0] + b[:, :, dim + 0]

    b2 = b  # (x,y+h)
    b2[:, :, 1] = b[:, :, 1] + b[:, :, dim + 1]

    b3 = b  # (x+w,y+h)
    b3[:, :, 0] = b1[:, :, 0]
    b3[:, :, 1] = b2[:, :, 1]

    res = 0
    accum = 0
    for i in range(b.shape[1]):
        overlap0 = check_point_inside(a, b0, 2, i)
        overlap1 = check_point_inside(a, b1, 2, i)
        overlap2 = check_point_inside(a, b2, 2, i)
        overlap3 = check_point_inside(a, b3, 2, i)

        minimums = torch.cat([overlap0, overlap1, overlap2, overlap3])
        accum = accum + torch.sum(minimums) / minimums.numel()

    return accum


class OverlapLoss:
    def __init__(self, dim: int = 2):
        self.dim = dim

    def __call__(self, y: torch.Tensor, x: torch.Tensor):
        final_tensor = torch.clone(x)
        final_tensor[:, :, 0:2] = y
        # print('final_tensor', final_tensor)
        ans = check_overlap_2d(final_tensor, final_tensor, dim=2)
        # print('res', ans)
        return ans


class RegionalLoss:
    def __init__(self, dim: int = 2, target: float = 0, alpha=0.001):
        self.overlap_loss = OverlapLoss(dim=dim)
        self.distance_loss = DistanceLoss(target=0)
        self.alpha = alpha

    def __call__(self, y: torch.Tensor, x: torch.Tensor) -> None:
        final_tensor = torch.clone(x)
        final_tensor[:, :, 0:2] = y

        loss = self.overlap_loss(y, x)
        
        centers = rectangle_center(x=final_tensor)
        return loss + self.alpha*self.distance_loss(y=centers)