from deep_organize.loss import check_overlap_2d, check_point_inside, overlap_loss
from deep_organize.datasets import RectangleDataset
import torch


def test():

    dataset = RectangleDataset(num_rectangles=10, num_samples=7, dim=2)
    dataiter = iter(dataset)
    x = next(dataiter).unsqueeze(0)
    y = next(dataiter)[:,:2].unsqueeze(0) # Other only has corner points
    print('data', x.shape)

    res = overlap_loss(x, y,dim=2)
