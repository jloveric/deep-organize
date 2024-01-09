from deep_organize.loss import check_overlap_2d, check_point_inside, overlap_loss
from deep_organize.datasets import RectangleDataset
import torch


def test():

    dataset = RectangleDataset(num_rectangles=10, num_samples=7, dim=2)
    dataiter = iter(dataset)
    data = next(dataiter).unsqueeze(0)
    other = next(dataiter)[:,:2].unsqueeze(0) # Other only has corner points
    print('data', data.shape)

    res = overlap_loss(data, other,dim=2)
