from deep_organize.loss import check_overlap_2d, check_point_inside, OverlapLoss
from deep_organize.datasets import RectangleDataset
import torch


def test_rectangle_dataset():

    dataset = RectangleDataset(num_rectangles=10, num_samples=7, dim=2)
    dataiter = iter(dataset)
    x = next(dataiter).unsqueeze(0)
    x = torch.cat([x, x], dim=0)
    y = next(dataiter)[:,:2].unsqueeze(0) # Other only has corner points
    y = torch.cat([y,y],dim=0)
    print('data', x.shape)
    print('y.shape', y.shape, 'x.shape', x.shape)
    loss = OverlapLoss(dim=2)
    res = loss(y, x)

    assert res > 0.0

def test_overlap_loss() :
    loss = OverlapLoss(dim=2)

    data = torch.ones(1,2,4)
    
    # First rectangle
    data[0,0,0] = 0.5
    data[0,0,1] = 0.5
    data[0,0,2] = 1
    data[0,0,3] = 1
    
    # Second rectangle
    data[0,1,0] = 0.0
    data[0,1,1] = 0.0
    data[0,1,2] = 1
    data[0,1,3] = 1

    print('data', data)

    # The distance should be 0.5 and since it's counted
    # twice (as implemented) result should be 0.1
    y = data[:,:,0:2]
    x= data 

    res = loss(y, x)
    assert res == 0.25
    print('res', res)