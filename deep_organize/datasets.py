# from PIL import Image
from matplotlib import image
import torch
from torch import Tensor
import numpy as np
import math
import logging
from torch.utils.data import DataLoader, Dataset
from typing import Optional, List
from lightning.pytorch import LightningDataModule
import random

logger = logging.getLogger(__name__)


class PointDataset(Dataset):
    def __init__(self, num_points: int, num_samples: int = 1, dim: int = 2):
        """
        :params num_points: number of points in a sample.
        :params num_samples: number of examples.
        """
        super().__init__()
        # Fixing for now
        self._data = torch.rand(num_samples, num_points, dim)
        self._num_samples = num_samples

    def __len__(self) -> int:
        return self._num_samples*self._num_samples

    def __getitem__(self, idx: int):
        return self._data[idx%self._num_samples]


class PointDataModule(LightningDataModule):
    def __init__(
        self,
        num_points: int = 10,
        num_samples: int = 1,
        num_workers: int = 10,
        dim: int = 2,
        pin_memory: int = True,
        batch_size: int = 32,
        shuffle: bool = True,
    ):
        super().__init__()

        self._num_points = num_points
        self._num_samples = num_samples
        self._num_workers = num_workers
        self._dim = dim
        self._pin_memory = pin_memory
        self._batch_size = batch_size
        self._shuffle = shuffle

    def collate_fn(self, batch) -> tuple[Tensor, Tensor, list[int]]:
        # TODO: this does not make sense to me
        # The max size includes the output

        this_size = random.randint(2, self._num_points - 1)
        final_features = torch.stack([element[:this_size, :] for element in batch])
        return final_features

    def setup(self, stage: Optional[str] = None):
        self._train_dataset = PointDataset(
            num_points=self._num_points, num_samples=self._num_samples, dim=self._dim
        )
        self._test_dataset = PointDataset(
            num_points=self._num_points, num_samples=self._num_samples, dim=self._dim
        )

    @property
    def train_dataset(self) -> Dataset:
        return self._train_dataset

    @property
    def test_dataset(self) -> Dataset:
        return self._test_dataset

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self._train_dataset,
            batch_size=self._batch_size,
            shuffle=self._shuffle,
            pin_memory=self._pin_memory,
            num_workers=self._num_workers,
            drop_last=True,  # Needed for batchnorm
            collate_fn=self.collate_fn,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self._test_dataset,
            batch_size=self._batch_size,
            shuffle=False,
            pin_memory=self._pin_memory,
            num_workers=self._num_workers,
            drop_last=True,
            collate_fn=self.collate_fn,
        )
