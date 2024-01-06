from lightning.pytorch.callbacks import Callback
from lightning.pytorch import Trainer, LightningModule
import torch
from torch import Tensor
import logging
import matplotlib.pyplot as plt
from typing import List
from lightning.pytorch.utilities.rank_zero import rank_zero_only
import numpy as np
import copy
import io
from torchvision import transforms
import PIL

logger = logging.getLogger(__name__)


def generate_result(
    model: torch.nn,
    dim: int = 2,
    device: str = "cpu",
) -> List[Tensor]:
    model.eval()

    result_list = []
    for power in range(5):
        data = torch.rand(1, pow(2, power + 4), dim).to(device)
        result = model(data)
        result_list.append(result)

    return result_list


class ImageSampler(Callback):
    def __init__(
        self,
        dim,
        image_size: int = 64,
    ) -> None:
        super().__init__()
        self._dim = dim
        self._image_size = image_size

    @rank_zero_only
    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        pl_module.eval()
        logger.info("Generating sample")
        all_data_list = generate_result(
            model=pl_module,
            dim=self._dim,
            image_size=self._image_size,
        )

        for data in all_data_list:
            x = data[0, :, 0]
            y = data[0, :, 1]
            plt.scatter(x, y)

            buf = io.BytesIO()
            plt.savefig(
                buf,
                dpi="figure",
                format=None,
                metadata=None,
                bbox_inches=None,
                pad_inches=0.1,
                facecolor="auto",
                edgecolor="auto",
                backend=None,
            )
            buf.seek(0)
            image = PIL.Image.open(buf)
            image = transforms.ToTensor()(image)

            trainer.logger.experiment.add_image(
                "img",
                torch.tensor(image).permute(2, 0, 1),
                global_step=trainer.global_step,
            )
            plt.close()
