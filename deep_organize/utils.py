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
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image


logger = logging.getLogger(__name__)


def generate_result(
    model: torch.nn,
    dim: int = 2,
    device: str = "cpu",
) -> List[Tensor]:
    model.eval()

    result_list = []
    for power in range(5):
        data = torch.rand(1, pow(2, power + 4), dim).to(model.device)
        result = model(data).to("cpu")
        result_list.append(result.detach())

    return result_list


def generate_rectangle_result(
    model: torch.nn,
    dim: int = 2,
    device: str = "cpu",
) -> List[Tensor]:
    model.eval()

    result_list = []
    for power in range(5):
        data = torch.rand(1, pow(2, power + 1), dim * 2).to(model.device)
        result = model(data)
        data[:,:,0:2]=result
        result_list.append(data.detach().to('cpu'))

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
        all_data_list = generate_result(
            model=pl_module,
            dim=self._dim,
        )

        for index, data in enumerate(all_data_list):
            x = data[0, :, 0]
            y = data[0, :, 1]
            plt.scatter(x, y)

            buf = io.BytesIO()
            plt.savefig(
                buf,
                dpi="figure",
                format=None,
                metadata=None,
                bbox_inches='tight',
                pad_inches=0.1,
                facecolor="auto",
                edgecolor="auto",
                backend=None,
            )
            buf.seek(0)
            image = PIL.Image.open(buf)
            image = transforms.ToTensor()(image)
            
            trainer.logger.experiment.add_image(
                f"img{index}",
                torch.tensor(image).detach(),
                global_step=trainer.global_step,
            )
            plt.clf()
        # plt.close()


class RectangleSampler(Callback):
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
        all_data_list = generate_rectangle_result(
            model=pl_module,
            dim=self._dim,
        )

        for index, data in enumerate(all_data_list):
            ax = plt.gca()

            for element in data[0]:
                ax.add_patch(
                    Rectangle(
                        (element[0], element[1]),
                        element[2],
                        element[3],
                        edgecolor="blue",
                        facecolor="none",
                        linewidth=2,
                    )
                )
            ax.axis('scaled')

            buf = io.BytesIO()
            plt.savefig(
                buf,
                dpi="figure",
                format=None,
                metadata=None,
                bbox_inches='tight',
                pad_inches=0.1,
                facecolor="auto",
                edgecolor="auto",
                backend=None,
            )
            buf.seek(0)
            image = PIL.Image.open(buf)
            image = transforms.ToTensor()(image)
            
            trainer.logger.experiment.add_image(
                f"img{index}",
                torch.tensor(image).detach(),
                global_step=trainer.global_step,
            )
            plt.clf()
        # plt.close()
