import os
from omegaconf import DictConfig, OmegaConf
import hydra
from lightning import Trainer
import matplotlib.pyplot as plt
from lightning.pytorch.callbacks import LearningRateMonitor
from deep_organize.datasets import (
    PointDataModule, RectangleDataModule
)
import logging
from deep_organize.networks import Net
from deep_organize.utils import ImageSampler, RectangleSampler
import torch

logging.basicConfig()
logger = logging.getLogger(__name__)
logging.getLogger().setLevel(logging.DEBUG)


@hydra.main(config_path="../config", config_name="organize")
def run_organize(cfg: DictConfig):
    logger.info(OmegaConf.to_yaml(cfg))
    logger.info(f"Working directory {os.getcwd()}")
    logger.info(f"Orig working directory {hydra.utils.get_original_cwd()}")

    root_dir = hydra.utils.get_original_cwd()

    if cfg.train is True:

        if cfg.network.name == "points" :
            data_module = PointDataModule(
                num_points=cfg.data.num_points,
                num_samples=cfg.data.num_samples,
                batch_size=cfg.batch_size,
                dim=cfg.data.dims,
            )
            lr_monitor = LearningRateMonitor(logging_interval="epoch")
            trainer = Trainer(
                max_epochs=cfg.max_epochs,
                accelerator=cfg.accelerator,
                callbacks=[lr_monitor, ImageSampler(dim=cfg.data.dims, image_size=64)],
            )
        elif cfg.network.name == "boxes" :
            data_module = RectangleDataModule(
                num_rectangles=cfg.data.num_rectangles,
                num_samples=cfg.data.num_samples,
                batch_size=cfg.batch_size,
                dim=2, #cfg.data.dims,
            )
            lr_monitor = LearningRateMonitor(logging_interval="epoch")
            trainer = Trainer(
                max_epochs=cfg.max_epochs,
                accelerator=cfg.accelerator,
                callbacks=[lr_monitor, RectangleSampler(dim=2, image_size=64)]
            )
            
        model = Net(cfg)
        with torch.autograd.set_detect_anomaly(True):
            trainer.fit(model, datamodule=data_module)
        logger.info("testing")

        trainer.test(model, datamodule=data_module)
        logger.info("finished testing")
        logger.info("best check_point", trainer.checkpoint_callback.best_model_path)
    else:
        raise NotImplementedError("The case where Train=False still needs to be implemented")

if __name__ == "__main__":
    run_organize()
