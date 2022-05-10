#!/usr/bin/env python3
# coding: utf-8

import hydra
import pytorch_lightning as pl
import torch

from vqa import logger
from vqa.train import do_ee_eval
from vqa.train.data import VqaCrossEarlyExitDataModule
from vqa.train.data import VqaEarlyExitDataModule
from vqa.train.vqa import VqaCrossEarlyExitModule
from vqa.train.vqa import VqaEarlyExitModule


@hydra.main(config_path="conf", config_name="early_exit")
def my_app(cfg) -> None:
    logger.info(cfg)
    pl.seed_everything(1)
    if cfg.use_cross_exit:
        dm = VqaCrossEarlyExitDataModule(cfg)
        pl_model = VqaCrossEarlyExitModule(cfg)
    else:
        dm = VqaEarlyExitDataModule(cfg)
        pl_model = VqaEarlyExitModule(cfg)
    pl_model.eval()
    pl_model.freeze()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    pl_model.to(device)
    logger.info(f'using {torch.cuda.device_count()} GPUs')
    pl_model = torch.nn.DataParallel(pl_model)

    do_ee_eval(dm, pl_model, cfg.out_file)
    logger.info("all done")


if __name__ == "__main__":
    my_app()
