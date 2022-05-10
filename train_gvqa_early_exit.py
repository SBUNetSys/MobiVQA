#!/usr/bin/env python3
# coding: utf-8

import hydra
import pytorch_lightning as pl
import torch

from vqa import logger
from vqa.train import do_ee_eval
from vqa.train.data import VqaCrossEarlyExitDataModule
from vqa.train.data import VqaEarlyExitDataModule
from vqa.train.io import CheckpointEveryNSteps
from vqa.train.vqa import VqaCrossEarlyExitModule
from vqa.train.vqa import VqaEarlyExitModule


@hydra.main(config_path="conf", config_name="early_exit")
def my_app(cfg) -> None:
    logger.info(cfg)
    pl.seed_everything(1)
    if cfg.use_cross_exit:
        dm = VqaCrossEarlyExitDataModule(cfg)
        model = VqaCrossEarlyExitModule(cfg)
        prefix = 'gvqa-cross-ee-'
    else:
        dm = VqaEarlyExitDataModule(cfg)
        model = VqaEarlyExitModule(cfg)
        prefix = 'gvqa-ee-'

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor="val_acc",
        dirpath=cfg.weights_save_path,
        save_weights_only=cfg.save_weights_only,
        filename=prefix + "{epoch:02d}-{val_acc:.4f}",
        save_top_k=3,
        mode="max",
    )
    trainer = pl.Trainer(
        callbacks=[
            checkpoint_callback,
            CheckpointEveryNSteps(
                cfg.save_step_frequency,
                cfg.save_step_max_keep,
            ),
        ],
        gpus=cfg.num_gpus,
        precision=cfg.hparams.precision,
        amp_level=cfg.hparams.amp_level,
        stochastic_weight_avg=cfg.hparams.stochastic_weight_avg,
        accelerator="dp",
        deterministic=True,
        log_every_n_steps=cfg.log_every_n_steps,
        max_epochs=cfg.hparams.max_epochs,
        overfit_batches=cfg.overfit_batches,
        resume_from_checkpoint=cfg.resume_from_checkpoint,
        gradient_clip_val=cfg.hparams.gradient_clip_val,
    )
    trainer.fit(model, datamodule=dm)
    model.eval()
    model.freeze()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    logger.info(f'using {torch.cuda.device_count()} GPUs')
    model = torch.nn.DataParallel(model)
    do_ee_eval(dm, model, cfg.out_file)
    logger.info("all done")


if __name__ == "__main__":
    my_app()
