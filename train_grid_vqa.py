#!/usr/bin/env python3
# coding: utf-8

import hydra
import pytorch_lightning as pl
from omegaconf import OmegaConf

from vqa import logger
from vqa.train.data import VqaDataModule
from vqa.train.io import CheckpointEveryNSteps
from vqa.train.vqa import VqaLightningModule


@hydra.main(config_path="conf", config_name="finetune_grid")
def my_app(cfg) -> None:
    logger.info(OmegaConf.to_yaml(cfg, resolve=True))
    pl.seed_everything(1)
    dm = VqaDataModule(cfg)
    model = VqaLightningModule(cfg)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor="val_acc",
        dirpath=cfg.weights_save_path,
        save_weights_only=cfg.save_weights_only,
        filename="grid-vqa-{epoch:02d}-{val_acc:.4f}",
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
    logger.info("all done")


if __name__ == "__main__":
    my_app()
