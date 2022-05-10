#!/usr/bin/env python3
# coding: utf-8
import os
from collections import deque

import pytorch_lightning as pl
from pytorch_lightning.utilities.cloud_io import get_filesystem

from vqa import logger


class CheckpointEveryNSteps(pl.Callback):
    """
    Save a checkpoint every N steps, instead of Lightning's default that checkpoints
    based on validation loss.
    """

    def __init__(
        self,
        save_step_frequency,
        max_keep=3,
        prefix="",
        use_modelcheckpoint_filename=False,
    ):
        """
        Args:
            save_step_frequency: how often to save in steps
            prefix: add a prefix to the name, only used if
                use_modelcheckpoint_filename=False
            use_modelcheckpoint_filename: just use the ModelCheckpoint callback's
                default filename, don't use ours.
        """
        self.save_step_frequency = save_step_frequency
        self.prefix = prefix
        self.use_modelcheckpoint_filename = use_modelcheckpoint_filename
        self.max_keep = max_keep
        self.ckpt_list = deque()

    def on_batch_end(self, trainer: pl.Trainer, _):
        """ Check if we should save a checkpoint after every train batch """
        epoch = trainer.current_epoch
        global_step = trainer.global_step
        dirpath = trainer.checkpoint_callback.dirpath
        _fs = get_filesystem(str(dirpath) if dirpath else "")
        if global_step % self.save_step_frequency == 0:
            if self.use_modelcheckpoint_filename:
                filename = trainer.checkpoint_callback.filename
            else:
                filename = f"{self.prefix}ep{epoch}step{global_step}.ckpt"
            ckpt_path = os.path.join(dirpath, filename)
            if len(self.ckpt_list) == self.max_keep:
                obsolete_ckpt = self.ckpt_list.popleft()
                if _fs.exists(obsolete_ckpt):
                    logger.info(f"delete {obsolete_ckpt=}")
                    _fs.rm(obsolete_ckpt)
                else:
                    logger.info(f"{obsolete_ckpt=} not exist, skip")
            self.ckpt_list.append(ckpt_path)
            logger.info(f"saving {ckpt_path=}")
            trainer.save_checkpoint(ckpt_path, weights_only=True)
