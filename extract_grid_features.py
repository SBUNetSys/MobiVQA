#!/usr/bin/env python3
# coding: utf-8

import json
import os
import socket
import time

import hydra
import torch
import numpy as np
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import default_setup
from detectron2.modeling import build_model
from vqa import logger
from vqa.utils import Timer
from vqa.vision.processing_image import ImageLoader
from vqa.vision.roi_heads import add_attribute_config


def setup(cfg_file, model_weights=None):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_attribute_config(cfg)
    cfg.merge_from_file(cfg_file)
    # force the final residual block to have dilations 1
    cfg.MODEL.RESNETS.RES5_DILATION = 1
    cfg.MODEL.WEIGHTS = model_weights
    if not torch.cuda.is_available():
        cfg.MODEL.DEVICE = "cpu"
    cfg.freeze()
    default_setup(cfg, None)
    return cfg


@hydra.main(config_path="conf", config_name="extract_grid")
def my_app(cfg) -> None:
    logger.info(cfg)
    logger.info(f"process {os.getpid()}")
    model_cfg = cfg.model_cfg

    # todo: use hydra conf, port build model to not depend on detectron2
    frcnn_cfg = setup(model_cfg, cfg.model_weights)
    frcnn = build_model(frcnn_cfg)
    frcnn.eval()
    DetectionCheckpointer(frcnn, save_dir=frcnn_cfg.OUTPUT_DIR).resume_or_load(
        frcnn_cfg.MODEL.WEIGHTS, resume=True
    )

    # device = torch.device(frcnn_cfg.MODEL.DEVICE)

    imag_loader = ImageLoader(frcnn_cfg)
    img_folder = cfg.img_folder
    os.makedirs(cfg.feature_folder, exist_ok=True)
    qa_data = [json.loads(line) for line in open(cfg.vqa_data_file)]
    total = len(qa_data)
    start_time = time.perf_counter()
    count = 0
    logger.info(f"start extraction...")
    for qa_item in qa_data:
        img_id = qa_item["img_id"]

        img_file = os.path.join(img_folder, img_id) # coco dataset needs to add  + ".jpg"
        with Timer("img_prep"):
            images = imag_loader(img_file)
        # images, sizes, scales_yx = image_preprocess(img_file)
        with Timer("img_cnn"), torch.no_grad():
            feat = frcnn.backbone(images.tensor)["res5"]
        
        # set image feature path
        img_int_id = int(img_id.split(".")[0].split("_")[-1])
        with open(os.path.join(cfg.feature_folder, f'{img_int_id}.pth'), "wb") as f:
            # save as CPU tensors
            torch.save(feat.cpu(), f)
        count += 1
        if count % 10 == 0:
            duration = time.perf_counter() - start_time
            logger.info(f"{duration:.3f}s processed: {count}/{total}")
    duration = time.perf_counter() - start_time
    logger.info("all done in ")


if __name__ == "__main__":
    my_app()
