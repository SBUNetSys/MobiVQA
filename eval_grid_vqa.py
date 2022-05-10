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
from transformers import AutoConfig
from transformers import LxmertTokenizer

from vqa import logger
from vqa.lxmert.modeling_lxmert import LxmertForQuestionAnswering
from vqa.utils import Timer
from vqa.utils import timings
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


@hydra.main(config_path="conf", config_name="eval_grid")
def my_app(cfg) -> None:
    logger.info(cfg)
    logger.info(f"process {os.getpid()}")

    label2answer_file = cfg.label2answer_file
    vqa_answers = json.load(open(label2answer_file))
    model_cfg = cfg.model_cfg

    # todo: use hydra conf, port build model to not depend on detectron2
    frcnn_cfg = setup(model_cfg, cfg.model_weights)
    frcnn = build_model(frcnn_cfg)
    frcnn.eval()
    DetectionCheckpointer(frcnn, save_dir=frcnn_cfg.OUTPUT_DIR).resume_or_load(
        frcnn_cfg.MODEL.WEIGHTS, resume=True
    )

    # frcnn = GeneralizedRCNN.from_pretrained("unc-nlp/frcnn-vg-finetuned",
    #                                         config=frcnn_cfg)
    device = torch.device(frcnn_cfg.MODEL.DEVICE)

    imag_loader = ImageLoader(frcnn_cfg)

    lxmert_tokenizer = LxmertTokenizer.from_pretrained("unc-nlp/lxmert-base-uncased")
    config = AutoConfig.from_pretrained("unc-nlp/lxmert-vqa-uncased")
    config.use_x_layers = cfg.use_x_layers
    lxmert_vqa = LxmertForQuestionAnswering(config)
    if cfg.vqa_model_weights:
        logger.info(f"loading model states from {cfg.vqa_model_weights}")
        model_states = torch.load(cfg.vqa_model_weights, map_location="cpu")
        if 'state_dict' in model_states:
            model_states = model_states['state_dict']
        for key in list(model_states.keys()):
            if "vqa_model." in key:
                logger.info(f"replacing {key}...")
                model_states[key.replace("vqa_model.", "")] = model_states[key]
                del model_states[key]

        lxmert_vqa.load_state_dict(model_states)
    lxmert_vqa.to(device)

    # split = cfg.split
    img_folder = cfg.img_folder

    qa_data = [json.loads(line) for line in open(cfg.vqa_data_file)]
    if cfg.profile_num:
        qa_data = qa_data[: cfg.profile_num]
    total = len(qa_data)

    correct = 0
    count = 0
    val_data = []

    start_time = time.perf_counter()
    prof_info = 'profiling mode' if os.environ.get('profile', False) else ''
    logger.info(f"start inference...{prof_info}")
    num_grids = 0  # 0 means dynamic
    e2e_latency = []
    for qa_item in qa_data:
        iter_start_time = time.perf_counter()
        img_id = qa_item["img_id"]
        q_id = qa_item["question_id"]
        question = qa_item["sent"]
        label = qa_item.get("label", None)

        img_file = os.path.join(img_folder, img_id + ".jpg")
        with Timer("img_prep"):
            images = imag_loader(img_file)
        # images, sizes, scales_yx = image_preprocess(img_file)
        with Timer("img_cnn"), torch.no_grad():
            feat = frcnn.backbone(images.tensor)["res5"]
        # img_int_id = int(img_id.split(".")[0].split("_")[-1])
        # with open(os.path.join(cfg.data_dir, f'{img_int_id}.pth'), "wb") as f:
        #     # save as CPU tensors
        #     torch.save(outputs.cpu(), f)
        b, c, h, w = feat.shape
        img_max_features = cfg.img_max_features
        if img_max_features:
            feat = feat.view(b, c, -1)
            pad_feat = torch.zeros(
                (b, c, img_max_features), device=feat.device, dtype=torch.float
            )
            pad_feat[:, :, : h * w] = feat
            # feat = pad_feat.unsqueeze(-1)
            img_feat = pad_feat.permute(0, 2, 1)
        else:
            img_feat = feat
        num_grids_ratio = cfg.get("num_grids_ratio", 1)
        assert 0 < num_grids_ratio <= 1, f"{num_grids_ratio} must be (0,1]"
        num_grids = int(h * w * num_grids_ratio)
        # logger.info(f'num_grids_ratio={num_grids_ratio}, grids={num_grids}')
        img_feat = img_feat[:, :num_grids, :]
        with Timer("q_tok"):
            inputs = lxmert_tokenizer(
                question,
                padding="max_length",
                max_length=20,
                truncation=True,
                return_token_type_ids=True,
                return_attention_mask=True,
                add_special_tokens=True,
                return_tensors="pt",
            )
            input_ids = inputs.input_ids.pin_memory().to(device, non_blocking=True)
            attention_mask = inputs.attention_mask.pin_memory().to(
                device, non_blocking=True
            )
            token_type_ids = inputs.token_type_ids.pin_memory().to(
                device, non_blocking=True
            )
        with Timer("lxmert_vqa"):
            output_vqa = lxmert_vqa(
                input_ids=input_ids,
                attention_mask=attention_mask,
                visual_feats=img_feat,
                visual_pos=None,
                token_type_ids=token_type_ids,
                output_attentions=False,
            )
        # get prediction
        with Timer("argmax"):
            pred_vqa = output_vqa["question_answering_score"][-1].argmax(-1)
        with Timer("output_cpu"):
            # torch.utils.dlpack.to_dlpack()
            pred_idx = pred_vqa.cpu()
        with Timer("numpy"):
            pred = vqa_answers[pred_idx.numpy()[0]]
        e2e_latency.append((time.perf_counter() - iter_start_time) * 1e3)
        if cfg.print_pred:
            logger.info(f"q{q_id}: {question}\n\t" f"pred: {pred}\n\tlabel: {label}\n")
        val_data.append({"question_id": int(q_id), "answer": pred})
        correct += label.get(pred, 0) if label else 0
        del feat, pad_feat, output_vqa, pred_vqa
        torch.cuda.empty_cache()
        count += 1
        if cfg.profile_num and count == 3:
            # warm up
            timings.clear()
        if count % 10 == 0:
            duration = time.perf_counter() - start_time
            logger.info(
                f"{duration:.3f}s processed: {count}/{total}, "
                f"acc={correct / count * 100:.2f}"
            )
            # logger.info(f'{duration:.3f}s processed: {count}/{total}')
    logger.info(correct / count * 100)
    hostname = socket.gethostname()
    xl = cfg.use_x_layers or 5
    ng = num_grids
    time_str = [f"{hostname}-x{xl}-ng{ng}, num, key, avg, std, min, max"]
    for tk, tv in timings.items():
        time_str.append(
            f"{hostname}-x{xl}-ng{ng}, {len(tv)}, {tk}, "
            f"{np.mean(tv) * 1e3:.3f}, {np.std(tv) * 1e3:.3f}, "
            f"{np.min(tv) * 1e3:.3f}, {np.max(tv) * 1e3:.3f}"
        )
    logger.info("\n".join(time_str))
    logger.info(f"e2e-{hostname}-x{xl}-ng{ng}, {len(e2e_latency)}, "
                f"{np.mean(e2e_latency):.3f}, {np.std(e2e_latency):.3f}, "
                f"{np.min(e2e_latency):.3f}, {np.max(e2e_latency):.3f}")
    with open(cfg.out_file, "w") as outfile:
        json.dump(timings, outfile)

    logger.info("all done")


if __name__ == "__main__":
    my_app()
