#!/usr/bin/env python3
# coding: utf-8
import json

import hydra
import pytorch_lightning as pl
import torch
from pathlib import Path
from tqdm import tqdm

from vqa import logger
from vqa.train.data import VqaDataModule
from vqa.train.metrics import compute_vqa_accuracy
from vqa.train.vqa import VqaLightningModule


@hydra.main(config_path="conf", config_name="finetune_grid")
def my_app(cfg) -> None:
    logger.info(cfg)
    pl.seed_everything(1)
    dm = VqaDataModule(cfg)
    pl_model = VqaLightningModule(cfg)
    pl_model.eval()
    # model = VqaLightningModule.load_from_checkpoint(
    #     checkpoint_path=cfg.resume_from_checkpoint,
    #     hparams_file=cfg.hparams_file, cfg=cfg)
    pl_model.freeze()
    if torch.cuda.device_count() > 1:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        pl_model.to(device)
        logger.info(f'using {torch.cuda.device_count()} GPUs')
        pl_model = torch.nn.DataParallel(pl_model)

    dm.setup('predict')
    label2ans = json.load(open(cfg.label2ans_file))
    use_x_layers = cfg.use_x_layers
    predictions = [[] for _ in range(use_x_layers)]
    out_dir = Path(cfg.out_file).parent
    feat_dir = out_dir / cfg.save_feature_path
    feat_dir.mkdir(exist_ok=True)
    for batch in tqdm(dm.predict_dataloader(), desc='predicting...'):
        output_vqa = pl_model(batch)
        q_ids = output_vqa['qid']
        qa_scores = output_vqa["qa_scores"]
        cross_attn = output_vqa["cross_encoder_attentions"] 
        for layer, i_pred in enumerate(qa_scores):
            preds = i_pred.argmax(-1)
            layer_attn = cross_attn[layer]
            for i, qid in enumerate(q_ids):
                pred_idx = preds[i]
                pred_ans = label2ans[pred_idx]
                predictions[layer].append({
                    "question_id": int(qid),
                    "pred": int(pred_idx),
                    # 'logits': [float(logit) for logit in logits],
                    "answer": pred_ans,
                })
                if cfg.save_feature_path:
                    logits = i_pred[i]
                    ex_attn = layer_attn[i]
                    feat_file = feat_dir / f'{int(qid)}.th'
                    with open(feat_file, "wb") as f:
                        # save as CPU tensors
                        save = logits.cpu(), ex_attn.cpu()
                        torch.save(save, f)

    with open(cfg.out_file, "w") as f:
        json.dump(predictions, f)

    if cfg.local_eval:
        gt_file = cfg.test_file
        gt_data = [json.loads(line) for line in open(gt_file)]
        for layer, layer_predictions in enumerate(predictions):
            accuracy = compute_vqa_accuracy(gt_data, layer_predictions)
            logger.info(f"{layer=},{accuracy=:.2f}")
    logger.info("all done")


if __name__ == "__main__":
    my_app()
