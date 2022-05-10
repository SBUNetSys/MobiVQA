#!/usr/bin/env python3
# coding: utf-8
import json

from sklearn.metrics import classification_report
from tqdm import tqdm

from vqa import logger
from vqa.train.data import VqaEarlyExitDataModule
from vqa.train.io import CheckpointEveryNSteps
from vqa.train.vqa import VqaEarlyExitModule


def do_ee_eval(dm, pl_model, out_file):
    dm.setup('predict')
    predictions = []
    for batch in tqdm(dm.predict_dataloader(), desc='predicting...'):
        output_vqa = pl_model(batch)
        pred = output_vqa['pred']
        q_ids = output_vqa['qid']
        exit_scores = output_vqa['exit_scores']
        if 'labels' in output_vqa:
            labels = output_vqa['labels'] #.argmax(-1)
        else:
            labels = [-1] * len(q_ids)
        for i, qid in enumerate(q_ids):
            pred_idx = pred[i]
            label = labels[i]
            exit_score = exit_scores[i]
            q_pred = {
                "question_id": int(qid),
                "score": [float(s) for s in exit_score],
                "pred": int(pred_idx),
            }
            if label != -1:
                q_pred['label'] = int(label)
            # logits = i_pred[i]
            predictions.append(q_pred)
    with open(out_file, "w") as f:
        json.dump(predictions, f)

    all_pred = [i['pred'] for i in predictions]
    all_labels = [i['label'] for i in predictions]
    logger.info(f'\n{classification_report(all_labels, all_pred)}')
