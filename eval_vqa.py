#!/usr/bin/env python3
# coding: utf-8

import json
import logging
import os
import time
from collections import defaultdict

import torch
from transformers import LxmertForQuestionAnswering
from transformers import LxmertTokenizer

from vqa.vision.modeling_frcnn import GeneralizedRCNN
from vqa.vision.processing_image import Preprocess
from vqa.utils import Config

logger = logging.getLogger('vqa')

logger.setLevel(logging.INFO)
fmt_str = "%(levelname)s:%(asctime)s.%(msecs)03d: %(message)s"
fmt = logging.Formatter(fmt_str, "%Y-%m-%d_%H:%M:%S")
handler = logging.StreamHandler()
handler.setFormatter(fmt)
logger.addHandler(handler)


def read_data(data_file):
    with open(data_file) as f:
        data = f.read().split("\n")


ROOT_DIR = '/home/qqcao/work/MobiVQA/lxmert'

# OBJ_URL = "data/visual-genome/objects_vocab.txt"
# ATTR_URL = "data/visual-genome/attributes_vocab.txt"
VQA_URL = f"{ROOT_DIR}/data/vqa/trainval_label2ans.json"

# objids = read_data(OBJ_URL)
# attrids = read_data(ATTR_URL)
# gqa_answers = utils.get_data(GQA_URL)
vqa_answers = json.load(open(VQA_URL))

# load models and model components
frcnn_cfg = Config.from_pretrained("unc-nlp/frcnn-vg-finetuned")
frcnn_cfg.model.device = 'cuda'
frcnn = GeneralizedRCNN.from_pretrained("unc-nlp/frcnn-vg-finetuned",
                                        config=frcnn_cfg)
device = torch.device(frcnn_cfg.model.device)

image_preprocess = Preprocess(frcnn_cfg)

lxmert_tokenizer = LxmertTokenizer.from_pretrained(
    "unc-nlp/lxmert-base-uncased")
lxmert_vqa = LxmertForQuestionAnswering.from_pretrained(
    "unc-nlp/lxmert-vqa-uncased")
lxmert_vqa.to(device)

split = 'minival'  # minival, nominival, test, train
# test2015-mscoco-images, val2014-mscoco-images,
img_folder = 'val2014-mscoco-images'

qa_data = json.load(open(f"{ROOT_DIR}/data/vqa/{split}.json"))
total = len(qa_data)

image_qa = defaultdict(list)
for qa_item in qa_data:
    image_qa[qa_item['img_id']].append(qa_item)


def get_img_feat(img_id):
    img_file = os.path.join(ROOT_DIR, 'data', img_folder, img_id + '.jpg')
    images, sizes, scales_yx = image_preprocess(img_file)
    output_dict = frcnn(
        images,
        sizes,
        scales_yx=scales_yx,
        padding="max_detections",
        max_detections=frcnn_cfg.max_detections,
        return_tensors="pt"
    )
    return output_dict


correct = 0
count = 0
val_data = []

start_time = time.perf_counter()
logger.info('start inference...')

for img_id, qa_item_list in image_qa.items():
    img_feat = get_img_feat(img_id)
    normalized_boxes = img_feat.get("normalized_boxes")
    features = img_feat.get("roi_features")
    q_ids = [qa_item['question_id'] for qa_item in qa_item_list]
    questions = [qa_item['sent'] for qa_item in qa_item_list]
    labels = [qa_item['label'] for qa_item in qa_item_list if
              'label' in qa_item]
    inputs = lxmert_tokenizer(
        questions,
        padding="max_length",
        max_length=20,
        truncation=True,
        return_token_type_ids=True,
        return_attention_mask=True,
        add_special_tokens=True,
        return_tensors="pt"
    )
    inputs = inputs.to(device)
    features = features.to(device)
    normalized_boxes = normalized_boxes.to(device)
    output_vqa = lxmert_vqa(
        input_ids=inputs.input_ids,
        attention_mask=inputs.attention_mask,
        visual_feats=features,
        visual_pos=normalized_boxes,
        token_type_ids=inputs.token_type_ids,
        output_attentions=False,
    )
    # get prediction
    pred_vqa = output_vqa["question_answering_score"].argmax(-1)
    for q_id, q, pred_idx, label in zip(q_ids, questions, pred_vqa.tolist(),
                                        labels):
        pred = vqa_answers[pred_idx]
        val_data.append({
            'question_id': q_id,
            'question': q,
            'img_id': img_id,
            'label': label,
            'answer': pred
        })
        #         print("Question:", q)
        #         print("prediction from LXMERT VQA:", pred)
        #         print("Label:", label)
        #         print()
        correct += label.get(pred, 0)
        count += 1
        if count % 100 == 0:
            duration = time.perf_counter() - start_time
            logger.info(f'{duration:.3f}s processed: {count}/{total}, '
                        f'acc={correct / count * 100:.2f}')
            # logger.info(f'{duration:.3f}s processed: {count}/{total}')

logger.info(correct / count * 100)
with open(f'data/{split}_data.json', 'w') as outfile:
    json.dump(val_data, outfile)

logger.info('all done')
