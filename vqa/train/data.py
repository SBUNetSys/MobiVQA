import json
from pathlib import Path
from typing import Optional

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from transformers import AutoTokenizer


class VqaDataset(Dataset):
    def __init__(self, cfg, data_file):
        super().__init__()
        self.cfg = cfg
        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.tokenizer,
                                                       use_fast=True)
        self.data = [json.loads(line) for line in open(data_file)]
        self.img_feat_dir = Path(cfg.img_feat_dir)

        self._init_label_file(cfg)

    def _init_label_file(self, cfg):
        self.ans2label = json.load(open(cfg.ans2label_file))
        # self.label2ans = json.load(open(cfg.label2ans_file))
        # assert len(self.ans2label) == len(self.label2ans)
        self.num_answers = len(self.ans2label)

    def load_img_feat(self, img_id):
        if isinstance(img_id, str):
            if img_id.startswith('COCO'):
                # COCO_val2014_000000000042
                parts = img_id.split('_')
                split = parts[1]
                img_id = int(parts[-1])
            elif img_id.startswith("VizWiz"):
                # VizWiz_val_00000000.jpg
                parts = img_id.split('_')
                split = '-'.join(parts[:2]).lower()
                img_id = int(parts[-1][:-4]) # account for .jpg
            else:
                pass
        else:
            # int, 2367512, vg img_id
            # assert isinstance(img_id, int), f'img_id={img_id} must be int!'
            split = self.cfg.img_feat_name
        feat = torch.load(self.img_feat_dir / split / f'{img_id}.pth')
        b, c, h, w = feat.shape
        img_max_features = self.cfg.img_max_features
        if self.cfg.img_max_features:
            feat = feat.view(b, c, -1)
            pad_feat = torch.zeros((b, c, img_max_features), dtype=torch.float)
            pad_feat[:, :, : h * w] = feat
            # feat = pad_feat.unsqueeze(-1)
            img_feat = pad_feat.squeeze(0).permute(1, 0)
            img_mask = torch.zeros((b, img_max_features), dtype=torch.float)
            img_mask[:, :h * w] = 1.0
        else:
            img_feat = feat
            img_mask = None
        return img_feat, img_mask

    def __getitem__(self, index):
        qa_item = self.data[index]
        img_id = qa_item['img_id']
        q_id = qa_item['question_id']
        question = qa_item['sent']
        inputs = self.tokenizer(
            question,
            padding="max_length",
            max_length=self.cfg.question_max_length,
            truncation=True,
            return_token_type_ids=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt"
        )
        input_ids = inputs.input_ids.squeeze(0)
        attention_mask = inputs.attention_mask.squeeze(0)
        token_type_ids = inputs.token_type_ids.squeeze(0)
        # load processed img features
        img_feat, img_mask = self.load_img_feat(img_id)
        img_mask = img_mask.squeeze(0)
        data_tuple = (q_id, input_ids, attention_mask,
                      token_type_ids, img_feat, img_mask)
        data_tuple = self._process_label(data_tuple, qa_item)
        return data_tuple

    def _process_label(self, data_tuple, qa_item):
        if 'label' in qa_item:
            label = qa_item['label']
            target = torch.zeros(self.num_answers)
            for ans, score in label.items():
                if ans not in self.ans2label:
                    continue
                target[self.ans2label[ans]] = score
            data_tuple += (target,)
        return data_tuple

    def __len__(self) -> int:
        return len(self.data)


class VqaEarlyExitDataset(VqaDataset):

    def __init__(self, cfg, data_file):
        super().__init__(cfg, data_file)
        self.num_exit_layers = cfg.num_exit_layers

    def _init_label_file(self, cfg):
        pass

    def _process_label(self, data_tuple, qa_item):
        if 'ee_layer' in qa_item:
            label = qa_item['ee_layer']
            # target = torch.tensor([label], dtype=torch.float64)
            target = torch.zeros(self.num_exit_layers)
            target[label] = 1
            data_tuple += (target,)
        return data_tuple


class VqaCrossEarlyExitDataset(VqaEarlyExitDataset):

    def _process_label(self, data_tuple, qa_item):
        if 'ee_layer' in qa_item:
            label = qa_item['ee_layer']
            if label == 0:
                label = 1
            # target = torch.zeros(self.num_exit_layers - 1)
            # target[label - 1] = 1
            target = torch.tensor(label - 1)
            data_tuple += (target,)
        return data_tuple


class VqaDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def prepare_data(self, *args, **kwargs):
        pass

    def setup(self, stage: Optional[str] = None):
        if stage == 'fit' or stage is None:
            self.train_data = VqaDataset(self.cfg, self.cfg.train_file)
            self.val_data = VqaDataset(self.cfg, self.cfg.val_file)
        if stage == 'predict' or stage is None:
            self.pred_data = VqaDataset(self.cfg, self.cfg.test_file)

    def train_dataloader(self):
        return DataLoader(self.train_data, shuffle=True,
                          num_workers=self.cfg.data_num_works,
                          drop_last=True, pin_memory=True,
                          batch_size=self.cfg.hparams.train_batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_data, shuffle=False,
                          num_workers=self.cfg.data_num_works,
                          drop_last=False, pin_memory=True,
                          batch_size=self.cfg.val_batch_size)

    def predict_dataloader(self):
        return DataLoader(self.pred_data, shuffle=False,
                          num_workers=self.cfg.data_num_works,
                          drop_last=False, pin_memory=True,
                          batch_size=self.cfg.test_batch_size)


class VqaEarlyExitDataModule(VqaDataModule):
    def __init__(self, cfg):
        super().__init__(cfg)

    def setup(self, stage: Optional[str] = None):
        if stage == 'fit' or stage is None:
            self.train_data = VqaEarlyExitDataset(self.cfg, self.cfg.train_file)
            self.val_data = VqaEarlyExitDataset(self.cfg, self.cfg.val_file)
        if stage == 'predict' or stage is None:
            self.pred_data = VqaEarlyExitDataset(self.cfg, self.cfg.test_file)


class VqaCrossEarlyExitDataModule(VqaDataModule):
    def __init__(self, cfg):
        super().__init__(cfg)

    def setup(self, stage: Optional[str] = None):
        if stage == 'fit' or stage is None:
            self.train_data = VqaCrossEarlyExitDataset(self.cfg,
                                                       self.cfg.train_file)
            self.val_data = VqaCrossEarlyExitDataset(self.cfg,
                                                     self.cfg.val_file)
        if stage == 'predict' or stage is None:
            self.pred_data = VqaCrossEarlyExitDataset(self.cfg,
                                                      self.cfg.test_file)
