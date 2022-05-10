import copy
import json
import logging
import re
from dataclasses import dataclass
from typing import Any
from typing import List
from typing import Optional

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam
from torchmetrics import Accuracy
from transformers import AdamW
from transformers import AutoConfig
from transformers import get_linear_schedule_with_warmup
from transformers.activations import get_activation
from transformers.file_utils import ModelOutput

from vqa.lxmert.modeling_lxmert import LxmertForQuestionAnswering
from vqa.train.metrics import VqaAccuracy

log = logging.getLogger(__name__)

def map_ckpt_key(ckpt_key):
    # answer_head.logit_fc.0.weight -> answer_heads.0.logit_fc.0.weight
    parts = ckpt_key.split('.')
    return '.'.join(['answer_heads', ])

class VqaLightningModule(pl.LightningModule):

    def _init_vqa(self, cfg):
        self.ee_model = None
        # metric = VqaAccuracy()
        # self.train_acc = metric.clone()
        # self.val_acc = metric.clone()
        self.train_acc = nn.ModuleList([VqaAccuracy()
                                        for _ in range(self.num_ans_heads)])
        self.val_acc = nn.ModuleList([VqaAccuracy()
                                      for _ in range(self.num_ans_heads)])
        # freeze_last = cfg.finetune_heads_only_except_last
        # if cfg.finetune_heads_only:
        #     for param_name, param in self.vqa_model.named_parameters():
        #         if param_name.startswith('answer_heads'):
        #             prefix = f'answer_heads.{self.x_layers - 1}'
        #             if freeze_last and param_name.startswith(prefix):
        #                 # answer_heads.4 for x_layers=5
        #                 param.requires_grad = False
        #                 log.info(f'freeze answer_heads grad: {param_name}')
        #             else:
        #                 param.requires_grad = True
        #                 log.info(f'enable answer_heads grad: {param_name}')
        #         else:
        #             param.requires_grad = False
        #             log.info(f'freeze param: {param_name}')

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters(cfg.hparams)
        model_config = cfg.model_config
        config = AutoConfig.from_pretrained(model_config)
        config.update(cfg)
        # config.num_qa_labels = cfg.num_qa_labels
        # config.x_layers = cfg.x_layers
        # self.x_layers = cfg.x_layers
        self.num_ans_heads = 1 if cfg.single_ans_head else cfg.x_layers 
        self.vqa_model = LxmertForQuestionAnswering(config)
        self._init_vqa(cfg)
        self.init_states(cfg)
        self.load_states(cfg)

    def init_states(self, cfg):
        if cfg.model_states:
            vqa_states = self.vqa_model.state_dict()
            head_weights = {}
            head_new_weights = {}
            log.info(f'initializing states from {cfg.model_states}')
            model_states = torch.load(cfg.model_states,
                                      map_location="cpu")
            if 'state_dict' in model_states:
                model_states = model_states['state_dict']
            for key in list(model_states.keys()):
                # remove suffix
                suffix = 'vqa_model.'
                if key.startswith(suffix):
                    new_key = key[len(suffix):]
                    if cfg.print_model_info:
                        log.info(f'replacing {key} with {new_key}, '
                                 f'shape: {model_states[key].shape}')
                    model_states[new_key] = model_states[key]
                    model_states.pop(key)
                    key = new_key

                if 'box_' in key:  # pop box features weights
                    model_states.pop(key)

                if 'answer_head' in key:
                    if cfg.reuse_heads:
                        if cfg.single_ans_head:
                            x_layers = [0]
                        else:
                            x_layers = range(self.num_ans_heads)
                        for x_layer in x_layers:
                            new_key = key.replace('answer_head',
                                                  f'answer_heads.{x_layer}')
                            if 'logit_fc.3' in key:
                                continue
                            if cfg.print_model_info:
                                log.info(f'reuse_heads {new_key} with {key}')
                            model_states[new_key] = copy.deepcopy(
                                model_states[key])
                            if cfg.load_vg_head and 'logit_fc.3' in key:
                                head_weights[new_key] = copy.deepcopy(vqa_states[new_key])
                                head_new_weights[new_key] = model_states[key]
                    if cfg.pop_ans_head:
                        # Do not load any answer head
                        log.info(f'popping answer head: {key}')
                        model_states.pop(key)

                # handle x_layers weights
                m = re.search(r'x_layers\.(\d)', key)
                if m:
                    x_layer = int(m.group(1))
                    if x_layer >= cfg.x_layers:
                        log.info(f'popping x_layer weights:{key}')
                        model_states.pop(key)
            if cfg.load_vg_head:
                vg_ans = json.load(open(cfg.vg_ans2label))
                vqa_labels = json.load(open(cfg.label2ans_file))
                log.info(f'load answer head weights '
                         f'labels from {cfg.vg_ans2label}={len(vg_ans)} '
                         f'and {cfg.label2ans_file}={len(vqa_labels)}')
                # do surgery, use vg common label classifier head
                for key, h_w in head_weights.items():
                    new_w = head_new_weights[key]
                    log.info(f'new_w={new_w.shape} h_w={h_w.shape}')
                    loaded = 0
                    unload = 0
                    # replace h_w row with corresponding new_w
                    for i, vqa_label in enumerate(vqa_labels):
                        if vqa_label in vg_ans:
                            vg_idx = vg_ans[vqa_label]
                            h_w[i] = new_w[vg_idx]
                            loaded += 1
                        else:
                            unload += 1
                    # set h_w
                    model_states[key] = copy.deepcopy(h_w)
                    log.info(f'{loaded=}, {unload=} for {key}')

            self.vqa_model.load_state_dict(model_states, strict=False)

            if cfg.print_model_info:
                keys = '\n'.join([k for k in self.state_dict().keys()])
                log.info(f'model keys: {keys}')

    def load_states(self, cfg):
        if cfg.load_states:
            log.info(f'loading states from {cfg.load_states}')
            loaded_states = torch.load(cfg.load_states, map_location="cpu")
            if 'state_dict' in loaded_states:
                loaded_states = loaded_states['state_dict']
            self.load_state_dict(loaded_states)

    def forward(self, batch, *args, **kwargs):
        q_ids = batch[0]
        outputs = self._shared_step(batch)
        if 'qid' not in outputs:
            outputs['qid'] = q_ids
        return outputs

    def _share_forward(self, batch):
        if len(batch) == 7:  # train example
            q_ids, input_ids, q_mask, tok_ids, img, img_mask, targets = batch
        else:
            assert len(batch) == 6, f'{len(batch)} must be 6 for test examples!'
            targets = None
            q_ids, input_ids, q_mask, tok_ids, img, img_mask = batch
        output_vqa = self.vqa_model(
            input_ids=input_ids,
            attention_mask=q_mask,
            visual_feats=img,
            labels=targets,
            visual_pos=None,
            token_type_ids=tok_ids,
            output_attentions=True,
            output_hidden_states=True,
        )
        return output_vqa, targets, (q_mask, img_mask)

    def _shared_step(self, batch):
        output_vqa, targets, _ = self._share_forward(batch)
        qa_scores = output_vqa["question_answering_score"]
        losses = output_vqa.get('loss', [])
        count = 0
        loss = 0
        results = {}
        for layer, i_pred in enumerate(qa_scores):
            results[f'preds_{layer}'] = i_pred.argmax(-1)

        for layer, i_loss in enumerate(losses):
            results[f'loss_{layer}'] = i_loss
            loss += (layer + 1) * i_loss
            count += (layer + 1)
        if count > 0:
            results['loss'] = loss / count
        results['labels'] = targets
        results['qa_scores'] = qa_scores
        results['cross_encoder_attentions'] = output_vqa['cross_encoder_attentions']
        return results

    def training_step(self, batch, *args, **kwargs):
        outputs = self._shared_step(batch)
        return outputs

    def training_step_end(self, step_outputs):
        # self.train_acc(step_outputs['preds'], step_outputs['labels'])
        # self.log('train_acc', self.train_acc, prog_bar=True)
        for layer in range(self.num_ans_heads):
            pred = step_outputs[f'preds_{layer}']
            self.train_acc[layer](pred, step_outputs['labels'])
            self.log(f'train_acc_{layer}',
                     self.train_acc[layer],
                     on_step=True,
                     prog_bar=False)
            self.log(f'train_loss_{layer}',
                     step_outputs[f'loss_{layer}'].sum(),
                     on_step=True,
                     prog_bar=False)
        return step_outputs['loss'].sum()

    def training_epoch_end(self, outputs: List[Any]) -> None:
        loss = torch.stack([x['loss'] for x in outputs]).mean()
        # self.log('train_loss', outputs['loss'], prog_bar=True)
        self.log('train_loss', loss, prog_bar=False)
        # self.log('train_acc', self.train_acc)
        for layer in range(self.num_ans_heads):
            self.log(f'train_acc_{layer}', self.train_acc[layer], prog_bar=True)

    def validation_step(self, batch, *args, **kwargs):
        outputs = self._shared_step(batch)
        return outputs

    def validation_step_end(self, step_outputs):
        # self.val_acc(step_outputs['preds'], step_outputs['labels'])
        # self.log('val_acc', self.val_acc)
        for layer in range(self.num_ans_heads - 1):
            pred = step_outputs[f'preds_{layer}']
            self.val_acc[layer](pred, step_outputs['labels'])
            # self.log(f'val_acc_{layer}', self.val_acc[layer], prog_bar=True)

        pred = step_outputs[f'preds_{self.num_ans_heads - 1}']
        self.val_acc[-1](pred, step_outputs['labels'])
        # self.log(f'val_acc', self.val_acc[-1], prog_bar=True)
        # self.log(f'val_loss', step_outputs['loss'].sum(), prog_bar=True)
        return step_outputs['loss'].sum()

    # def _shared_epoch_end(self, outputs):
    #     preds = torch.cat([x['preds'] for x in outputs]).numpy()
    #     labels = torch.cat([x['labels'] for x in outputs]).numpy()
    #     loss = torch.stack([x['loss'] for x in outputs]).mean()
    #     accuracy = self.compute_accuracy(preds, labels)
    #     return {'loss': loss, "accuracy": accuracy}

    def validation_epoch_end(self, outputs: List[Any]) -> None:
        # outputs = self._shared_epoch_end(outputs)
        loss = torch.stack(outputs).mean()
        self.log('val_loss', loss)
        # self.log('val_acc', self.val_acc)
        for layer in range(self.num_ans_heads - 1):
            self.log(f'val_acc_{layer}', self.val_acc[layer], prog_bar=True)
        self.log(f'val_acc', self.val_acc[-1], prog_bar=True)

    def setup(self, stage):
        if stage == 'fit':
            # Get dataloader by calling it - train_dataloader()
            # is called after setup() by default
            train_loader = self.train_dataloader()
            # Calculate total steps
            ds = len(train_loader.dataset)
            bs = self.hparams.train_batch_size
            ng = self.cfg.num_gpus
            ep = float(self.hparams.max_epochs)
            ag = self.hparams.accumulate_grad_batches
            self.total_steps = ((ds // (bs * max(1, ng))) // ag * ep)

    def get_progress_bar_dict(self):
        tqdm_dict = super().get_progress_bar_dict()
        tqdm_dict.pop("v_num", None)
        return tqdm_dict

    def _get_opt_model(self):
        return self.vqa_model

    def configure_optimizers(self):
        "Prepare optimizer and schedule (linear warmup and decay)"
        model = self._get_opt_model()
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if
                           not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if
                           any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters,
                          lr=self.hparams.learning_rate)
        warmup_steps = self.total_steps * self.hparams.warmup_ratio
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps,
            num_training_steps=self.total_steps
        )
        scheduler = {
            'scheduler': scheduler,
            'interval': 'step',
            'frequency': 1
        }
        return [optimizer], [scheduler]


class VqaEarlyExitModule(VqaLightningModule):

    def _init_vqa(self, cfg):
        self.train_acc = Accuracy()
        self.val_acc = Accuracy()
        # freeze vqa model parameters, only train early exiting model
        for param_name, param in self.vqa_model.named_parameters():
            param.requires_grad = False
            if cfg.print_model_info:
                log.info(f'freeze grad: {param_name}')

        self.ee_model = ExitModel(cfg)

    def _shared_step(self, batch):
        if len(batch) == 7:  # train example
            q_ids, input_ids, q_mask, tok_ids, img, img_mask, targets = batch
        else:
            assert len(batch) == 6, f'{len(batch)} must be 6 for test examples!'
            targets = None
            q_ids, input_ids, q_mask, tok_ids, img, img_mask = batch
        output_vqa = self.vqa_model(
            input_ids=input_ids,
            attention_mask=q_mask,
            visual_feats=img,
            labels=None, # notice we don't pass labels to vqa_model
            visual_pos=None,
            token_type_ids=tok_ids,
            output_attentions=True,
            output_hidden_states=True,
        )
        # output_vqa, targets, masks = self._share_forward(batch)
        # q_mask, img_mask = masks
        last_l_layer = self.cfg.lxmert.l_layer - 1
        qh = output_vqa['language_hidden_states'][last_l_layer]  # :-1]
        last_v_layer = self.cfg.lxmert.v_layer - 1
        img_h = output_vqa['vision_hidden_states'][last_v_layer]  # :-1]
        ee_outputs = self.ee_model(qh, img_h, q_mask, img_mask, labels=targets)
        exit_scores = ee_outputs['exit_scores']
        pred = exit_scores.argmax(-1)
        loss = ee_outputs.get('loss', None)
        results = {
            'exit_scores': exit_scores,
            'pred': pred,
        }
        if loss is not None:
            results['loss'] = loss
        if targets is not None:
            labels = targets.argmax(-1)
            results['labels'] = labels
        # log.info(f"results: {results}")
        return results

    def training_step_end(self, step_outputs):
        self.train_acc(step_outputs['pred'], step_outputs['labels'])
        self.log(f'train_acc', self.train_acc, on_step=True, prog_bar=False)
        self.log(f'train_loss', step_outputs[f'loss'].sum(),
                 on_step=True, prog_bar=False)
        return step_outputs['loss'].sum()

    def training_epoch_end(self, outputs: List[Any]) -> None:
        loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.log('train_loss', loss, prog_bar=False)
        self.log(f'train_acc', self.train_acc, prog_bar=True)

    def validation_step_end(self, step_outputs):
        # log.info(f"step_outputs: {step_outputs['pred']}, {step_outputs['labels']}")
        self.val_acc(step_outputs['pred'], step_outputs['labels'])
        return step_outputs['loss'].sum()

    def validation_epoch_end(self, outputs: List[Any]) -> None:
        self.log('val_loss', torch.stack(outputs).mean())
        self.log(f'val_acc', self.val_acc, prog_bar=True)

    def _get_opt_model(self):
        return self.ee_model

    # def configure_optimizers(self):
    #    parameters = [p for p in self.ee_model.parameters() if p.requires_grad]
    #     optimizer = Adam(parameters, lr=self.hparams.learning_rate)
    #     return optimizer


class VqaCrossEarlyExitModule(VqaEarlyExitModule):

    def _init_vqa(self, cfg):
        self.train_acc = VqaAccuracy()
        self.val_acc = VqaAccuracy()
        # freeze vqa model parameters, only train early exiting model
        for param_name, param in self.vqa_model.named_parameters():
            param.requires_grad = False
            if cfg.print_model_info:
                log.info(f'freeze grad: {param_name}')

        self.ee_model = CrossExitModel(cfg)

    def _shared_step(self, batch):
        output_vqa, targets, _ = self._share_forward(batch)
        # last_l_layer = self.cfg.lxmert.l_layer - 1
        qa_scores = output_vqa['question_answering_score']
        # pool_states = output_vqa['pooled_outputs']  # [last_l_layer: -1]
        states = torch.stack(qa_scores, dim=0)
        states = states.permute(1, 0, 2)
        states = torch.softmax(states, -1)  # normalize to [0,1]
        new_targets = []
        new_states = []
        all_ids = []
        q_ids = batch[0]
        for t, state, qid in zip(targets.unbind(), states.unbind(),
                                 q_ids.unbind()):
            new_s = state[:t + 1]
            new_states.append(new_s)
            new_t = torch.zeros(t + 1, 2, device=targets.device)
            new_t[:t, 0] = 1
            new_t[t, 1] = 1
            new_targets.append(new_t)
            all_ids.append(torch.stack([qid] * (t + 1)))
        new_targets = torch.cat(new_targets, 0)
        states = torch.cat(new_states, 0)
        all_ids = torch.cat(all_ids, 0)
        ee_outputs = self.ee_model(states, labels=new_targets)
        exit_scores = ee_outputs['exit_scores']
        pred = exit_scores.argmax(-1)
        loss = ee_outputs['loss']
        results = {
            'exit_scores': exit_scores,
            'pred': pred,
            'qid': all_ids,
            'loss': loss,
            'labels': new_targets,
        }
        return results


def weighted_avg(x, weights):
    """Return a weighted average of x (a sequence of vectors).
    Args:
        x: batch * len * hdim
        weights: batch * len, sum(dim = 1) = 1
    Output:
        x_avg: batch * hdim
    """
    return weights.unsqueeze(1).bmm(x).squeeze(1)


def weighted_pool(linear, q, q_mask):
    q_flat = q.view(-1, q.size(-1))
    q_scores = linear(q_flat).view(q.size(0), q.size(1))
    if q_mask is not None:
        q_scores += (1.0 - q_mask) * -1e6
    q_alpha = F.softmax(q_scores, dim=-1)
    weighted_q = weighted_avg(q, q_alpha)
    return weighted_q


class ExitModel(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        h_dim = cfg.lxmert.hidden_size
        num_exit_layers = cfg.num_exit_layers
        # ffn_size = cfg.early_exit.ffn_size
        # ffn2_size = cfg.early_exit.ffn2_size
        self.use_concat = cfg.early_exit.use_concat
        self.q_linear = nn.Linear(h_dim, 1)
        self.img_linear = nn.Linear(h_dim, 1)
        self.q_ffn = nn.Linear(h_dim, h_dim)
        self.v_ffn = nn.Linear(h_dim, h_dim)

        # self.bi_linear = nn.Linear(h_dim, h_dim)
        # self.ffn = nn.Linear(h_dim, ffn_size)
        # if ffn2_size:
        #     self.ffn2 = nn.Linear(ffn_size, ffn2_size)
        #     last_ffn_size = ffn2_size
        # else:
        #     self.ffn2 = None
        #     last_ffn_size = ffn_size
        # log.info(f'use_concat={self.use_concat}, ffn2_size={ffn2_size}')
        self.dropout = nn.Dropout(cfg.early_exit.droput_prob)
        self.ffn_act = get_activation(cfg.early_exit.ffn_act)
        # self.classifier = nn.Linear(h_dim * last_ffn_size, x_layer + 1)
        self.classifier = nn.Linear(h_dim, num_exit_layers)
        self.loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(0.2))

    def forward(self, q, img, q_mask=None, img_mask=None, labels=None):
        """
        Args:
            q: batch * len * hdim
            img: batch * len * hdim
            q_mask: batch * len (1 for padding, 0 for true)
            img_mask: batch * len (1 for padding, 0 for true)
            labels: batch * (x_layer + 1)
        Output:
            alpha: batch * len
        """

        # concat model
        if self.use_concat:
            q_img = torch.cat([q, img], dim=1)
            q_img_mask = torch.cat([q_mask, img_mask], dim=1)
            weighted_q = weighted_pool(self.q_linear, q_img, q_img_mask)
            weighted_v = weighted_pool(self.img_linear, q_img, q_img_mask)
        else:
            # individual model
            weighted_q = weighted_pool(self.q_linear, q, q_mask)
            weighted_v = weighted_pool(self.img_linear, img, img_mask)
        
        wq = self.q_ffn(weighted_q)
        wa = self.ffn_act(wq)  # batch, h_dim
        wq = self.dropout(wq)
        wv = self.v_ffn(weighted_v)
        wv = self.ffn_act(wv)  # batch, h_dim
        wv = self.dropout(wv)
        fv = wq*wv
        logits = self.classifier(fv)  # batch, x_layer

        # wy = self.bi_linear(weighted_v).unsqueeze(2)  # batch, h_dim, 1
        # x = weighted_q.unsqueeze(1)  # batch, 1, h_dim
        # wyx = wy.bmm(x)  # batch, h_dim, h_dim
        # v = self.ffn(wyx)  # batch, h_dim, ffn_size
        # v = self.dropout(v)
        # if self.ffn2 is not None:
        #     v = self.ffn2(v)
        # av = self.ffn_act(v.view(v.size(0), -1))  # batch, h_dim*ffn_size
        # logits = self.classifier(av)  # batch, x_layer

        ee_loss = None if labels is None else self.loss(logits, labels)
        return EarlyExitOutput(exit_scores=logits, loss=ee_loss)


class CrossExitModel(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        h_dim = cfg.num_qa_labels
        ffn_size = cfg.cross_exit.ffn_size
        ffn2_size = cfg.cross_exit.ffn2_size
        self.ffn = nn.Linear(h_dim, ffn_size)
        if ffn2_size:
            self.ffn2 = nn.Linear(ffn_size, ffn2_size)
            last_ffn_size = ffn2_size
        else:
            self.ffn2 = None
            last_ffn_size = ffn_size
        log.info(f'ffn_size={ffn_size}, ffn2_size={ffn2_size}')
        self.dropout = nn.Dropout(cfg.cross_exit.droput_prob)
        self.ffn_act = get_activation(cfg.cross_exit.ffn_act)
        self.classifier = nn.Linear(last_ffn_size, 2)
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, state, labels=None):
        """
        Args:
            state: batch * hdim
            labels: batch * 2
        Output:
            alpha: batch * len
        """
        v = self.ffn(state)  # batch, h_dim, ffn_size
        # v = self.dropout(v)
        # if self.ffn2 is not None:
        #     v = self.ffn2(v)
        # av = self.ffn_act(v.view(v.size(0), -1))  # batch, h_dim*ffn_size
        logits = self.classifier(v)  # batch, x_layer
        ee_loss = None if labels is None else self.loss(logits, labels)
        return EarlyExitOutput(exit_scores=logits, loss=ee_loss)


@dataclass
class EarlyExitOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    exit_scores: Optional[torch.FloatTensor] = None
