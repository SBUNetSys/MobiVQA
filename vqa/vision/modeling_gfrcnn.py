"""
 coding=utf-8
 Copyright 2018, Antonio Mendoza Hao Tan, Mohit Bansal
 Adapted From Facebook Inc, Detectron2 && Huggingface Co.

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.import copy
 """
import itertools
import math
import os
from abc import ABCMeta, abstractmethod
from collections import OrderedDict, namedtuple
from typing import Dict, List, Tuple

import numpy as np
import torch
from detectron2.modeling import build_backbone
from torch import nn
from torch.nn import functional as F
from torch.nn.modules.batchnorm import BatchNorm2d
from torchvision.ops import RoIPool
from torchvision.ops.boxes import batched_nms, nms

from ..utils import WEIGHTS_NAME, Config, cached_path, hf_bucket_url, is_remote_url, load_checkpoint


# other:
def norm_box(boxes, raw_sizes):
    if not isinstance(boxes, torch.Tensor):
        normalized_boxes = boxes.copy()
    else:
        normalized_boxes = boxes.clone()
    normalized_boxes[:, :, (0, 2)] /= raw_sizes[:, 1, None, None]
    normalized_boxes[:, :, (1, 3)] /= raw_sizes[:, 0, None, None]
    return normalized_boxes


def pad_list_tensors(
    list_tensors,
    preds_per_image,
    max_detections=None,
    return_tensors=None,
    padding=None,
    pad_value=0,
    location=None,
):
    """
    location will always be cpu for np tensors
    """
    if location is None:
        location = "cpu"
    assert return_tensors in {"pt", "np", None}
    assert padding in {"max_detections", "max_batch", None}
    new = []
    if padding is None:
        if return_tensors is None:
            return list_tensors
        elif return_tensors == "pt":
            if not isinstance(list_tensors, torch.Tensor):
                return torch.stack(list_tensors).to(location)
            else:
                return list_tensors.to(location)
        else:
            if not isinstance(list_tensors, list):
                return np.array(list_tensors.to(location))
            else:
                return list_tensors.to(location)
    if padding == "max_detections":
        assert max_detections is not None, "specify max number of detections per batch"
    elif padding == "max_batch":
        max_detections = max(preds_per_image)
    for i in range(len(list_tensors)):
        too_small = False
        tensor_i = list_tensors.pop(0)
        if tensor_i.ndim < 2:
            too_small = True
            tensor_i = tensor_i.unsqueeze(-1)
        assert isinstance(tensor_i, torch.Tensor)
        tensor_i = F.pad(
            input=tensor_i,
            pad=(0, 0, 0, max_detections - preds_per_image[i]),
            mode="constant",
            value=pad_value,
        )
        if too_small:
            tensor_i = tensor_i.squeeze(-1)
        if return_tensors is None:
            if location == "cpu":
                tensor_i = tensor_i.cpu()
            tensor_i = tensor_i.tolist()
        if return_tensors == "np":
            if location == "cpu":
                tensor_i = tensor_i.cpu()
            tensor_i = tensor_i.numpy()
        else:
            if location == "cpu":
                tensor_i = tensor_i.cpu()
        new.append(tensor_i)
    if return_tensors == "np":
        return np.stack(new, axis=0)
    elif return_tensors == "pt" and not isinstance(new, torch.Tensor):
        return torch.stack(new, dim=0)
    else:
        return list_tensors


def do_nms(boxes, scores, image_shape, score_thresh, nms_thresh, mind, maxd):
    scores = scores[:, :-1]
    num_bbox_reg_classes = boxes.shape[1] // 4
    # Convert to Boxes to use the `clip` function ...
    boxes = boxes.reshape(-1, 4)
    _clip_box(boxes, image_shape)
    boxes = boxes.view(-1, num_bbox_reg_classes, 4)  # R x C x 4

    # Select max scores
    max_scores, max_classes = scores.max(1)  # R x C --> R
    num_objs = boxes.size(0)
    boxes = boxes.view(-1, 4)
    idxs = torch.arange(num_objs).to(boxes.device) * num_bbox_reg_classes + max_classes
    max_boxes = boxes[idxs]  # Select max boxes according to the max scores.

    # Apply NMS
    keep = nms(max_boxes, max_scores, nms_thresh)
    keep = keep[:maxd]
    if keep.shape[-1] >= mind and keep.shape[-1] <= maxd:
        max_boxes, max_scores = max_boxes[keep], max_scores[keep]
        classes = max_classes[keep]
        return max_boxes, max_scores, classes, keep
    else:
        return None


# Helper Functions
def _clip_box(tensor, box_size: Tuple[int, int]):
    # assert torch.isfinite(tensor).all(), "Box tensor contains infinite or NaN!"
    h, w = box_size
    tensor[:, 0].clamp_(min=0, max=w)
    tensor[:, 1].clamp_(min=0, max=h)
    tensor[:, 2].clamp_(min=0, max=w)
    tensor[:, 3].clamp_(min=0, max=h)


def _nonempty_boxes(box, threshold: float = 0.0) -> torch.Tensor:
    widths = box[:, 2] - box[:, 0]
    heights = box[:, 3] - box[:, 1]
    keep = (widths > threshold) & (heights > threshold)
    return keep


def get_norm(norm, out_channels):
    if isinstance(norm, str):
        if len(norm) == 0:
            return None
        norm = {
            "BN": BatchNorm2d,
            "GN": lambda channels: nn.GroupNorm(32, channels),
            "nnSyncBN": nn.SyncBatchNorm,  # keep for debugging
            "": lambda x: x,
        }[norm]
    return norm(out_channels)


def _create_grid_offsets(size: List[int], stride: int, offset: float, device):

    grid_height, grid_width = size
    shifts_x = torch.arange(
        offset * stride,
        grid_width * stride,
        step=stride,
        dtype=torch.float32,
        device=device,
    )
    shifts_y = torch.arange(
        offset * stride,
        grid_height * stride,
        step=stride,
        dtype=torch.float32,
        device=device,
    )

    shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
    shift_x = shift_x.reshape(-1)
    shift_y = shift_y.reshape(-1)
    return shift_x, shift_y


class GeneralizedRCNN(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.device = torch.device(cfg.MODEL.DEVICE)
        self.backbone = build_backbone(cfg)
        self.to(self.device)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        config = kwargs.pop("config", None)
        state_dict = kwargs.pop("state_dict", None)
        cache_dir = kwargs.pop("cache_dir", None)
        from_tf = kwargs.pop("from_tf", False)
        force_download = kwargs.pop("force_download", False)
        resume_download = kwargs.pop("resume_download", False)
        proxies = kwargs.pop("proxies", None)
        local_files_only = kwargs.pop("local_files_only", False)
        use_cdn = kwargs.pop("use_cdn", True)

        # Load config if we don't provide a configuration
        if not isinstance(config, Config):
            config_path = config if config is not None else pretrained_model_name_or_path
            # try:
            config = Config.from_pretrained(
                config_path,
                cache_dir=cache_dir,
                force_download=force_download,
                resume_download=resume_download,
                proxies=proxies,
                local_files_only=local_files_only,
            )

        # Load model
        if pretrained_model_name_or_path is not None:
            if os.path.isdir(pretrained_model_name_or_path):
                if os.path.isfile(os.path.join(pretrained_model_name_or_path, WEIGHTS_NAME)):
                    # Load from a PyTorch checkpoint
                    archive_file = os.path.join(pretrained_model_name_or_path, WEIGHTS_NAME)
                else:
                    raise EnvironmentError(
                        "Error no file named {} found in directory {} ".format(
                            WEIGHTS_NAME,
                            pretrained_model_name_or_path,
                        )
                    )
            elif os.path.isfile(pretrained_model_name_or_path) or is_remote_url(pretrained_model_name_or_path):
                archive_file = pretrained_model_name_or_path
            elif os.path.isfile(pretrained_model_name_or_path + ".index"):
                assert (
                    from_tf
                ), "We found a TensorFlow checkpoint at {}, please set from_tf to True to load from this checkpoint".format(
                    pretrained_model_name_or_path + ".index"
                )
                archive_file = pretrained_model_name_or_path + ".index"
            else:
                archive_file = hf_bucket_url(
                    pretrained_model_name_or_path,
                    filename=WEIGHTS_NAME,
                    use_cdn=use_cdn,
                )

            try:
                # Load from URL or cache if already cached
                resolved_archive_file = cached_path(
                    archive_file,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    proxies=proxies,
                    resume_download=resume_download,
                    local_files_only=local_files_only,
                )
                if resolved_archive_file is None:
                    raise EnvironmentError
            except EnvironmentError:
                msg = f"Can't load weights for '{pretrained_model_name_or_path}'."
                raise EnvironmentError(msg)

            if resolved_archive_file == archive_file:
                print("loading weights file {}".format(archive_file))
            else:
                print("loading weights file {} from cache at {}".format(archive_file, resolved_archive_file))
        else:
            resolved_archive_file = None

        # Instantiate model.
        model = cls(config)

        if state_dict is None:
            try:
                try:
                    state_dict = torch.load(resolved_archive_file, map_location="cpu")
                except Exception:
                    state_dict = load_checkpoint(resolved_archive_file)

            except Exception:
                raise OSError(
                    "Unable to load weights from pytorch checkpoint file. "
                    "If you tried to load a PyTorch model from a TF 2.0 checkpoint, please set from_tf=True. "
                )

        missing_keys = []
        unexpected_keys = []
        error_msgs = []

        # Convert old format to new format if needed from a PyTorch state_dict
        old_keys = []
        new_keys = []
        for key in state_dict.keys():
            new_key = None
            if "gamma" in key:
                new_key = key.replace("gamma", "weight")
            if "beta" in key:
                new_key = key.replace("beta", "bias")
            if new_key:
                old_keys.append(key)
                new_keys.append(new_key)
        for old_key, new_key in zip(old_keys, new_keys):
            state_dict[new_key] = state_dict.pop(old_key)

        # copy state_dict so _load_from_state_dict can modify it
        metadata = getattr(state_dict, "_metadata", None)
        state_dict = state_dict.copy()
        if metadata is not None:
            state_dict._metadata = metadata

        model_to_load = model
        model_to_load.load_state_dict(state_dict)

        if model.__class__.__name__ != model_to_load.__class__.__name__:
            base_model_state_dict = model_to_load.state_dict().keys()
            head_model_state_dict_without_base_prefix = [
                key.split(cls.base_model_prefix + ".")[-1] for key in model.state_dict().keys()
            ]
            missing_keys.extend(head_model_state_dict_without_base_prefix - base_model_state_dict)

        if len(unexpected_keys) > 0:
            print(
                f"Some weights of the model checkpoint at {pretrained_model_name_or_path} were not used when "
                f"initializing {model.__class__.__name__}: {unexpected_keys}\n"
                f"- This IS expected if you are initializing {model.__class__.__name__} from the checkpoint of a model trained on another task "
                f"or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).\n"
                f"- This IS NOT expected if you are initializing {model.__class__.__name__} from the checkpoint of a model that you expect "
                f"to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model)."
            )
        else:
            print(f"All model checkpoint weights were used when initializing {model.__class__.__name__}.\n")
        if len(missing_keys) > 0:
            print(
                f"Some weights of {model.__class__.__name__} were not initialized from the model checkpoint at {pretrained_model_name_or_path} "
                f"and are newly initialized: {missing_keys}\n"
                f"You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference."
            )
        else:
            print(
                f"All the weights of {model.__class__.__name__} were initialized from the model checkpoint at {pretrained_model_name_or_path}.\n"
                f"If your task is similar to the task the model of the checkpoint was trained on, "
                f"you can already use {model.__class__.__name__} for predictions without further training."
            )
        if len(error_msgs) > 0:
            raise RuntimeError(
                "Error(s) in loading state_dict for {}:\n\t{}".format(
                    model.__class__.__name__, "\n\t".join(error_msgs)
                )
            )
        # Set model in evaluation mode to deactivate DropOut modules by default
        model.eval()

        return model

    def forward(
        self,
        images,
        image_shapes,
        gt_boxes=None,
        proposals=None,
        scales_yx=None,
        **kwargs,
    ):
        """
        kwargs:
            max_detections (int), return_tensors {"np", "pt", None}, padding {None,
            "max_detections"}, pad_value (int), location = {"cuda", "cpu"}
        """
        if self.training:
            raise NotImplementedError()
        return self.inference(
            images=images,
            image_shapes=image_shapes,
            gt_boxes=gt_boxes,
            proposals=proposals,
            scales_yx=scales_yx,
            **kwargs,
        )

    @torch.no_grad()
    def inference(
        self,
        images,
        image_shapes,
        gt_boxes=None,
        proposals=None,
        scales_yx=None,
        **kwargs,
    ):
        # run images through backbone
        original_sizes = image_shapes * scales_yx
        features = self.backbone(images)
        # return features
        return features


