"""
 coding=utf-8
 Copyright 2018, Antonio Mendoza Hao Tan, Mohit Bansal
 Adapted From Facebook Inc, Detectron2

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
import sys
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from detectron2.structures import ImageList
import detectron2.data.detection_utils as utils
import detectron2.data.transforms as T

from ..utils import img_tensorize


class ResizeShortestEdge:
    def __init__(self, short_edge_length, max_size=sys.maxsize):
        """
        Args:
            short_edge_length (list[min, max])
            max_size (int): maximum allowed longest edge length.
        """
        self.interp_method = "bilinear"
        self.max_size = max_size
        self.short_edge_length = short_edge_length

    def __call__(self, imgs):
        img_augs = []
        for img in imgs:
            h, w = img.shape[:2]
            # later: provide list and randomly choose index for resize
            size = np.random.randint(self.short_edge_length[0], self.short_edge_length[1] + 1)
            if size == 0:
                return img
            scale = size * 1.0 / min(h, w)
            if h < w:
                newh, neww = size, scale * w
            else:
                newh, neww = scale * h, size
            if max(newh, neww) > self.max_size:
                scale = self.max_size * 1.0 / max(newh, neww)
                newh = newh * scale
                neww = neww * scale
            neww = int(neww + 0.5)
            newh = int(newh + 0.5)

            if img.dtype == np.uint8:
                pil_image = Image.fromarray(img)
                pil_image = pil_image.resize((neww, newh), Image.BILINEAR)
                img = np.asarray(pil_image)
            else:
                img = img.permute(2, 0, 1).unsqueeze(0)  # 3, 0, 1)  # hw(c) -> nchw
                img = F.interpolate(img, (newh, neww), mode=self.interp_method, align_corners=False).squeeze(0)
            img_augs.append(img)

        return img_augs


class ImageLoader:

    def __init__(self, cfg):
        self.input_format = cfg.INPUT.FORMAT
        min_size = cfg.INPUT.MIN_SIZE_TEST
        max_size = cfg.INPUT.MAX_SIZE_TEST
        sample_style = "choice"
        self.tfm_gens = [T.ResizeShortestEdge(min_size, max_size, sample_style)]
        self.device = torch.device(cfg.MODEL.DEVICE)
        # normalize
        assert len(cfg.MODEL.PIXEL_MEAN) == len(cfg.MODEL.PIXEL_STD)
        num_channels = len(cfg.MODEL.PIXEL_MEAN)
        self.pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(num_channels, 1, 1)
        self.pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(num_channels, 1, 1)
        self.normalizer = lambda x: (x - self.pixel_mean) / self.pixel_std

    def __call__(self, image_file, *args, **kwargs):
        image = utils.read_image(image_file, format=self.input_format)
        image, transforms = T.apply_transform_gens(self.tfm_gens, image)
        # image_shape = image.shape[:2]  # h, w
        img_tensor = torch.as_tensor(np.ascontiguousarray(
            image.transpose(2, 0, 1)))
        if self.device.type == 'cuda':
            img_tensor = img_tensor.pin_memory().to(
                self.device, non_blocking=True)
        normed_img = self.normalizer(img_tensor)
        images = ImageList.from_tensors([normed_img])
        return images


class Preprocess:
    def __init__(self, cfg):
        self.aug = ResizeShortestEdge([cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST)
        self.input_format = cfg.INPUT.FORMAT
        self.size_divisibility = 0  # cfg.SIZE_DIVISIBILITY
        self.pad_value = 0  # cfg.PAD_VALUE
        self.max_image_size = cfg.INPUT.MAX_SIZE_TEST
        self.device = cfg.MODEL.DEVICE
        self.pixel_std = torch.tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(len(cfg.MODEL.PIXEL_STD), 1, 1)
        self.pixel_mean = torch.tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(len(cfg.MODEL.PIXEL_STD), 1, 1)
        self.normalizer = lambda x: (x - self.pixel_mean) / self.pixel_std

    def pad(self, images):
        max_size = tuple(max(s) for s in zip(*[img.shape for img in images]))
        image_sizes = [im.shape[-2:] for im in images]
        images = [
            F.pad(
                im,
                [0, max_size[-1] - size[1], 0, max_size[-2] - size[0]],
                value=self.pad_value,
            )
            for size, im in zip(image_sizes, images)
        ]

        return torch.stack(images), torch.tensor(image_sizes)

    def __call__(self, images, single_image=False):
        with torch.no_grad():
            if not isinstance(images, list):
                images = [images]
            if single_image:
                assert len(images) == 1
            for i in range(len(images)):
                if isinstance(images[i], torch.Tensor):
                    images.insert(i, images.pop(i).to(self.device).float())
                elif not isinstance(images[i], torch.Tensor):
                    images.insert(
                        i,
                        torch.as_tensor(img_tensorize(images.pop(i), input_format=self.input_format))
                        .to(self.device)
                        .float(),
                    )
            # resize smallest edge
            raw_sizes = torch.tensor([im.shape[:2] for im in images])
            images = self.aug(images)
            # transpose images and convert to torch tensors
            # images = [torch.as_tensor(i.astype("float32")).permute(2, 0, 1).to(self.device) for i in images]
            # now normalize before pad to avoid useless arithmetic
            images = [self.normalizer(x) for x in images]
            # now pad them to do the following operations
            images, sizes = self.pad(images)
            # Normalize

            if self.size_divisibility > 0:
                raise NotImplementedError()
            # pad
            scales_yx = torch.true_divide(raw_sizes, sizes)
            if single_image:
                return images[0], sizes[0], scales_yx[0]
            else:
                return images, sizes, scales_yx


def _scale_box(boxes, scale_yx):
    boxes[:, 0::2] *= scale_yx[:, 1]
    boxes[:, 1::2] *= scale_yx[:, 0]
    return boxes


def _clip_box(tensor, box_size: Tuple[int, int]):
    assert torch.isfinite(tensor).all(), "Box tensor contains infinite or NaN!"
    h, w = box_size
    tensor[:, 0].clamp_(min=0, max=w)
    tensor[:, 1].clamp_(min=0, max=h)
    tensor[:, 2].clamp_(min=0, max=w)
    tensor[:, 3].clamp_(min=0, max=h)
