# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Tuple, Union

import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmengine.model import BaseModule
from torch import Tensor

from mmdet.registry import MODELS
from mmdet.utils import ConfigType, MultiConfig, OptConfigType

import cv2
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image


@MODELS.register_module()
class FPN(BaseModule):
    r"""Feature Pyramid Network.

    This is an implementation of paper `Feature Pyramid Networks for Object
    Detection <https://arxiv.org/abs/1612.03144>`_.

    Args:
        in_channels (list[int]): Number of input channels per scale.
        out_channels (int): Number of output channels (used at each scale).
        num_outs (int): Number of output scales.
        start_level (int): Index of the start input backbone level used to
            build the feature pyramid. Defaults to 0.
        end_level (int): Index of the end input backbone level (exclusive) to
            build the feature pyramid. Defaults to -1, which means the
            last level.
        add_extra_convs (bool | str): If bool, it decides whether to add conv
            layers on top of the original feature maps. Defaults to False.
            If True, it is equivalent to `add_extra_convs='on_input'`.
            If str, it specifies the source feature map of the extra convs.
            Only the following options are allowed

            - 'on_input': Last feat map of neck inputs (i.e. backbone feature).
            - 'on_lateral': Last feature map after lateral convs.
            - 'on_output': The last output feature map after fpn convs.
        relu_before_extra_convs (bool): Whether to apply relu before the extra
            conv. Defaults to False.
        no_norm_on_lateral (bool): Whether to apply norm on lateral.
            Defaults to False.
        conv_cfg (:obj:`ConfigDict` or dict, optional): Config dict for
            convolution layer. Defaults to None.
        norm_cfg (:obj:`ConfigDict` or dict, optional): Config dict for
            normalization layer. Defaults to None.
        act_cfg (:obj:`ConfigDict` or dict, optional): Config dict for
            activation layer in ConvModule. Defaults to None.
        upsample_cfg (:obj:`ConfigDict` or dict, optional): Config dict
            for interpolate layer. Defaults to dict(mode='nearest').
        init_cfg (:obj:`ConfigDict` or dict or list[:obj:`ConfigDict` or \
            dict]): Initialization config dict.

    Example:
        >>> import torch
        >>> in_channels = [2, 3, 5, 7]
        >>> scales = [340, 170, 84, 43]
        >>> inputs = [torch.rand(1, c, s, s)
        ...           for c, s in zip(in_channels, scales)]
        >>> self = FPN(in_channels, 11, len(in_channels)).eval()
        >>> outputs = self.forward(inputs)
        >>> for i in range(len(outputs)):
        ...     print(f'outputs[{i}].shape = {outputs[i].shape}')
        outputs[0].shape = torch.Size([1, 11, 340, 340])
        outputs[1].shape = torch.Size([1, 11, 170, 170])
        outputs[2].shape = torch.Size([1, 11, 84, 84])
        outputs[3].shape = torch.Size([1, 11, 43, 43])
    """

    def __init__(
        self,
        in_channels: List[int],
        out_channels: int,
        num_outs: int,
        start_level: int = 0,
        end_level: int = -1,
        add_extra_convs: Union[bool, str] = False,
        relu_before_extra_convs: bool = False,
        no_norm_on_lateral: bool = False,
        conv_cfg: OptConfigType = None,
        norm_cfg: OptConfigType = None,
        act_cfg: OptConfigType = None,
        upsample_cfg: ConfigType = dict(mode='nearest'),
        init_cfg: MultiConfig = dict(
            type='Xavier', layer='Conv2d', distribution='uniform'),
    ) -> None:
        super().__init__(init_cfg=init_cfg)
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.relu_before_extra_convs = relu_before_extra_convs
        self.no_norm_on_lateral = no_norm_on_lateral
        self.fp16_enabled = False
        self.upsample_cfg = upsample_cfg.copy()

        if end_level == -1 or end_level == self.num_ins - 1:
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            # if end_level is not the last level, no extra level is allowed
            self.backbone_end_level = end_level + 1
            assert end_level < self.num_ins
            assert num_outs == end_level - start_level + 1
        self.start_level = start_level
        self.end_level = end_level
        self.add_extra_convs = add_extra_convs
        assert isinstance(add_extra_convs, (str, bool))
        if isinstance(add_extra_convs, str):
            # Extra_convs_source choices: 'on_input', 'on_lateral', 'on_output'
            assert add_extra_convs in ('on_input', 'on_lateral', 'on_output')
        elif add_extra_convs:  # True
            self.add_extra_convs = 'on_input'

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        for i in range(self.start_level, self.backbone_end_level):
            l_conv = ConvModule(
                in_channels[i],
                out_channels,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
                act_cfg=act_cfg,
                inplace=False)
            fpn_conv = ConvModule(
                out_channels,
                out_channels,
                3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False)

            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

        # add extra conv layers (e.g., RetinaNet)
        extra_levels = num_outs - self.backbone_end_level + self.start_level
        if self.add_extra_convs and extra_levels >= 1:
            for i in range(extra_levels):
                if i == 0 and self.add_extra_convs == 'on_input':
                    in_channels = self.in_channels[self.backbone_end_level - 1]
                else:
                    in_channels = out_channels
                extra_fpn_conv = ConvModule(
                    in_channels,
                    out_channels,
                    3,
                    stride=2,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    inplace=False)
                self.fpn_convs.append(extra_fpn_conv)
        
        self.surf_map = 1
        if self.surf_map:
            self.imagenet_bgr256_mean = torch.tensor([123.675, 116.28, 103.53]).view(1, -1, 1, 1)  # Reshape for broadcasting
            self.imagenet_bgr256_std = torch.tensor([58.395, 57.12, 57.375]).view(1, -1, 1, 1)   # Reshape for broadcasting

            self.surf = cv2.cuda.SURF_CUDA_create(_hessianThreshold=400)
            self.transform = transforms.Compose(
                [
                    transforms.Lambda(lambda x: (x * self.imagenet_bgr256_std.to(x.device) + self.imagenet_bgr256_mean.to(x.device))),  # Denormalize
                    transforms.Grayscale(num_output_channels=1),
                ]
            )
            self.image_gpu = cv2.cuda_GpuMat()
            
            self.sigma = 11
            kernel_size = int(2 * torch.ceil(torch.tensor(3 * self.sigma)) + 1)  # Covers ~99.7% of the Gaussian
            self.gaussian_blur = transforms.GaussianBlur(kernel_size=kernel_size, sigma=self.sigma)
            self.debug_image_counter = 0
            self.strides = [4, 8, 16, 32]
            
    
    def generate_attention_map(self, keypoints_tensor, image_shape, stride, sigma):
        """
        Generates attention maps (density heatmaps) for a batch of images based on keypoint locations.
        This version is fully vectorized for GPU execution (no explicit for loops).

        Args:
            keypoints_tensor: A tensor of shape [batch_size, max_keypoints, 2] containing the (x, y) coordinates
                              of keypoints for each image in the batch. Keypoints are padded with -1.
                              Should be on the GPU.
            image_shape: A tuple (height, width) representing the original image shape.
            stride: The downscaling factor for the output attention map.
            sigma: The standard deviation of the Gaussian kernel used for blurring.

        Returns:
            A tensor of shape [batch_size, height // stride, width // stride] containing the attention maps.
            Will be on the GPU.
        """

        batch_size = keypoints_tensor.shape[0]
        heatmap_height = image_shape[0] // stride
        heatmap_width = image_shape[1] // stride

        # Initialize attention maps to 1.0 (uniform attention)
        attention_maps = torch.ones((batch_size, 1, heatmap_height, heatmap_width), device=keypoints_tensor.device)

        for i in range(batch_size):
            # 1. Remove padding and scale keypoints for this image
            valid_keypoints_mask = keypoints_tensor[i, :, 0] != -1  # Shape: [max_keypoints]
            valid_keypoints = keypoints_tensor[i, valid_keypoints_mask]  # Shape: [num_valid_keypoints, 2]

            if valid_keypoints.numel() > 0:
                # Keypoints exist for this image
                keypoints_scaled = (valid_keypoints / stride).long()

                # 3. Create delta map for this image
                delta_map = torch.zeros((heatmap_height, heatmap_width), device=keypoints_tensor.device)
                keypoints_scaled[:, 0] = torch.clamp(keypoints_scaled[:, 0], 0, heatmap_width - 1)
                keypoints_scaled[:, 1] = torch.clamp(keypoints_scaled[:, 1], 0, heatmap_height - 1)
                delta_map[keypoints_scaled[:, 1], keypoints_scaled[:, 0]] += 1.0

                # 4. Apply Gaussian blur
                delta_map = delta_map.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
                blurred_map = self.gaussian_blur(delta_map)

                # 5. Normalize attention map
                blurred_map_flat = blurred_map.view(-1)
                max_value = blurred_map_flat.max()
                attention_map = blurred_map_flat / (max_value + 1e-8)
                attention_map = attention_map.view(1, 1, heatmap_height, heatmap_width)

                # Assign the computed attention map to the correct index in the batch
                attention_maps[i] = attention_map

        return attention_maps 
    

    def forward(self, inputs: Tuple[Tensor], batch_inputs: Tensor) -> tuple:
        """Forward function.

        Args:
            inputs (tuple[Tensor]): Features from the upstream network, each
                is a 4D-tensor.
            batch_inputs (Tensor): Image tensor with shape (N, C, H ,W).

        Returns:
            tuple: Feature maps, each is a 4D-tensor.
        """
        assert len(inputs) == len(self.in_channels)

        if self.surf_map:
            x_gray = self.transform(batch_inputs).byte().squeeze(1).cpu().numpy()  # Shape: [N, H, W]
            all_keypoints = []
            for i, image_gray in enumerate(x_gray):
                self.image_gpu.upload(image_gray)  # Upload grayscale image to GPU
                keypoints = self.surf.detect(self.image_gpu, None)
                self.image_gpu.release()
                keypoints = self.surf.downloadKeypoints(keypoints)  # Download keypoints to CPU
                # Extract keypoint coordinates (x, y) into a PyTorch tensor
                if keypoints:  # Check if keypoints exist
                    keypoints_tensor = torch.tensor([[kp.pt[0], kp.pt[1]] for kp in keypoints], device='cuda')
                else:
                    keypoints_tensor = torch.ones((0, 2), device='cuda')  # Handle case with no keypoints
                all_keypoints.append(keypoints_tensor)
                
            all_keypoints = torch.nn.utils.rnn.pad_sequence(all_keypoints, batch_first=True, padding_value=-1)
            attention_maps = self.generate_attention_map(all_keypoints, batch_inputs.shape[2:], stride=4, sigma=self.sigma)
            grads = (torch.ones_like(attention_maps) - attention_maps) / 4
        
        laterals = []
        for i, lateral_conv in enumerate(self.lateral_convs):
            input_tensor = inputs[i + self.start_level]
            lateral_output = lateral_conv(input_tensor)
            if self.surf_map == 1:
                with torch.no_grad():
                    mask = (attention_maps + (grads * i))
                    mask = F.interpolate(mask, size=lateral_output.shape[2:], **self.upsample_cfg) # [N, 1, H, W]
                lateral_output *= mask
            laterals.append(lateral_output)
        
        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            # In some cases, fixing `scale factor` (e.g. 2) is preferred, but
            #  it cannot co-exist with `size` in `F.interpolate`.
            if 'scale_factor' in self.upsample_cfg:
                # fix runtime error of "+=" inplace operation in PyTorch 1.10
                laterals[i - 1] = laterals[i - 1] + F.interpolate(
                    laterals[i], **self.upsample_cfg)
            else:
                prev_shape = laterals[i - 1].shape[2:]
                laterals[i - 1] = laterals[i - 1] + F.interpolate(
                    laterals[i], size=prev_shape, **self.upsample_cfg)

        # build outputs
        # part 1: from original levels
        outs = [
            self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)
        ]
        # part 2: add extra levels
        if self.num_outs > len(outs):
            # use max pool to get more levels on top of outputs
            # (e.g., Faster R-CNN, Mask R-CNN)
            if not self.add_extra_convs:
                for i in range(self.num_outs - used_backbone_levels):
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))
            # add conv layers on top of original feature maps (RetinaNet)
            else:
                if self.add_extra_convs == 'on_input':
                    extra_source = inputs[self.backbone_end_level - 1]
                elif self.add_extra_convs == 'on_lateral':
                    extra_source = laterals[-1]
                elif self.add_extra_convs == 'on_output':
                    extra_source = outs[-1]
                else:
                    raise NotImplementedError
                outs.append(self.fpn_convs[used_backbone_levels](extra_source))
                for i in range(used_backbone_levels + 1, self.num_outs):
                    if self.relu_before_extra_convs:
                        outs.append(self.fpn_convs[i](F.relu(outs[-1])))
                    else:
                        outs.append(self.fpn_convs[i](outs[-1]))
        return tuple(outs)
