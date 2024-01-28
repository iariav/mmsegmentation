# Copyright (c) OpenMMLab. All rights reserved.
# Originally from https://github.com/visual-attention-network/segnext
# Licensed under the Apache License, Version 2.0 (the "License")
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmengine.device import get_device

from mmseg.registry import MODELS
from ..utils import resize
from .decode_head import BaseDecodeHead

from torch import Tensor
from ..losses import accuracy
from mmseg.utils import ConfigType, SampleList
from typing import List, Tuple

class Matrix_Decomposition_2D_Base(nn.Module):
    """Base class of 2D Matrix Decomposition.

    Args:
        MD_S (int): The number of spatial coefficient in
            Matrix Decomposition, it may be used for calculation
            of the number of latent dimension D in Matrix
            Decomposition. Defaults: 1.
        MD_R (int): The number of latent dimension R in
            Matrix Decomposition. Defaults: 64.
        train_steps (int): The number of iteration steps in
            Multiplicative Update (MU) rule to solve Non-negative
            Matrix Factorization (NMF) in training. Defaults: 6.
        eval_steps (int): The number of iteration steps in
            Multiplicative Update (MU) rule to solve Non-negative
            Matrix Factorization (NMF) in evaluation. Defaults: 7.
        inv_t (int): Inverted multiple number to make coefficient
            smaller in softmax. Defaults: 100.
        rand_init (bool): Whether to initialize randomly.
            Defaults: True.
    """

    def __init__(self,
                 MD_S=1,
                 MD_R=64,
                 train_steps=6,
                 eval_steps=7,
                 inv_t=100,
                 rand_init=True):
        super().__init__()

        self.S = MD_S
        self.R = MD_R

        self.train_steps = train_steps
        self.eval_steps = eval_steps

        self.inv_t = inv_t

        self.rand_init = rand_init

    def _build_bases(self, B, S, D, R, device=None):
        raise NotImplementedError

    def local_step(self, x, bases, coef):
        raise NotImplementedError

    def local_inference(self, x, bases):
        # (B * S, D, N)^T @ (B * S, D, R) -> (B * S, N, R)
        coef = torch.bmm(x.transpose(1, 2), bases)
        coef = F.softmax(self.inv_t * coef, dim=-1)

        steps = self.train_steps if self.training else self.eval_steps
        for _ in range(steps):
            bases, coef = self.local_step(x, bases, coef)

        return bases, coef

    def compute_coef(self, x, bases, coef):
        raise NotImplementedError

    def forward(self, x, return_bases=False):
        """Forward Function."""
        B, C, H, W = x.shape

        # (B, C, H, W) -> (B * S, D, N)
        D = C // self.S
        N = H * W
        x = x.view(B * self.S, D, N)
        if not self.rand_init and not hasattr(self, 'bases'):
            bases = self._build_bases(1, self.S, D, self.R, device=x.device)
            self.register_buffer('bases', bases)

        # (S, D, R) -> (B * S, D, R)
        if self.rand_init:
            bases = self._build_bases(B, self.S, D, self.R, device=x.device)
        else:
            bases = self.bases.repeat(B, 1, 1)

        bases, coef = self.local_inference(x, bases)

        # (B * S, N, R)
        coef = self.compute_coef(x, bases, coef)

        # (B * S, D, R) @ (B * S, N, R)^T -> (B * S, D, N)
        x = torch.bmm(bases, coef.transpose(1, 2))

        # (B * S, D, N) -> (B, C, H, W)
        x = x.view(B, C, H, W)

        return x


class NMF2D(Matrix_Decomposition_2D_Base):
    """Non-negative Matrix Factorization (NMF) module.

    It is inherited from ``Matrix_Decomposition_2D_Base`` module.
    """

    def __init__(self, args=dict()):
        super().__init__(**args)

        self.inv_t = 1

    def _build_bases(self, B, S, D, R, device=None):
        """Build bases in initialization."""
        if device is None:
            device = get_device()
        bases = torch.rand((B * S, D, R)).to(device)
        bases = F.normalize(bases, dim=1)

        return bases

    def local_step(self, x, bases, coef):
        """Local step in iteration to renew bases and coefficient."""
        # (B * S, D, N)^T @ (B * S, D, R) -> (B * S, N, R)
        numerator = torch.bmm(x.transpose(1, 2), bases)
        # (B * S, N, R) @ [(B * S, D, R)^T @ (B * S, D, R)] -> (B * S, N, R)
        denominator = coef.bmm(bases.transpose(1, 2).bmm(bases))
        # Multiplicative Update
        coef = coef * numerator / (denominator + 1e-6)

        # (B * S, D, N) @ (B * S, N, R) -> (B * S, D, R)
        numerator = torch.bmm(x, coef)
        # (B * S, D, R) @ [(B * S, N, R)^T @ (B * S, N, R)] -> (B * S, D, R)
        denominator = bases.bmm(coef.transpose(1, 2).bmm(coef))
        # Multiplicative Update
        bases = bases * numerator / (denominator + 1e-6)

        return bases, coef

    def compute_coef(self, x, bases, coef):
        """Compute coefficient."""
        # (B * S, D, N)^T @ (B * S, D, R) -> (B * S, N, R)
        numerator = torch.bmm(x.transpose(1, 2), bases)
        # (B * S, N, R) @ (B * S, D, R)^T @ (B * S, D, R) -> (B * S, N, R)
        denominator = coef.bmm(bases.transpose(1, 2).bmm(bases))
        # multiplication update
        coef = coef * numerator / (denominator + 1e-6)

        return coef


class Hamburger(nn.Module):
    """Hamburger Module. It consists of one slice of "ham" (matrix
    decomposition) and two slices of "bread" (linear transformation).

    Args:
        ham_channels (int): Input and output channels of feature.
        ham_kwargs (dict): Config of matrix decomposition module.
        norm_cfg (dict | None): Config of norm layers.
    """

    def __init__(self,
                 ham_channels=512,
                 ham_kwargs=dict(),
                 norm_cfg=None,
                 **kwargs):
        super().__init__()

        self.ham_in = ConvModule(
            ham_channels, ham_channels, 1, norm_cfg=None, act_cfg=None)

        self.ham = NMF2D(ham_kwargs)

        self.ham_out = ConvModule(
            ham_channels, ham_channels, 1, norm_cfg=norm_cfg, act_cfg=None)

    def forward(self, x):
        enjoy = self.ham_in(x)
        enjoy = F.relu(enjoy, inplace=True)
        enjoy = self.ham(enjoy)
        enjoy = self.ham_out(enjoy)
        ham = F.relu(x + enjoy, inplace=True)

        return ham

class classification_block(nn.Module):
    """ Multilayer perceptron."""

    def __init__(self, in_channels, out_channels, norm_cfg, act_cfg, drop=0.):
        super().__init__()
        self.drop = nn.Dropout2d(drop)
        self.conv1 = ConvModule(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg)
        self.conv2 = ConvModule(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=3,
            padding=1,
            stride=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

        self.conv3 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.drop(x)
        x = self.conv2(x)
        x = self.drop(x)
        x1 = self.conv3(x)

        return x, x1
@MODELS.register_module()
class LightHamMultiHead(BaseDecodeHead):
    """SegNeXt decode head.

    This decode head is the implementation of `SegNeXt: Rethinking
    Convolutional Attention Design for Semantic
    Segmentation <https://arxiv.org/abs/2209.08575>`_.
    Inspiration from https://github.com/visual-attention-network/segnext.

    Specifically, LightHamHead is inspired by HamNet from
    `Is Attention Better Than Matrix Decomposition?
    <https://arxiv.org/abs/2109.04553>`.

    Args:
        ham_channels (int): input channels for Hamburger.
            Defaults: 512.
        ham_kwargs (int): kwagrs for Ham. Defaults: dict().
    """

    def __init__(self, ham_channels=512, ham_kwargs=dict(), num_classes=[10, 12, 28, 40], **kwargs):
        super().__init__(num_classes=10, input_transform='multiple_select', **kwargs)
        self.ham_channels = ham_channels
        num_inputs = len(self.in_channels)
        self.num_classes = num_classes
        self.num_outputs = len(self.num_classes)
        self.hierarchy_correction_enabled = True
        self.hierarchy_correction_mode = 'soft'
        self.hierarchies = {'Semantic': 0,
                            'SemanticExtended': 1,
                            'Material': 2,
                            'MaterialMorphology': 3}
        self.hierarchy_correction_indices = [
            torch.as_tensor([5, 8, 0, 0, 0, 1, 4, 2, 3, 6, 7, 9]),
            torch.as_tensor([0, 0, 1, 5, 7, 6, 4, 2, 3, 3, 11, 10, 10, 10, 10, 10, 8, 8, 7, 8, 8, 8, 7, 9, 9, 9, 9, 9]),
            torch.as_tensor(
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20, 21, 22, 23, 23, 15, 23, 23, 23,
                 23, 24, 24, 15, 25, 15, 25, 25, 26, 26, 15, 27])]

        # self.convs = nn.ModuleList()
        # for i in range(num_inputs):
        #     self.convs.append(
        #         ConvModule(
        #             in_channels=self.in_channels[i],
        #             out_channels=self.channels,
        #             kernel_size=1,
        #             stride=1,
        #             norm_cfg=self.norm_cfg,
        #             act_cfg=self.act_cfg))

        self.squeeze = ConvModule(
            sum(self.in_channels),
            self.ham_channels,
            1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

        self.hamburger = Hamburger(ham_channels, ham_kwargs, **kwargs)

        self.align = ConvModule(
            self.ham_channels,
            self.channels,
            1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

        self.classification_blocks = nn.ModuleList()
        for i in range(self.num_outputs):
            self.classification_blocks.append(
                classification_block(
                    in_channels=self.channels,
                    out_channels=self.num_classes[i],
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg,
                    drop=0.2
                )
            )

    def hierarchy_correction(self, x, y, x_to_y, x_num_o):

        if self.hierarchy_correction_mode == 'hard':
            x_preds = torch.argmax(x, axis=1)
            x_probs = F.one_hot(x_preds, x_num_o)
            x_to_y = x_to_y.to(device=x_probs.device)
            x_to_y_probs = torch.index_select(x_probs, dim=-1, index=x_to_y)
            x_to_y_probs = torch.permute(x_to_y_probs, (0, 3, 1, 2))
        elif self.hierarchy_correction_mode == 'soft':
            x_probs = torch.softmax(x, dim=1)
            x_to_y = x_to_y.to(device=x_probs.device)
            x_to_y_probs = torch.index_select(x_probs, dim=1, index=x_to_y)
        else:
            raise NameError('hierarchy_correction_mode must be one of [hard, soft]')

        return torch.mul(y, x_to_y_probs)
    def forward(self, inputs):
        """Forward function."""
        inputs = self._transform_inputs(inputs)

        # outs = []
        # for level,idx in enumerate(inputs):
        #     conv = self.convs[idx]
        #     inputs[idx] = resize(
        #             input=conv(level),
        #             size=inputs[0].shape[2:],
        #             mode='bilinear',
        #             align_corners=self.align_corners)

        inputs = [
            resize(
                level,
                size=inputs[0].shape[2:],
                mode='bilinear',
                align_corners=self.align_corners) for level in inputs
        ]

        inputs = torch.cat(inputs, dim=1)
        # apply a conv block to squeeze feature map
        x = self.squeeze(inputs)
        # apply hamburger module
        x = self.hamburger(x)
        # apply a conv block to align feature map
        output = self.align(x)

        x, x_pred = self.classification_blocks[0](output)
        outs = [x_pred]
        for idx in range(1, self.num_outputs):
            x, x_pred = self.classification_blocks[idx](x)
            if self.hierarchy_correction_enabled:
                y_pred = outs[-1]
                num_o_y = self.num_classes[idx - 1]
                hierarchy_y_to_x_indices = self.hierarchy_correction_indices[idx - 1]
                x_pred = self.hierarchy_correction(y_pred, x_pred, hierarchy_y_to_x_indices, num_o_y)
            outs.append(x_pred)


        return outs

    def loss_by_feat(self, seg_logits: Tensor,
                     batch_data_samples: SampleList) -> dict:
        """Compute segmentation loss.

        Argx_1 = layers.bilinear_interp(x_1, new_shape=tf.convert_to_tensor([input_shape[1], input_shape[2]]))s:
            seg_logits (Tensor): The output from decode head forward function.
            batch_data_samples (List[:obj:`SegDataSample`]): The seg
                data samples. It usually includes information such
                as `metainfo` and `gt_sem_seg`.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """

        seg_label = self._stack_batch_gt(batch_data_samples)
        hierarchy = batch_data_samples[0].seg_map_path.split('/Labels_mask_')[1].split('/')[0]
        seg_logits = seg_logits[self.hierarchies[hierarchy]]

        loss = dict()
        seg_logits = resize(
            input=seg_logits,
            size=seg_label.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        if self.sampler is not None:
            seg_weight = self.sampler.sample(seg_logits, seg_label)
        else:
            seg_weight = None
        seg_label = seg_label.squeeze(1)

        if not isinstance(self.loss_decode, nn.ModuleList):
            losses_decode = [self.loss_decode]
        else:
            losses_decode = self.loss_decode


        for loss_decode in losses_decode:
            loss_h = loss_decode.loss_name.split('_')[1]
            if loss_h != hierarchy:
                continue
            if loss_decode.loss_name not in loss:
                loss[loss_decode.loss_name] = loss_decode(
                    seg_logits,
                    seg_label,
                    weight=seg_weight,
                    ignore_index=self.ignore_index)
            else:
                loss[loss_decode.loss_name] += loss_decode(
                    seg_logits,
                    seg_label,
                    weight=seg_weight,
                    ignore_index=self.ignore_index)

        loss['acc_seg'] = accuracy(
            seg_logits, seg_label, ignore_index=self.ignore_index)
        return loss

    def predict_by_feat(self, seg_logits: Tensor,
                        batch_img_metas: List[dict]) -> Tensor:
        """Transform a batch of output seg_logits to the input shape.

        Args:
            seg_logits (Tensor): The output from decode head forward function.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.

        Returns:
            Tensor: Outputs segmentation logits map.
        """

        # for hierarchy in self.hierarchies:
        # hierarchy = 'MaterialMorphology'#((batch_img_metas[0])['seg_map_path']).split('/Labels_mask_')[1].split('/')[0]
        hierarchy = 'SemanticExtended'
        # hierarchy = ((batch_img_metas[0])['seg_map_path']).split('/Labels_mask_')[1].split('/')[0]
        seg_logits = seg_logits[self.hierarchies[hierarchy]]

        seg_logits = resize(
            input=seg_logits,
            size=batch_img_metas[0]['img_shape'],
            mode='bilinear',
            align_corners=self.align_corners)
        return seg_logits
