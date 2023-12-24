# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from torch import Tensor
from typing import List, Tuple

from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.registry import MODELS
from ..utils import resize
from mmseg.utils import ConfigType, SampleList
from ..builder import build_loss
from ..losses import accuracy

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
class SegformerMultiHead(BaseDecodeHead):
    """The all mlp Head of segformer.

    This head is the implementation of
    `Segformer <https://arxiv.org/abs/2105.15203>` _.

    Args:
        interpolate_mode: The interpolate mode of MLP head upsample operation.
            Default: 'bilinear'.
    """

    def __init__(self, interpolate_mode='bilinear', num_classes=[10, 12, 28, 40], **kwargs):
        super().__init__(input_transform='multiple_select', num_classes=10, **kwargs)

        self.interpolate_mode = interpolate_mode
        self.num_classes = num_classes
        num_inputs = len(self.in_channels)
        self.num_outputs = len(self.num_classes)
        self.hierarchy_correction_enabled = True
        self.hierarchy_correction_mode = 'soft'
        self.hierarchies = {'Semantic':0,
                            'SemanticExtended':1,
                            'Material':2,
                            'MaterialMorphology':3}
        self.hierarchy_correction_indices = [
            torch.as_tensor([5, 8, 0, 0, 0, 1, 4, 2, 3, 6, 7, 9]),
            torch.as_tensor([0, 0, 1, 5, 7, 6, 4, 2, 3, 3, 11, 10, 10, 10, 10, 10, 8, 8, 7, 8, 8, 8, 7, 9, 9, 9, 9, 9]),
            torch.as_tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20, 21, 22, 23, 23, 15, 23, 23, 23, 23, 24, 24, 15, 25, 15, 25, 25, 26, 26, 15, 27])]

        assert num_inputs == len(self.in_index)

        self.convs = nn.ModuleList()
        for i in range(num_inputs):
            self.convs.append(
                ConvModule(
                    in_channels=self.in_channels[i],
                    out_channels=self.channels,
                    kernel_size=1,
                    stride=1,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg))

        self.fusion_conv = ConvModule(
            in_channels=self.channels * num_inputs,
            out_channels=self.channels,
            kernel_size=1,
            norm_cfg=self.norm_cfg)

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
        # Receive 4 stage backbone feature map: 1/4, 1/8, 1/16, 1/32
        inputs = self._transform_inputs(inputs)
        outs = []
        for idx in range(len(inputs)):
            x = inputs[idx]
            conv = self.convs[idx]
            outs.append(
                resize(
                    input=conv(x),
                    size=inputs[0].shape[2:],
                    mode=self.interpolate_mode,
                    align_corners=self.align_corners))

        out = self.fusion_conv(torch.cat(outs, dim=1))
        x, x_pred = self.classification_blocks[0](out)
        outs = [x_pred]
        for idx in range(1,self.num_outputs):
            x, x_pred = self.classification_blocks[idx](x)
            if self.hierarchy_correction_enabled:
                y_pred = outs[-1]
                num_o_y = self.num_classes[idx-1]
                hierarchy_y_to_x_indices = self.hierarchy_correction_indices[idx-1]
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
        hierarchy = 'SemanticExtended'#((batch_img_metas[0])['seg_map_path']).split('/Labels_mask_')[1].split('/')[0]
        # hierarchy = ((batch_img_metas[0])['seg_map_path']).split('/Labels_mask_')[1].split('/')[0]
        seg_logits = seg_logits[self.hierarchies[hierarchy]]

        seg_logits = resize(
            input=seg_logits,
            size=batch_img_metas[0]['img_shape'],
            mode='bilinear',
            align_corners=self.align_corners)
        return seg_logits


