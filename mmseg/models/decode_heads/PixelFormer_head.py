import torch
import torch.nn as nn
import torch.nn.functional as F
from mmseg.registry import MODELS
from .decode_head import BaseDecodeHead

from ..utils import resize

from .PQI import PSP
from .SAM import SAM
########################################################################################################################

class BCP(nn.Module):
    """ Multilayer perceptron."""

    def __init__(self, max_depth, min_depth, in_features=512, hidden_features=512*4, out_features=256, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.min_depth = min_depth
        self.max_depth = max_depth

    def forward(self, x):
        x = torch.mean(x.flatten(start_dim=2), dim = 2)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        bins = torch.softmax(x, dim=1)
        bins = bins / bins.sum(dim=1, keepdim=True)
        bin_widths = (self.max_depth - self.min_depth) * bins
        bin_widths = nn.functional.pad(bin_widths, (1, 0), mode='constant', value=self.min_depth)
        bin_edges = torch.cumsum(bin_widths, dim=1)
        centers = 0.5 * (bin_edges[:, :-1] + bin_edges[:, 1:])
        n, dout = centers.size()
        centers = centers.contiguous().view(n, dout, 1, 1)
        return centers

@MODELS.register_module()
class PixelFormerHead(BaseDecodeHead):

    def __init__(self,
                 embed_dims=768,
                 post_process_channels=[96, 192, 384, 768],
                 expand_channels=False,
                 act_cfg=dict(type='ReLU'),
                 inv_depth=False,
                 num_heads=[4, 8, 16, 32],
                 norm_cfg=dict(type='BN'),
                 window_size=7,
                 max_depth=65,
                 min_depth=0.01,
                 pretrained=None,
                 **kwargs):
        super().__init__(**kwargs)

        self.inv_depth = inv_depth
        self.in_channels = self.in_channels
        self.num_heads = num_heads
        self.embed_dims = embed_dims
        self.window_size = window_size
        self.max_depth = max_depth
        self.min_depth = min_depth

        decoder_cfg = dict(
            in_channels=self.in_channels,
            in_index=[0, 1, 2, 3],
            pool_scales=(1, 2, 3, 6),
            channels=self.embed_dims,
            dropout_ratio=0.0,
            num_classes=self.num_classes,
            norm_cfg=norm_cfg,
            align_corners=False
        )

        v_dim = decoder_cfg['num_classes']*4
        sam_dims=[128, 256, 512, 1024]
        v_dims = [64, 128, 256, self.embed_dims]
        self.sam4 = SAM(input_dim=self.in_channels[3], embed_dim=sam_dims[3], window_size=self.window_size, v_dim=v_dims[3], num_heads=self.num_heads[3])
        self.sam3 = SAM(input_dim=self.in_channels[2], embed_dim=sam_dims[2], window_size=self.window_size, v_dim=v_dims[2], num_heads=self.num_heads[2])
        self.sam2 = SAM(input_dim=self.in_channels[1], embed_dim=sam_dims[1], window_size=self.window_size, v_dim=v_dims[1], num_heads=self.num_heads[1])
        self.sam1 = SAM(input_dim=self.in_channels[0], embed_dim=sam_dims[0], window_size=self.window_size, v_dim=v_dims[0], num_heads=self.num_heads[0])

        self.decoder = PSP(**decoder_cfg)
        self.disp_head1 = DispHead(input_dim=sam_dims[0])

        self.bcp = BCP(max_depth=max_depth, min_depth=min_depth)

        self.init_weights(pretrained=pretrained)

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone and heads.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        print(f'== Load encoder backbone from: {pretrained}')
        self.decoder.init_weights()

    def forward(self, inputs):

        q4 = self.decoder(inputs)
        q3 = self.sam4(inputs[3], q4)
        q3 = nn.PixelShuffle(2)(q3)
        q2 = self.sam3(inputs[2], q3)
        q2 = nn.PixelShuffle(2)(q2)
        q1 = self.sam2(inputs[1], q2)
        q1 = nn.PixelShuffle(2)(q1)
        q0 = self.sam1(inputs[0], q1)
        bin_centers = self.bcp(q4)
        f = self.disp_head1(q0, bin_centers, 4)
        return f

    def loss(self, inputs, batch_data_samples,
             train_cfg):
        """Forward function for training.

        Args:
            inputs (Tuple[Tensor]): List of multi-level img features.
            batch_data_samples (list[:obj:`SegDataSample`]): The seg
                data samples. It usually includes information such
                as `img_metas` or `gt_semantic_seg`.
            train_cfg (dict): The training config.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        pred_depth = self.forward(inputs)
        losses = self.loss_by_feat(pred_depth, batch_data_samples)
        return losses

    def _stack_batch_gt_depth(self, batch_data_samples):
        gt_depths = [
            data_sample.gt_depth.data for data_sample in batch_data_samples
        ]
        return torch.stack(gt_depths, dim=0)

    def _stack_batch_images(self, batch_data_samples):
        images = [
            data_sample.input_img.data for data_sample in batch_data_samples
        ]
        return torch.stack(images, dim=0)
    def loss_by_feat(self, pred_depth,
                     batch_data_samples):
        """Compute segmentation loss.

        Args:
            seg_logits (Tensor): The output from decode head forward function.
            batch_data_samples (List[:obj:`SegDataSample`]): The seg
                data samples. It usually includes information such
                as `metainfo` and `gt_sem_seg`.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """

        gt_depths = self._stack_batch_gt_depth(batch_data_samples)
        images = self._stack_batch_images(batch_data_samples)

        loss = dict()
        pred_depth = resize(
            input=pred_depth,
            size=gt_depths.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)

        # seg_label = seg_label.squeeze(1)

        if not isinstance(self.loss_decode, nn.ModuleList):
            losses_decode = [self.loss_decode]
        else:
            losses_decode = self.loss_decode
        for loss_decode in losses_decode:
            if loss_decode.loss_name not in loss:
                if 'smoothness' in loss_decode.loss_name:
                    loss[loss_decode.loss_name] = loss_decode(
                        pred_depth,
                        images)
                else:
                    loss[loss_decode.loss_name] = loss_decode(
                        pred_depth,
                        gt_depths)
            else:
                loss[loss_decode.loss_name] += loss_decode(
                    pred_depth,
                    gt_depths)

        return loss

class DispHead(nn.Module):
    def __init__(self, input_dim=100):
        super(DispHead, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, 256, 3, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, centers, scale):
        x = self.conv1(x)
        x = x.softmax(dim=1)
        x = torch.sum(x * centers, dim=1, keepdim=True)
        if scale > 1:
            x = upsample(x, scale_factor=scale)
        return x


def upsample(x, scale_factor=2, mode="bilinear", align_corners=False):
    """Upsample input tensor by a factor of 2
    """
    return F.interpolate(x, scale_factor=scale_factor, mode=mode, align_corners=align_corners)
