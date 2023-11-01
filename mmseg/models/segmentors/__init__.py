# Copyright (c) OpenMMLab. All rights reserved.
from .base import BaseSegmentor
from .cascade_encoder_decoder import CascadeEncoderDecoder
# from .encoder_decoder import EncoderDecoder
from .encoder_decoder_multihead import EncoderDecoderMultihead as EncoderDecoder
from .seg_tta import SegTTAModel

__all__ = [
    'BaseSegmentor', 'EncoderDecoder', 'CascadeEncoderDecoder', 'SegTTAModel'
]
