import torch
import torch.nn.functional as F

from mmcv.cnn import Conv2d
from mmcv.cnn.bricks.transformer import (build_transformer_layer_sequence,
                                         build_positional_encoding)

from ..builder import NECKS
from .fpn import FPN


@NECKS.register_module()
class TransFPN(FPN):
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 encoder=None,
                 positional_encoding=dict(
                     type='SinePositionalEncoding',
                     num_feats=128,
                     normalize=True),
                 encoder_in_channels=2048):
        super(TransFPN, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            num_outs=num_outs)
        self.positional_encoding = build_positional_encoding(
            positional_encoding)
        self.encoder = build_transformer_layer_sequence(encoder)
        self.input_proj = Conv2d(
            encoder_in_channels, self.out_channels, kernel_size=1)
    
    def forward(self, inputs, img_metas):
        last_feats = inputs[-1]
        last_feats = self.input_proj(last_feats)
        bs, c, h, w = last_feats.shape
        input_img_h, input_img_w = img_metas[0]['batch_input_shape']
        masks = last_feats.new_ones((bs, input_img_h, input_img_w))
        for img_id in range(bs):
            img_h, img_w, _ = img_metas[img_id]['img_shape']
            masks[img_id, :img_h, :img_w] = 0
        masks = F.interpolate(
            masks.unsqueeze(1), size=last_feats.shape[-2:]).to(torch.bool).squeeze(1)
        pos_embed = self.positional_encoding(masks)  # [bs, embed_dim, h, w]
        masks = masks.view(bs, -1)  # [bs, h, w] -> [bs, h*w]
        pos_embed = pos_embed.view(bs, c, -1).permute(2, 0, 1)
        last_feats = last_feats.view(bs, c, -1).permute(2, 0, 1)  # [bs, c, h, w] -> [h*w, bs, c]
        memory = self.encoder(
            query=last_feats,
            key=None,
            value=None,
            query_pos=pos_embed,
            query_key_padding_mask=masks)
        memory = memory.permute(1, 0, 2).reshape(bs, c, h, w)
        inputs = (*inputs[:-1], memory)
        return super(TransFPN, self).forward(inputs)