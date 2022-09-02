# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from mmcv.cnn import Scale
from mmcv.cnn import ConvModule, normal_init
from mmcv.ops import batched_nms
from mmcv.runner import force_fp32
from mmdet.core import multi_apply, reduce_mean, bbox2result
from mmdet.core.utils import filter_scores_and_topk, select_single_mlvl


from .anchor_free_head import AnchorFreeHead
from ..builder import HEADS, build_loss
from ..utils import ConvUpsample

INF = 1e8


class Folder(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, feature_map):
        N,_,H,W = feature_map.size()
        feature_map = F.unfold(feature_map,kernel_size=3,padding=1)
        feature_map = feature_map.view(N,-1,H,W)
        return feature_map

@HEADS.register_module()
class RepParserHead(AnchorFreeHead):
    """Anchor-free head used in `FCOS <https://arxiv.org/abs/1904.01355>`_.

    The FCOS head does not use anchor boxes. Instead bounding boxes are
    predicted at each pixel and a centerness measure is used to suppress
    low-quality predictions.
    Here norm_on_bbox, centerness_on_reg, dcn_on_last_conv are training
    tricks used in official repo, which will bring remarkable mAP gains
    of up to 4.9. Please see https://github.com/tianzhi0549/FCOS for
    more detail.

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        strides (list[int] | list[tuple[int, int]]): Strides of points
            in multiple feature levels. Default: (4, 8, 16, 32, 64).
        regress_ranges (tuple[tuple[int, int]]): Regress range of multiple
            level points.
        center_sampling (bool): If true, use center sampling. Default: False.
        center_sample_radius (float): Radius of center sampling. Default: 1.5.
        norm_on_bbox (bool): If true, normalize the regression targets
            with FPN strides. Default: False.
        centerness_on_reg (bool): If true, position centerness on the
            regress branch. Please refer to https://github.com/tianzhi0549/FCOS/issues/89#issuecomment-516877042.
            Default: False.
        conv_bias (bool | str): If specified as `auto`, it will be decided by the
            norm_cfg. Bias of conv will be set as True if `norm_cfg` is None, otherwise
            False. Default: "auto".
        loss_cls (dict): Config of classification loss.
        loss_bbox (dict): Config of localization loss.
        loss_centerness (dict): Config of centerness loss.
        norm_cfg (dict): dictionary to construct and config norm layer.
            Default: norm_cfg=dict(type='GN', num_groups=32, requires_grad=True).
        init_cfg (dict or list[dict], optional): Initialization config dict.

    Example:
        >>> self = FCOSHead(11, 7)
        >>> feats = [torch.rand(1, 7, s, s) for s in [4, 8, 16, 32, 64]]
        >>> cls_score, bbox_pred, centerness = self.forward(feats)
        >>> assert len(cls_score) == len(self.scales)
    """  # noqa: E501

    def __init__(self,
                 num_classes,
                 num_part_classes,
                 in_channels,
                 regress_ranges=((-1, 64), (64, 128), (128, 256), (256, 512),
                                 (512, INF)),
                 center_sampling=False,
                 center_sample_radius=1.5,
                 norm_on_bbox=False,
                 centerness_on_reg=False,
                 feat_levels = [0, 1, 2],
                 sem_stack_convs=2,
                 sem_channels=128,
                 parse_b_stack_conv=2,
                 parse_b_channels=128,
                 parse_h_stack_convs=3,
                 parse_h_channels=32,
                 mask_feat_stride=8,
                 parsing_out_stride=4,
                 loss_cls=dict(
                     type='FocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=1.0),
                 loss_bbox=dict(type='IoULoss', loss_weight=1.0),
                 loss_centerness=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,
                     loss_weight=1.0),
                loss_parsing=dict(
                    type='CrossEntropyLoss', loss_weight=1.0),
                loss_seg=dict(
                    type='CrossEntropyLoss', loss_weight=2.0, ignore_index=255),
                norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
                init_cfg=dict(
                     type='Normal',
                     layer='Conv2d',
                     std=0.01,
                     override=dict(
                         type='Normal',
                         name='conv_cls',
                         std=0.01,
                         bias_prob=0.01)),
                **kwargs):
        # fcos
        self.regress_ranges = regress_ranges
        self.center_sampling = center_sampling
        self.center_sample_radius = center_sample_radius
        self.norm_on_bbox = norm_on_bbox
        self.centerness_on_reg = centerness_on_reg

        # semantic fpn
        self.feat_levels= feat_levels
        self.sem_stack_convs = sem_stack_convs
        self.sem_channels = sem_channels

        # mask branch
        self.parse_b_stack_conv = parse_b_stack_conv
        self.parse_b_channels = parse_b_channels

        # parsing branch
        self.num_part_classes = num_part_classes
        self.parse_h_stack_convs = parse_h_stack_convs
        self.parse_h_channels = parse_h_channels

        self.parsing_out_stride = parsing_out_stride
        self.mask_feat_stride = mask_feat_stride

        super().__init__(
            num_classes,
            in_channels,
            loss_cls=loss_cls,
            loss_bbox=loss_bbox,
            norm_cfg=norm_cfg,
            init_cfg=init_cfg,
            **kwargs)
        self.loss_centerness = build_loss(loss_centerness)
        self.loss_parsing = build_loss(loss_parsing)
        self.loss_seg = build_loss(loss_seg)

    def _init_layers(self):
        """Initialize layers of the head."""
        super()._init_layers()
        self.conv_centerness = nn.Conv2d(self.feat_channels, 1, 3, padding=1)
        self.scales = nn.ModuleList([Scale(1.0) for _ in self.strides])
        
        # semantics FPN
        self.sem_conv = nn.ModuleList()
        for i, _ in enumerate(self.feat_levels):
            self.sem_conv.append(
                ConvUpsample(
                    self.in_channels,
                    self.sem_channels,
                    num_layers=i if i > 0 else 1,
                    num_upsample=i if i > 0 else 0,
                    conv_cfg=None,
                    norm_cfg=dict(type='GN', num_groups=32, requires_grad=True)))
        
        # sem logit head
        sem_logit = []
        for _ in range(self.sem_stack_convs):
            sem_logit.append(
                ConvModule(
                    self.sem_channels,
                    self.sem_channels,
                    3,
                    padding=1,
                    stride=1,
                    conv_cfg=None,
                    norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
                    act_cfg=dict(type='ReLU', inplace=True),
                    bias=False))
        sem_logit.append(
            nn.Conv2d(
                self.sem_channels,
                self.num_part_classes,
                kernel_size=1,
                bias=False))
        self.sem_logit_head = nn.Sequential(*sem_logit)

        # parsing feat branch
        parse_branch = []
        for _ in range(self.parse_b_stack_conv):
            parse_branch.append(
                ConvModule(
                    self.parse_b_channels,
                    self.parse_b_channels,
                    kernel_size=3,
                    padding=1,
                    stride=1,
                    conv_cfg=None,
                    norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
                    act_cfg=dict(type='ReLU', inplace=True),
                    bias=False))
        parse_branch.append(
            nn.Conv2d(
                self.parse_b_channels,
                self.parse_h_channels,
                kernel_size=1,
                stride=1))
        self.parse_branch = nn.Sequential(*parse_branch)

        # parsing head
        self.weight_nums = []
        self.bias_nums = []
        for i in range(self.parse_h_stack_convs):
            if i == 0:
                # for rel_coords
                self.weight_nums.append((self.parse_h_channels + 2) * self.parse_h_channels)
                self.bias_nums.append(self.parse_h_channels)
            elif i == self.parse_h_stack_convs - 1:
                self.weight_nums.append(self.parse_h_channels*2 * self.num_part_classes)
                self.bias_nums.append(self.num_part_classes)
            else:
                self.weight_nums.append(self.parse_h_channels * self.parse_h_channels)
                self.bias_nums.append(self.parse_h_channels)
        self.total_params = 0
        self.total_params += sum(self.weight_nums)
        self.total_params += sum(self.bias_nums)

        # parsing controller
        normal_init(self.conv_cls, std=0.001, bias=0)
        self.top_module = Folder()
        self.part_offset = nn.Linear(9 * self.feat_channels, (self.num_part_classes-1) * 2)
        self.part_cls = nn.Linear(9 * self.feat_channels, (self.num_part_classes-1))
        self.controller = nn.Linear(10 * self.feat_channels, self.total_params)
        self.part_fusion = nn.Linear((self.num_part_classes-1) * self.feat_channels, self.feat_channels)
        self.part_feats = nn.Conv2d(
            self.parse_h_channels,
            8*(self.num_part_classes-1),
            kernel_size=3,
            padding=1)
        self.part_group = nn.Conv2d(
            10*(self.num_part_classes-1),
            10*(self.num_part_classes-1),
            kernel_size=3,
            padding=1,
            groups=(self.num_part_classes-1))
        self.part_proj = nn.Conv2d(
            10*(self.num_part_classes-1),
            self.parse_h_channels,
            kernel_size=3,
            padding=1)

    def forward_train(self,
                      x,
                      img_metas,
                      gt_bboxes,
                      gt_labels=None,
                      gt_bboxes_ignore=None,
                      proposal_cfg=None,
                      **kwargs):
        """
        Args:
            x (list[Tensor]): Features from FPN.
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes (Tensor): Ground truth bboxes of the image,
                shape (num_gts, 4).
            gt_labels (Tensor): Ground truth labels of each box,
                shape (num_gts,).
            gt_bboxes_ignore (Tensor): Ground truth bboxes to be
                ignored, shape (num_ignored_gts, 4).
            proposal_cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used

        Returns:
            tuple:
                losses: (dict[str, Tensor]): A dictionary of loss components.
                proposal_list (list[Tensor]): Proposals of each image.
        """
        pass

    def forward(self, feats):
        """Forward features from the upstream network.

        Args:
            feats (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.

        Returns:
            tuple:
                cls_scores (list[Tensor]): Box scores for each scale level, \
                    each is a 4D-tensor, the channel number is \
                    num_points * num_classes.
                bbox_preds (list[Tensor]): Box energies / deltas for each \
                    scale level, each is a 4D-tensor, the channel number is \
                    num_points * 4.
                centernesses (list[Tensor]): centerness for each scale level, \
                    each is a 4D-tensor, the channel number is num_points * 1.
        """
        # seg fpn forward
        sem_logit, sem_feats = self.sem_head_forward(feats)
        # parsing branch
        parse_feats = self.parse_branch_forward(sem_feats)

        cls_score, bbox_pred, centerness, inst_feats, reg_feats =\
            multi_apply(self.forward_single, feats, self.scales, self.strides)
        return cls_score, bbox_pred, centerness,\
               inst_feats, reg_feats, parse_feats,\
               sem_logit
    
    def sem_head_forward(self, x):
        feats = []
        for i, layer in enumerate(self.sem_conv):
            f = layer(x[self.feat_levels[i]])
            feats.append(f)
        feats = torch.sum(torch.stack(feats, dim=0), dim=0)
        sem_logit = self.sem_logit_head(feats)

        return sem_logit, feats

    def parse_branch_forward(self, sem_feats):
        return self.parse_branch(sem_feats)

    def forward_single(self, x, scale, stride):
        """Forward features of a single scale level.

        Args:
            x (Tensor): FPN feature maps of the specified stride.
            scale (:obj: `mmcv.cnn.Scale`): Learnable scale module to resize
                the bbox prediction.
            stride (int): The corresponding stride for feature maps, only
                used to normalize the bbox prediction when self.norm_on_bbox
                is True.

        Returns:
            tuple: scores for each class, bbox predictions and centerness \
                predictions of input feature maps.
        """
        cls_score, bbox_pred, cls_feat, reg_feat = super().forward_single(x)
        if self.centerness_on_reg:
            centerness = self.conv_centerness(reg_feat)
        else:
            centerness = self.conv_centerness(cls_feat)
        # scale the bbox_pred of different level
        # float to avoid overflow when enabling FP16
        bbox_pred = scale(bbox_pred).float()
        if self.norm_on_bbox:
            bbox_pred = F.relu(bbox_pred)
            if not self.training:
                bbox_pred *= stride
        else:
            bbox_pred = bbox_pred.exp()
        inst_feat = self.top_module(reg_feat)
        return cls_score, bbox_pred, centerness, inst_feat, reg_feat
    
    def parsing_heads_forward(self, features, weights, biases, num_instances, part_heatmaps):
        '''
        :param features
        :param weights: [w0, w1, ...]
        :param bias: [b0, b1, ...]
        :return:
        '''
        assert features.dim() == 4
        x = features
        x = F.conv2d(x,
                     weights[0],
                     biases[0],
                     stride=1,
                     padding=0,
                     groups=num_instances)
        inst_feats = F.relu(x)
        inst_feats = inst_feats.reshape(num_instances, -1, inst_feats.shape[2], inst_feats.shape[3])
        part_feats = self.part_feats(inst_feats) # relu
        part_feats = F.relu(part_feats)
        N, _, H, W = part_feats.shape
        part_feats = part_feats.reshape(N, self.num_part_classes-1, -1, H, W)
        part_heatmaps = part_heatmaps.reshape(N, self.num_part_classes-1, 2, H, W)
        part_feats = torch.cat([part_feats, part_heatmaps], dim=2)
        part_feats = part_feats.reshape(N, -1, H, W)
        part_feats = self.part_group(part_feats)
        part_feats = F.relu(part_feats)
        part_feats = self.part_proj(part_feats) # relu
        part_feats = F.relu(part_feats)
        part_enhance_feats = torch.cat([inst_feats, part_feats], dim=1)
        part_enhance_feats = part_enhance_feats.reshape(1, -1, H, W)
        part_logits = F.conv2d(part_enhance_feats,
                               weights[1],
                               biases[1],
                               stride=1,
                               padding=0,
                               groups=num_instances)
        return part_logits.reshape(num_instances, -1, H, W)

    def loss_sem_seg(self, sem_logits, gt_semantic_seg):
        pass

    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'centernesses'))
    def loss(self,
             cls_scores,
             bbox_preds,
             centernesses,
             inst_feats,
             reg_feats,
             parse_feats,
             sem_logit,
             gt_bboxes,
             gt_labels,
             img_metas,
             gt_bboxes_ignore=None,
             **kwargs):
        pass

    def get_targets(self, points, gt_bboxes_list, gt_labels_list):
        pass

    def _get_target_single(self, gt_bboxes, gt_labels, points, regress_ranges,
                           num_points_per_lvl):
        pass

    def centerness_target(self, pos_bbox_targets):
        pass
    
    def _get_points_single(self,
                           featmap_size,
                           stride,
                           dtype,
                           device,
                           flatten=False):
        """Get points according to feature map size.

        This function will be deprecated soon.
        """
        warnings.warn(
            '`_get_points_single` in `FCOSHead` will be '
            'deprecated soon, we support a multi level point generator now'
            'you can get points of a single level feature map '
            'with `self.prior_generator.single_level_grid_priors` ')

        y, x = super()._get_points_single(featmap_size, stride, dtype, device)
        points = torch.stack((x.reshape(-1) * stride, y.reshape(-1) * stride),
                             dim=-1) + stride // 2
        return points


    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'centernesses',
                          'kernel_preds'))
    def get_bboxes(self,
                   cls_scores,
                   bbox_preds,
                   centernesses,
                   inst_feats,
                   reg_feats,
                   parse_feats,
                   sem_logits,
                   img_metas,
                   cfg=None,
                   rescale=False,
                   with_nms=True):

        assert len(cls_scores) == len(bbox_preds)

        cfg = self.test_cfg if cfg is None else cfg
        num_levels = len(cls_scores)
        featmap_sizes = [cls_scores[i].shape[-2:] for i in range(num_levels)]

        all_level_points_strides = self.prior_generator.grid_priors(
            featmap_sizes,
            dtype=bbox_preds[0].dtype,
            device=bbox_preds[0].device,
            with_stride=True)
        all_level_points = [i[:, :2] for i in all_level_points_strides]
        all_level_strides = [i[:, 2:] for i in all_level_points_strides]

        det_results_list = []
        parsing_results_list = []

        for img_id in range(len(img_metas)):
            img_meta = img_metas[img_id]

            cls_score_list = select_single_mlvl(cls_scores, img_id)
            bbox_pred_list = select_single_mlvl(bbox_preds, img_id)
            centerness_pred_list = select_single_mlvl(centernesses, img_id)
            inst_feat_list = select_single_mlvl(inst_feats, img_id)
            parse_feats_i = parse_feats[img_id].unsqueeze(0)
            reg_feats_i = [ref_feat[img_id] for ref_feat in reg_feats]
            det_bboxes, det_labels, det_parsings = self._get_bboxes_single(
                cls_score_list,
                bbox_pred_list,
                centerness_pred_list,
                inst_feat_list,
                reg_feats_i,
                parse_feats_i,
                all_level_points,
                all_level_strides,
                img_meta,
                cfg,
                rescale=rescale,
                with_nms=with_nms)

            if det_bboxes.shape[0] == 0:
                det_results_list.append([np.zeros((0, 5), dtype=np.float32) for i in range(self.num_classes)])
                parsing_results_list.append([np.zeros((0, 0), dtype=np.float32) for i in range(self.num_classes)])
                continue

            bbox_results = bbox2result(det_bboxes, det_labels, self.num_classes)
            parsing_results = [[] for _ in range(self.num_classes)]
            for i in range(det_bboxes.shape[0]):
                label = det_labels[i]
                parsing = det_parsings[i].cpu().numpy()
                parsing_results[label].append(parsing)

            det_results_list.append(bbox_results)
            parsing_results_list.append(parsing_results)
        
        return [[det_results_list, parsing_results_list]]

    def _get_bboxes_single(self,
                           cls_score_list,
                           bbox_pred_list,
                           centerness_pred_list,
                           inst_feat_list,
                           reg_feats_i,
                           parse_feats_i,
                           mlvl_points,
                           mlvl_strides,
                           img_meta,
                           cfg,
                           rescale=False,
                           with_nms=True,
                           **kwargs):
        assert len(cls_score_list) == len(bbox_pred_list)
        cfg = self.test_cfg if cfg is None else cfg
        img_shape = img_meta['img_shape']
        scale_factor = img_meta['scale_factor']
        ori_shape = img_meta['ori_shape']
        nms_pre = cfg.get('nms_pre', -1)

        mlvl_bboxes = []
        mlvl_scores = []
        mlvl_labels = []
        mlvl_centerness = []
        mlvl_inst_feats_pred = []
        mlvl_reg_feats = []
        flatten_mlvl_points = []
        flatten_mlvl_strides = []
        for cls_score, bbox_pred, centerness,\
            inst_feats, reg_feat, points, strides in zip(
                cls_score_list, bbox_pred_list, centerness_pred_list,
                inst_feat_list,  reg_feats_i, mlvl_points,
                mlvl_strides):

            cls_score = cls_score.permute(1, 2, 0).reshape(
                -1, self.cls_out_channels).sigmoid()
            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4)
            centerness = centerness.permute(1, 2, 0).reshape(-1).sigmoid()
            inst_feats = inst_feats.permute(1, 2, 0).reshape(-1, 9 * self.feat_channels)
            reg_feat = reg_feat.permute(1, 2, 0).reshape(-1, self.feat_channels)

            # TODO: test whether scores mutilply centerness
            results = filter_scores_and_topk(
                cls_score, cfg.score_thr, nms_pre,
                dict(bbox_pred=bbox_pred,
                     centerness=centerness,
                     inst_feats=inst_feats,
                     points=points,
                     strides=strides))
            scores, labels, _, filtered_results = results

            bbox_pred = filtered_results['bbox_pred']
            centerness = filtered_results['centerness']
            inst_feats = filtered_results['inst_feats']
            points = filtered_results['points']
            strides = filtered_results['strides']

            bboxes = self.bbox_coder.decode(
                points, bbox_pred, max_shape=img_shape)
            
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
            mlvl_labels.append(labels)
            mlvl_centerness.append(centerness)
            mlvl_inst_feats_pred.append(inst_feats)
            mlvl_reg_feats.append(reg_feat)
            flatten_mlvl_points.append(points)
            flatten_mlvl_strides.append(strides)

        mlvl_bboxes = torch.cat(mlvl_bboxes)
        mlvl_scores = torch.cat(mlvl_scores)
        mlvl_labels = torch.cat(mlvl_labels)
        mlvl_inst_feats_pred = torch.cat(mlvl_inst_feats_pred)
        mlvl_reg_feats = torch.cat(mlvl_reg_feats)
        mlvl_centerness = torch.cat(mlvl_centerness)
        flatten_mlvl_points = torch.cat(flatten_mlvl_points)
        flatten_mlvl_strides = torch.cat(flatten_mlvl_strides)

        if rescale:
            mlvl_bboxes /= mlvl_bboxes.new_tensor(scale_factor)

        # nms process
        mlvl_scores = mlvl_scores * mlvl_centerness
        if with_nms:
            if mlvl_bboxes.numel() == 0:
                det_bboxes = torch.cat([mlvl_bboxes, mlvl_scores[:, None]], -1)
                return det_bboxes, mlvl_labels, []

            det_bboxes, keep_idxs = batched_nms(mlvl_bboxes, mlvl_scores,
                                                mlvl_labels, cfg.nms)
            det_bboxes = det_bboxes[:cfg.max_per_img]
            det_labels = mlvl_labels[keep_idxs][:cfg.max_per_img]
            det_points = flatten_mlvl_points[keep_idxs][:cfg.max_per_img]
            det_strides = flatten_mlvl_strides[keep_idxs][:cfg.max_per_img][:, 0]
            pos_inst_feats = mlvl_inst_feats_pred[keep_idxs][:cfg.max_per_img]
            part_offsets = self.part_offset(pos_inst_feats)
            part_offsets = part_offsets.reshape(-1, self.num_part_classes-1, 2)
            part_cls_scores = self.part_cls(pos_inst_feats).sigmoid()

            N, C, _ = part_offsets.shape
            scale_det_bboxes = det_bboxes[:, :4] * det_bboxes[:, :4].new_tensor(scale_factor)
            w = (scale_det_bboxes[:, 2] - scale_det_bboxes[:, 0]).unsqueeze(1)
            h = (scale_det_bboxes[:, 3] - scale_det_bboxes[:, 1]).unsqueeze(1)
            part_offsets[:, :, 0] *= w
            part_offsets[:, :, 1] *= h

            pos_part_points = det_points.unsqueeze(1) + part_offsets
            batch_input_shape = img_meta['batch_input_shape']
            pos_part_points[..., 0] = pos_part_points[..., 0].clamp(0, batch_input_shape[1])
            pos_part_points[..., 1] = pos_part_points[..., 1].clamp(0, batch_input_shape[0])

            num_points = [0]
            for i, center in enumerate(mlvl_points):
                num_points.append(num_points[i] + center.size(0))

            pos_part_inds = torch.zeros(
                (N, C), dtype=torch.long, device=pos_part_points.device)
            for i, stride in enumerate(self.strides):
                num_point = num_points[i]
                valid_inds = (det_strides == stride)
                if not valid_inds.any():
                    continue
                pos_part_points_per_stride = pos_part_points[valid_inds]
                ref_points = mlvl_points[i]
                n, c, _ = pos_part_points_per_stride.shape
                pos_part_points_per_stride = pos_part_points_per_stride.reshape(n*c, 1, -1)
                ref_points = ref_points.unsqueeze(0)
                dis = torch.sqrt(
                    (pos_part_points_per_stride[:, :, 0] - ref_points[:, :, 0])**2 +\
                    (pos_part_points_per_stride[:, :, 1] - ref_points[:, :, 1])**2)
                inds = dis.min(-1)[1]
                inds = num_point + inds
                inds = inds.reshape(-1, c)
                pos_part_inds[valid_inds] = inds

            pos_part_inds = pos_part_inds.reshape(-1)
            pos_part_feats = mlvl_reg_feats[pos_part_inds]
            pos_part_feats = pos_part_feats.reshape(N, C, self.feat_channels)
            pos_part_feats = pos_part_feats * part_cls_scores.unsqueeze(-1)
            pos_part_feats = pos_part_feats.reshape(N, -1)
            pos_part_feats = self.part_fusion(pos_part_feats)
            pos_enhanced_feats = torch.cat([pos_inst_feats, pos_part_feats], dim=-1)
            parsing_head_params = self.controller(pos_enhanced_feats)

        # generate mask
        parsings = []
        if det_bboxes.shape[0] > 0:
            num_instance = len(det_points)
            batch_inds = parsing_head_params.new_zeros((num_instance), dtype=torch.long)
            mask_head_inputs = self.relative_coordinate_feature_generator(
                parse_feats_i,
                self.mask_feat_stride,
                det_points,
                det_strides,
                batch_inds)
            weights, biases = self.parse_dynamic_params(
                parsing_head_params,
                self.parse_h_channels,
                self.weight_nums,
                self.bias_nums)

            part_cls_scores = part_cls_scores.reshape(-1,)
            pos_part_points = pos_part_points.reshape(-1, 2)
            pos_part_points[part_cls_scores < 0.5] = det_points.unsqueeze(1).repeat(
                1, self.num_part_classes-1, 1).reshape(-1, 2)[part_cls_scores < 0.5]
            part_coords = self.part_coord_generator(
                parse_feats_i,
                self.mask_feat_stride,
                pos_part_points,
                det_strides)
            
            mask_logits = self.parsing_heads_forward(
                mask_head_inputs,
                weights,
                biases,
                num_instance,
                part_coords)
            mask_logits = mask_logits.reshape(-1, self.num_part_classes, parse_feats_i.size(2), parse_feats_i.size(3))
            mask_logits = self.aligned_bilinear(mask_logits, self.mask_feat_stride // self.parsing_out_stride).softmax(dim=1)
            if rescale:
                pred_global_masks = self.aligned_bilinear(mask_logits, self.parsing_out_stride)
                pred_global_masks = pred_global_masks[:, :, :img_shape[0], :img_shape[1]]
                parsings = F.interpolate(
                    pred_global_masks,
                    size=(ori_shape[0], ori_shape[1]),
                    mode='bilinear',
                    align_corners=False)
            else:
                parsings = self.aligned_bilinear(mask_logits, self.parsing_out_stride).squeeze(1)
            parsings = parsings.argmax(dim=1)

        return det_bboxes, det_labels, parsings

    def part_coord_generator(self, mask_feats, mask_feat_stride, part_locations, strides):
        H, W = mask_feats[0].size()[1:]
        locations = self.compute_locations(H,
                                           W,
                                           stride=mask_feat_stride,
                                           device=mask_feats.device)
        part_locations = part_locations.reshape(-1, 2)
        strides = strides.unsqueeze(-1).repeat(1, self.num_part_classes-1).reshape(-1,)
        relative_coordinates = part_locations.reshape(-1, 1, 2) - locations.reshape(1, -1, 2)
        relative_coordinates = relative_coordinates.permute(0, 2, 1).float()
        relative_coordinates = relative_coordinates / (strides.float().reshape(-1, 1, 1) * 8.0)
        relative_coordinates = relative_coordinates.to(dtype=mask_feats.dtype)
        relative_coordinates = relative_coordinates.reshape(-1, self.num_part_classes-1, 2, H, W)

        return relative_coordinates

    def relative_coordinate_feature_generator(self,
                                              mask_feats,
                                              mask_feat_stride,
                                              instance_locations,
                                              strides,
                                              batch_inds):
        # obtain relative coordinate features for mask generator
        # TODO: stride * 8.0 equal the original implement of source code: sizes_of_interest
        num_instance = len(instance_locations)
        H, W = mask_feats[0].size()[1:]
        locations = self.compute_locations(H,
                                           W,
                                           stride=mask_feat_stride,
                                           device=mask_feats.device)
        relative_coordinates = instance_locations.reshape(-1, 1, 2) - locations.reshape(1, -1, 2)
        relative_coordinates = relative_coordinates.permute(0, 2, 1).float()
        relative_coordinates = relative_coordinates / (strides.float().reshape(-1, 1, 1) * 8.0)
        relative_coordinates = relative_coordinates.to(dtype=mask_feats.dtype)
        pos_mask_feats = mask_feats[batch_inds]
        coordinates_feat = torch.cat([
            relative_coordinates.view(num_instance, 2, H, W),
            pos_mask_feats], dim=1)
        coordinates_feat = coordinates_feat.view(1, -1, H, W)
        return coordinates_feat

    def parse_dynamic_params(self, params, channels, weight_nums, bias_nums):
        assert params.dim() == 2
        assert len(weight_nums) == len(bias_nums)
        assert params.size(1) == sum(weight_nums) + sum(bias_nums)
        num_instances = params.size(0)
        num_layers = len(weight_nums)

        params_splits = list(
            torch.split_with_sizes(params, weight_nums + bias_nums, dim=1))

        weight_splits = params_splits[:num_layers]
        bias_splits = params_splits[num_layers:]

        for l in range(num_layers):
            if l < num_layers - 1:
                # out_channels x in_channels x 1 x 1
                weight_splits[l] = weight_splits[l].reshape(
                    num_instances * channels, -1, 1, 1)
                bias_splits[l] = bias_splits[l].reshape(num_instances * channels)
            else:
                # out_channels x in_channels x 1 x 1
                weight_splits[l] = weight_splits[l].reshape(
                    num_instances * self.num_part_classes, -1, 1, 1)
                bias_splits[l] = bias_splits[l].reshape(num_instances * self.num_part_classes)
        return weight_splits, bias_splits
    
    def aligned_bilinear(self, tensor, factor):
        assert tensor.dim() == 4
        assert factor >= 1
        assert int(factor) == factor
        if factor == 1:
            return tensor

        h, w = tensor.size()[2:]
        tensor = F.pad(tensor, pad=(0, 1, 0, 1), mode="replicate")
        oh = factor * h + 1
        ow = factor * w + 1
        tensor = F.interpolate(tensor,
                            size=(oh, ow),
                            mode='bilinear',
                            align_corners=True)
        tensor = F.pad(tensor,
                    pad=(factor // 2, 0, factor // 2, 0),
                    mode="replicate")
        return tensor[:, :, :oh - 1, :ow - 1]
    
    def compute_locations(self, h, w, stride, device):
        shifts_x = torch.arange(
            0,
            w * stride,
            step=stride,
            dtype=torch.float32,
            device=device)
        shifts_y = torch.arange(0,
            h * stride,
            step=stride,
            dtype=torch.float32,
            device=device)
        shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
        shift_x = shift_x.reshape(-1)
        shift_y = shift_y.reshape(-1)
        locations = torch.stack((shift_x, shift_y), dim=1) + stride // 2
        return locations
