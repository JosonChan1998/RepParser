import os
import cv2
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.ops import batched_nms
from mmcv.cnn import ConvModule
from mmcv.runner import force_fp32
from mmdet.core import multi_apply, reduce_mean
from mmdet.core.utils import filter_scores_and_topk, select_single_mlvl

from .fcos_head import FCOSHead
from ..builder import HEADS, build_loss
from ..utils import (ConvUpsample, Folder, get_point_offset_target,
                     decode_point, relative_coordinate_maps,
                     parse_dynamic_params, dynamic_forward,
                     aligned_bilinear, points2bbox)

INF = 1e8


@HEADS.register_module()
class ReSParserHead(FCOSHead):

    def __init__(self,
                 *arg,
                 num_coarse_classes,
                 num_fine_classes,
                 mask_feat_stride=4,
                 parsing_out_stride=4,
                 sem_feat_levels = [0, 1, 2],
                 sem_stack_convs=2,
                 sem_channels=128,
                 parse_b_stack_conv=2,
                 parse_b_channels=128,
                 parse_h_stack_convs=2,
                 parse_h_channels=32,
                 num_parse_fcs=1,
                 num_part_feat_fcs=1,
                 num_rep_points=9,
                 fine_to_coarse_maps=None,
                 use_sample=True,
                 max_num_inst=32,
                 loss_part_cls=dict(
                    type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
                 loss_part_offset=dict(type='L1Loss', loss_weight=1.0),
                 loss_parsing=dict(type='CrossEntropyLoss', loss_weight=2.0),
                 loss_seg=dict(
                    type='CrossEntropyLoss', loss_weight=2.0, ignore_index=255),
                 loss_rep_bbox=dict(
                     type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=0.5),
                 loss_dice=dict(
                    type='DiceLoss',
                    use_sigmoid=True,
                    activate=True,
                    reduction='mean',
                    naive_dice=True,
                    eps=1.0,
                    loss_weight=1.0),
                 loss_mask=dict(
                    type='FocalLoss',
                    use_sigmoid=True,
                    gamma=2.0,
                    alpha=0.25,
                    reduction='mean',
                    loss_weight=20.0),
                 **kwargs):

        # basci setting
        self.num_fine_classes = num_fine_classes
        self.num_coarse_classes = num_coarse_classes
        self.parsing_out_stride = parsing_out_stride
        self.mask_feat_stride = mask_feat_stride

        # semantic fpn
        self.sem_feat_levels= sem_feat_levels
        self.sem_stack_convs = sem_stack_convs
        self.sem_channels = sem_channels

        # mask branch
        self.parse_b_stack_conv = parse_b_stack_conv
        self.parse_b_channels = parse_b_channels

        # parsing branch
        self.parse_h_stack_convs = parse_h_stack_convs
        self.parse_h_channels = parse_h_channels

        # num parse fcs
        self.num_parse_fcs = num_parse_fcs

        # num coarse part feature
        self.num_part_feat_fcs = num_part_feat_fcs

        # num rep points
        self.num_rep_points = num_rep_points
        self.fine_to_coarse_maps = fine_to_coarse_maps

        # sample
        self.use_sample = use_sample
        self.max_num_inst = max_num_inst

        super(ReSParserHead, self).__init__(*arg, **kwargs)

        # build loss
        self.loss_part_cls = build_loss(loss_part_cls)
        self.loss_part_offset = build_loss(loss_part_offset)
        self.loss_parsing = build_loss(loss_parsing)
        self.loss_seg = build_loss(loss_seg)
        self.loss_rep_bbox = build_loss(loss_rep_bbox)
        self.loss_dice = build_loss(loss_dice)
        self.loss_mask = build_loss(loss_mask)

    def _init_layers(self):
        """Initialize layers of the head."""
        super()._init_layers()
        # semantics FPN
        self.sem_conv = nn.ModuleList()
        for i, _ in enumerate(self.sem_feat_levels):
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
                self.num_fine_classes,
                kernel_size=1,
                bias=True))
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
                self.weight_nums.append(self.parse_h_channels * 1)
                self.bias_nums.append(1)
            else:
                self.weight_nums.append(self.parse_h_channels * self.parse_h_channels)
                self.bias_nums.append(self.parse_h_channels)
        self.total_params = 0
        self.total_params += sum(self.weight_nums)
        self.total_params += sum(self.bias_nums)

        # parsing controller
        self.inst_folder = Folder(kernel_size=3)
        part_cls = []
        for _ in range(self.num_parse_fcs):
            part_cls.append(nn.Linear(9*self.feat_channels, self.feat_channels))
            part_cls.append(nn.ReLU())
        part_cls.append(nn.Linear(self.feat_channels, self.num_fine_classes-1))
        self.part_cls = nn.Sequential(*part_cls)

        part_offset = []
        for _ in range(self.num_parse_fcs):
            part_offset.append(nn.Linear(9 * self.feat_channels, self.feat_channels))
            part_offset.append(nn.ReLU())
        part_offset.append(nn.Linear(self.feat_channels, (self.num_coarse_classes) * 2))
        self.part_offset = nn.Sequential(*part_offset)
        
        coarse_part_cls = []
        for _ in range(self.num_parse_fcs):
            coarse_part_cls.append(nn.Linear(9 * self.feat_channels, self.feat_channels))
            coarse_part_cls.append(nn.ReLU())
        coarse_part_cls.append(nn.Linear(self.feat_channels, self.num_coarse_classes))
        self.coarse_part_cls = nn.Sequential(*coarse_part_cls)

        coarse_part_feat = []
        for _ in range(self.num_part_feat_fcs):
            coarse_part_feat.append(nn.Linear(9 * self.feat_channels, self.feat_channels))
            coarse_part_feat.append(nn.ReLU())
        coarse_part_feat.append(nn.Linear(self.feat_channels,
            self.num_coarse_classes * self.feat_channels))
        self.coarse_part_feat = nn.Sequential(*coarse_part_feat)

        rep_offset = []
        for _ in range(self.num_parse_fcs):
            rep_offset.append(nn.Linear(2 * self.feat_channels, self.feat_channels))
            rep_offset.append(nn.ReLU())
        rep_offset.append(nn.Linear(self.feat_channels, (self.num_rep_points) * 2))
        self.rep_offset = nn.Sequential(*rep_offset)
        
        rep_cls = []
        for _ in range(self.num_parse_fcs):
            rep_cls.append(nn.Linear(2 * self.feat_channels, self.feat_channels))
            rep_cls.append(nn.ReLU())
        rep_cls.append(nn.Linear(
            self.feat_channels, self.num_rep_points * (self.num_fine_classes-1)))
        self.rep_cls = nn.Sequential(*rep_cls)

        self.rep_fusion = nn.Linear(9 * self.feat_channels, self.feat_channels)
        self.inst_fusion = nn.Linear(9 * self.feat_channels, self.feat_channels)

        controller = []
        for _ in range(self.num_parse_fcs):
            controller.append(nn.Linear(self.feat_channels, self.feat_channels))
            controller.append(nn.ReLU())
        controller.append(nn.Linear(self.feat_channels, self.total_params))
        self.controller = nn.Sequential(*controller)

        fine_to_coarse_maps = torch.tensor(self.fine_to_coarse_maps).reshape(-1,)
        fine_to_coarse_maps = F.one_hot(fine_to_coarse_maps).float()
        self.assign_matrix = nn.parameter.Parameter(
            fine_to_coarse_maps, requires_grad=False)

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
        outs = self(x)
        if gt_labels is None:
            loss_inputs = outs + (gt_bboxes, img_metas)
        else:
            loss_inputs = outs + (gt_bboxes, gt_labels, img_metas)
        losses = self.loss(*loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore, **kwargs)
        if proposal_cfg is None:
            return losses
        else:
            proposal_list = self.get_bboxes(
                *outs, img_metas=img_metas, cfg=proposal_cfg)
            return losses, proposal_list

    def sem_head_forward(self, x):
        feats = []
        for i, layer in enumerate(self.sem_conv):
            f = layer(x[self.sem_feat_levels[i]])
            feats.append(f)
        feats = torch.sum(torch.stack(feats, dim=0), dim=0)
        sem_logit = self.sem_logit_head(feats)

        return sem_logit, feats

    def parse_branch_forward(self, sem_feats):
        return self.parse_branch(sem_feats)

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
        # head forward
        cls_score, bbox_pred, centerness, inst_feats, img_feats =\
            multi_apply(self.forward_single, feats, self.scales, self.strides)
        return cls_score, bbox_pred, centerness, \
               inst_feats, img_feats, parse_feats, sem_logit

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
        cls_score, bbox_pred, cls_feat, reg_feat = super(FCOSHead, self).forward_single(x)
        if self.centerness_on_reg:
            centerness = self.conv_centerness(reg_feat)
        else:
            centerness = self.conv_centerness(cls_feat)
        # scale the bbox_pred of different level
        # float to avoid overflow when enabling FP16
        bbox_pred = scale(bbox_pred).float()
        if self.norm_on_bbox:
            # bbox_pred needed for gradient computation has been modified
            # by F.relu(bbox_pred) when run with PyTorch 1.10. So replace
            # F.relu(bbox_pred) with bbox_pred.clamp(min=0)
            bbox_pred = bbox_pred.clamp(min=0)
            if not self.training:
                bbox_pred *= stride
        else:
            bbox_pred = bbox_pred.exp()
        inst_feat = self.inst_folder(reg_feat)
        return cls_score, bbox_pred, centerness, inst_feat, x

    def loss_sem_seg(self, sem_logits, gt_semantic_seg):
        start = int(self.mask_feat_stride // 2)
        gt_semantic_seg = gt_semantic_seg[:, :, start::self.mask_feat_stride,\
                                                start::self.mask_feat_stride]
        loss_seg = self.loss_seg(sem_logits, gt_semantic_seg.squeeze().long())
        return loss_seg

    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'centernesses'))
    def loss(self,
             cls_scores,
             bbox_preds,
             centernesses,
             inst_feats,
             img_feats,
             parse_feats,
             sem_logit,
             gt_bboxes,
             gt_labels,
             img_metas,
             gt_bboxes_ignore=None,
             **kwargs):
        """Compute loss of the head.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level,
                each is a 4D-tensor, the channel number is
                num_points * num_classes.
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level, each is a 4D-tensor, the channel number is
                num_points * 4.
            centernesses (list[Tensor]): centerness for each scale level, each
                is a 4D-tensor, the channel number is num_points * 1.
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        gt_parsings = kwargs['gt_parsings']
        gt_parse_labels = kwargs['gt_parse_labels']
        gt_parse_points = kwargs['gt_parse_points']
        gt_parse_bboxes = kwargs['gt_parse_bboxes']
        gt_semantic_seg = kwargs['gt_semantic_seg']
        gt_fine_parse_labels = kwargs['gt_fine_parse_labels']

        loss_seg = self.loss_sem_seg(sem_logit, gt_semantic_seg)

        assert len(cls_scores) == len(bbox_preds) == len(centernesses)
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        all_level_points_strides = self.prior_generator.grid_priors(
            featmap_sizes,
            dtype=bbox_preds[0].dtype,
            device=bbox_preds[0].device,
            with_stride=True)
        all_level_points = [i[:, :2] for i in all_level_points_strides]
        all_level_strides = [i[:, 2:] for i in all_level_points_strides]
        
        labels, bbox_targets, bch_labels, bch_pos_gt_inds = self.get_targets(
            all_level_points, gt_bboxes, gt_labels)

        num_imgs = cls_scores[0].size(0)
        # flatten cls_scores, bbox_preds and centerness
        flatten_cls_scores = [
            cls_score.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels)
            for cls_score in cls_scores
        ]
        flatten_bbox_preds = [
            bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
            for bbox_pred in bbox_preds
        ]
        flatten_centerness = [
            centerness.permute(0, 2, 3, 1).reshape(-1)
            for centerness in centernesses
        ]
        flatten_cls_scores = torch.cat(flatten_cls_scores)
        flatten_bbox_preds = torch.cat(flatten_bbox_preds)
        flatten_centerness = torch.cat(flatten_centerness)
        flatten_labels = torch.cat(labels)
        flatten_bbox_targets = torch.cat(bbox_targets)
        # repeat points to align with bbox_preds
        flatten_points = torch.cat(
            [points.repeat(num_imgs, 1) for points in all_level_points])

        # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
        bg_class_ind = self.num_classes
        pos_inds = ((flatten_labels >= 0)
                    & (flatten_labels < bg_class_ind)).nonzero().reshape(-1)
        num_pos = torch.tensor(
            len(pos_inds), dtype=torch.float, device=bbox_preds[0].device)
        num_pos = max(reduce_mean(num_pos), 1.0)
        loss_cls = self.loss_cls(
            flatten_cls_scores, flatten_labels, avg_factor=num_pos)

        pos_bbox_preds = flatten_bbox_preds[pos_inds]
        pos_centerness = flatten_centerness[pos_inds]
        pos_bbox_targets = flatten_bbox_targets[pos_inds]
        pos_centerness_targets = self.centerness_target(pos_bbox_targets)
        # centerness weighted iou loss
        centerness_denorm = max(
            reduce_mean(pos_centerness_targets.sum().detach()), 1e-6)

        if len(pos_inds) > 0:
            pos_points = flatten_points[pos_inds]
            pos_decoded_bbox_preds = self.bbox_coder.decode(
                pos_points, pos_bbox_preds)
            pos_decoded_target_preds = self.bbox_coder.decode(
                pos_points, pos_bbox_targets)
            loss_bbox = self.loss_bbox(
                pos_decoded_bbox_preds,
                pos_decoded_target_preds,
                weight=pos_centerness_targets,
                avg_factor=centerness_denorm)
            loss_centerness = self.loss_centerness(
                pos_centerness, pos_centerness_targets, avg_factor=num_pos)
        else:
            loss_bbox = pos_bbox_preds.sum()
            loss_centerness = pos_centerness.sum()
            loss_parsing = pos_bbox_preds.sum() * 0.

        parse_pos_inds = ((bch_labels >= 0)
            & (bch_labels < self.num_classes)).nonzero(as_tuple=False).squeeze(1)
        if self.use_sample:
            bch_num_inst = [i.shape[0] for i in bch_pos_gt_inds]
            bch_parse_pos_inds = parse_pos_inds.split(bch_num_inst, dim=0)

            sample_parse_pos_inds = []
            sample_pos_gt_inds = []
            for i, num_inst in enumerate(bch_num_inst):
                if num_inst > self.max_num_inst:
                    index = torch.LongTensor(random.sample(range(num_inst), self.max_num_inst))
                    sample_parse_pos_inds.append(bch_parse_pos_inds[i][index])
                    sample_pos_gt_inds.append(bch_pos_gt_inds[i][index])
                else:
                    sample_parse_pos_inds.append(bch_parse_pos_inds[i])
                    sample_pos_gt_inds.append(bch_pos_gt_inds[i])
            parse_pos_inds = torch.cat(sample_parse_pos_inds, dim=0)
            bch_pos_gt_inds = sample_pos_gt_inds

        # gt parsing maps
        B, C, H, W = parse_feats.shape
        img_h, img_w = H * self.mask_feat_stride, W * self.mask_feat_stride
        gt_parsing_targets = []
        start = int(self.parsing_out_stride // 2)
        for i, parsing in enumerate(gt_parsings):
            h, w = parsing.size()[1:]
            parsing = F.pad(parsing, (0, img_w - w, 0, img_h - h), "constant", 0)
            parsing = parsing[:, start::self.parsing_out_stride, start::self.parsing_out_stride]
            parsing = parsing.to(parse_feats.device)
            gt_parsing_targets.append(torch.index_select(parsing, 0, bch_pos_gt_inds[i]).contiguous())
        gt_parsing_targets = torch.cat(gt_parsing_targets, dim=0).long()

        # gt parsing points
        pos_gt_part_points = []
        pos_gt_part_labels = []
        pos_gt_part_bboxes = []
        pos_gt_part_fine_labels = []
        pos_gt_bboxes = []
        for i, pos_gt_inds in enumerate(bch_pos_gt_inds):
            pos_gt_part_points.append(torch.index_select(gt_parse_points[i], 0, pos_gt_inds))
            pos_gt_part_labels.append(torch.index_select(gt_parse_labels[i], 0, pos_gt_inds))
            pos_gt_part_bboxes.append(torch.index_select(gt_parse_bboxes[i], 0, pos_gt_inds))
            pos_gt_part_fine_labels.append(torch.index_select(gt_fine_parse_labels[i], 0, pos_gt_inds))
            pos_gt_bboxes.append(torch.index_select(gt_bboxes[i], 0, pos_gt_inds))
        pos_gt_part_points = torch.cat(pos_gt_part_points, dim=0)
        pos_gt_part_labels = torch.cat(pos_gt_part_labels, dim=0)
        pos_gt_part_bboxes = torch.cat(pos_gt_part_bboxes, dim=0)
        pos_gt_part_fine_labels = torch.cat(pos_gt_part_fine_labels, dim=0)
        pos_gt_bboxes = torch.cat(pos_gt_bboxes, dim=0)

        # cat all lvl and all img
        num_points = sum([center.size(0) for center in all_level_points])
        bch_inds = [
            torch.zeros((num_points), dtype=torch.int64, device=parse_feats.device) + i
            for i in range(B)]
        bch_inds = torch.cat(bch_inds, dim=0)
        bch_points = torch.cat(all_level_points, dim=0).repeat(num_imgs, 1)
        bch_strides = torch.cat(all_level_strides, dim=0).repeat(num_imgs, 1)
        bch_inst_feats = [
            inst_feat.permute(0, 2, 3, 1).reshape(
                num_imgs, -1, 9 * self.feat_channels) for inst_feat in inst_feats
        ]
        bch_inst_feats = torch.cat(bch_inst_feats, dim=1)
        bch_inst_feats = bch_inst_feats.reshape(-1, 9 * self.feat_channels)
        bch_img_feats = [
            img_feat.permute(0, 2, 3, 1).reshape(
                num_imgs, -1, self.feat_channels) for img_feat in img_feats
        ]
        bch_img_feats = torch.cat(bch_img_feats, dim=1)
        bch_img_feats = bch_img_feats.reshape(-1, self.feat_channels)

        # get pos inds
        pos_bch_ids = bch_inds[parse_pos_inds]
        pos_strides = bch_strides[parse_pos_inds][:, 0]
        pos_points = bch_points[parse_pos_inds]
        pos_inst_feats = bch_inst_feats[parse_pos_inds]

        # part level forward
        coarse_part_offsets = self.part_offset(pos_inst_feats)
        coarse_part_offsets = coarse_part_offsets.reshape(-1, self.num_coarse_classes, 2)
        coarse_part_cls_scores = self.coarse_part_cls(pos_inst_feats)
        part_cls_scores = self.part_cls(pos_inst_feats)

        # part level loss
        valid_mask = pos_gt_part_labels > 0
        gt_coarse_offsets = get_point_offset_target(
            pos_gt_part_points, pos_points, pos_gt_bboxes)
        loss_coarse_part_offset = self.loss_part_offset(
            coarse_part_offsets[valid_mask], gt_coarse_offsets[valid_mask])
        loss_coarse_part_cls = self.loss_part_cls(
            coarse_part_cls_scores, pos_gt_part_labels.float())
        loss_fine_part_cls = self.loss_part_cls(
            part_cls_scores, pos_gt_part_fine_labels.float())

        # get part level features
        with torch.no_grad():
            pred_coarse_part_points = decode_point(coarse_part_offsets, pos_points, pos_gt_bboxes)
            # valid_mask = (pos_gt_part_labels > 0)
            # pred_coarse_part_points[valid_mask] = pos_gt_part_points[valid_mask]
            batch_input_shape = img_metas[0]['batch_input_shape']
            pred_coarse_part_points[..., 0] = pred_coarse_part_points[..., 0].clamp(0, batch_input_shape[1])
            pred_coarse_part_points[..., 1] = pred_coarse_part_points[..., 1].clamp(0, batch_input_shape[0])

            accum_num_points = [0]
            for i, center in enumerate(all_level_points):
                accum_num_points.append(accum_num_points[i] + center.size(0))

            pos_part_inds = torch.zeros(
                (pred_coarse_part_points.shape[0], self.num_coarse_classes),
                dtype=torch.long,
                device=pred_coarse_part_points.device)

            for i, stride in enumerate(self.strides):
                valid_inds = (pos_strides == stride)
                if not valid_inds.any():
                    continue
                num_point = accum_num_points[i]
                points_ps = all_level_points[i].unsqueeze(0)
                pred_part_points_ps = pred_coarse_part_points[valid_inds].reshape(-1, 1, 2)
                dis = torch.sqrt(
                    (pred_part_points_ps[:, :, 0] - points_ps[:, :, 0])**2 + \
                    (pred_part_points_ps[:, :, 1] - points_ps[:, :, 1])**2)
                inds = dis.min(-1)[1]
                inds = num_point + inds
                inds = inds.reshape(-1, self.num_coarse_classes)
                pos_part_inds[valid_inds] = inds

            for b_id in range(num_imgs):
                valid_inds = (pos_bch_ids == b_id)
                pos_part_inds[valid_inds] += b_id * accum_num_points[-1]
            part_weights = coarse_part_cls_scores.sigmoid()

        pos_part_inds = pos_part_inds.reshape(-1)
        pos_coarse_part_feats = bch_img_feats[pos_part_inds]
        pos_coarse_part_feats = pos_coarse_part_feats.reshape(
            -1, self.num_coarse_classes, self.feat_channels)

        # n x c x d
        num_instances = pos_coarse_part_feats.shape[0]
        pos_coarse_part_feats_sample = pos_coarse_part_feats * part_weights.unsqueeze(-1)
        pos_coarse_part_feats_generate = self.coarse_part_feat(pos_inst_feats).reshape(
            num_instances, self.num_coarse_classes, -1)
        pos_coarse_part_feats = torch.cat(
            [pos_coarse_part_feats_sample, pos_coarse_part_feats_generate], dim=-1)

        # rep points features sample
        rep_points_offsets = self.rep_offset(pos_coarse_part_feats).reshape(
            num_instances*self.num_coarse_classes, -1, 2)
        pred_coarse_part_points = pred_coarse_part_points.reshape(-1, 2)
        # construct factors used for rescale bboxes
        img_factors = []
        valid_ratios = []
        b_img_h, b_img_w = batch_input_shape
        for i, (img_meta,) in enumerate(zip(img_metas)):
            img_h, img_w, _ = img_meta['img_shape']
            num_pos = (pos_bch_ids == i).sum()
            img_factor = pos_bch_ids.new_tensor([0, 0, img_w,
                                           img_h]).unsqueeze(0).repeat(
                                               num_pos * self.num_coarse_classes, 1)
            valid_ratio = pos_bch_ids.new_tensor(
                [img_w / b_img_w, img_h / b_img_h],
                dtype=torch.float32).unsqueeze(0).repeat(num_pos * self.num_coarse_classes, 1)
            img_factors.append(img_factor)
            valid_ratios.append(valid_ratio)
        img_factors = torch.cat(img_factors, 0)
        valid_ratios = torch.cat(valid_ratios, 0)

        rep_points = decode_point(
            rep_points_offsets, pred_coarse_part_points, img_factors)
        norm_rep_points = rep_points / img_factors.unsqueeze(1)[..., 2:]
        norm_rep_points = norm_rep_points * valid_ratios.unsqueeze(1)
        norm_rep_points = 2 * norm_rep_points - 1
        norm_rep_points = norm_rep_points.reshape(num_instances, self.num_coarse_classes, -1, 2)

        pos_img_feats = img_feats[0][pos_bch_ids]
        major_points_feats = F.grid_sample(pos_img_feats, norm_rep_points, padding_mode='zeros')
        major_points_feats = major_points_feats.permute(0, 2, 3, 1)
        rep_points_feats = torch.einsum('nckd,fc->nfkd', major_points_feats, self.assign_matrix)

        # rep points scores
        rep_points_scores = self.rep_cls(pos_coarse_part_feats).reshape(
            num_instances, self.num_coarse_classes, -1, self.num_rep_points).permute(0, 2, 1, 3)
        fine_part_scores = torch.einsum('nfck,fc->nfk', rep_points_scores, self.assign_matrix).sigmoid()

        # fine parts features
        fine_part_feat = rep_points_feats * fine_part_scores.unsqueeze(-1)
        fine_part_feat = self.rep_fusion(
            fine_part_feat.reshape(num_instances, self.num_fine_classes-1, -1))

        # params generation
        part_params = self.controller(fine_part_feat)
        inst_params = self.controller(self.inst_fusion(pos_inst_feats))

        parsing_logits = self.parsing_head_forward(
            inst_params, part_params, parse_feats, pos_points,
            pos_strides, pos_bch_ids, pred_coarse_part_points)
        parsing_logits = parsing_logits.reshape(-1, H*W)

        parsing_cnt_masks = []
        for parsing in gt_parsing_targets:
            parsing_mask = []
            for class_id in range(self.num_fine_classes):
                parsing_mask.append(parsing == class_id)
            parsing_mask = torch.stack(parsing_mask, dim=0)
            parsing_cnt_masks.append(parsing_mask)
        parsing_cnt_masks = torch.stack(parsing_cnt_masks, dim=0)
        parsing_targets = parsing_cnt_masks.reshape(-1, H*W)
        parsing_masks = parsing_targets.sum(-1) > 0
        parsing_targets = parsing_targets[parsing_masks]

        loss_dice = self.loss_dice(
            parsing_logits[parsing_masks],
            parsing_targets,
            avg_factor=parsing_targets.shape[0])
        loss_mask = self.loss_mask(
            parsing_logits[parsing_masks].reshape(-1, 1),
            1 - parsing_targets.reshape(-1).long(),
            avg_factor=parsing_targets.shape[0]*H*W)

        # rep points min max loss
        rep_bboxes = points2bbox(rep_points.reshape(-1, 2*self.num_rep_points), y_first=False)
        rep_bboxes_mean = rep_points.mean(dim=1)
        normalize_term = img_factors[..., 2:].repeat(1, 2)
        num_parts = pos_gt_part_labels.reshape(-1).sum()
        loss_rep_bbox = self.loss_rep_bbox(
            rep_bboxes / normalize_term,
            pos_gt_part_bboxes.reshape(-1, 4) / normalize_term,
            pos_gt_part_labels.reshape(-1, 1).repeat(1, 4).float(),
            num_parts)

        # rep points mean loss
        pos_gt_part_bboxes_center_x = pos_gt_part_bboxes.reshape(-1, 4)[:, 0::2].sum(-1) / 2
        pos_gt_part_bboxes_center_y = pos_gt_part_bboxes.reshape(-1, 4)[:, 1::2].sum(-1) / 2
        pos_gt_part_bboxes_centers = torch.stack(
            [pos_gt_part_bboxes_center_x, pos_gt_part_bboxes_center_y], dim=-1)

        loss_rep_bbox_mean = self.loss_rep_bbox(
            rep_bboxes_mean / normalize_term[:, :2],
            pos_gt_part_bboxes_centers / normalize_term[:, :2],
            pos_gt_part_labels.reshape(-1, 1).repeat(1, 2).float(),
            num_parts)

        return dict(
            loss_cls=loss_cls,
            loss_bbox=loss_bbox,
            loss_centerness=loss_centerness,
            loss_dice=loss_dice,
            loss_mask=loss_mask,
            loss_fine_part_cls=loss_fine_part_cls,
            loss_coarse_part_offset=loss_coarse_part_offset,
            loss_coarse_part_cls=loss_coarse_part_cls,
            loss_rep_bbox=loss_rep_bbox,
            loss_rep_bbox_mean=loss_rep_bbox_mean,
            loss_seg=loss_seg)

    def parsing_head_forward(self,
                             inst_params,
                             part_params,
                             parse_feats,
                             centers,
                             strides,
                             bch_inds,
                             part_centers=None,
                             part_weights=None):

        if len(parse_feats.shape) == 3:
            parse_feats = parse_feats.unsqueeze(0)
        BS, D, H, W = parse_feats.size()
        parse_feats_list = []
        for i in range(BS):
            valid_inds = (bch_inds == i)
            num_pos = valid_inds.sum()
            parse_feats_per_img = parse_feats[i].repeat(num_pos, 1, 1).reshape(1, -1, H, W)
            parse_feats_list.append(parse_feats_per_img)
        parse_feats = torch.cat(parse_feats_list, 1)

        rel_coords = relative_coordinate_maps(
            parse_feats.shape, centers, strides, self.parsing_out_stride)
        inst_parse_feats = torch.cat([
            rel_coords.view(-1, 2, H, W),
            parse_feats.reshape(-1, D, H, W)], dim=1)
        inst_parse_feats = inst_parse_feats.view(1, -1, H, W)

        total_pos_num = inst_params.shape[0]
        inst_weights, inst_biases = parse_dynamic_params(
            inst_params,
            self.parse_h_channels,
            1,
            self.weight_nums,
            self.bias_nums)
        
        inst_masks = dynamic_forward(
            inst_parse_feats,
            inst_weights,
            inst_biases,
            total_pos_num)
        inst_masks = inst_masks.reshape(total_pos_num, 1, H, W)

        # part masks
        parse_feats = parse_feats.reshape(total_pos_num, D, H, W)
        part_parse_feats = parse_feats.unsqueeze(1).repeat(1, self.num_fine_classes-1, 1, 1, 1)
        part_parse_feats = part_parse_feats.reshape(-1, D, H, W)

        # part_strides = strides[None].repeat(1, self.num_coarse_classes).reshape(-1)
        # coarse_rel_coords = relative_coordinate_maps(
        #     parse_feats.shape, part_centers, part_strides, self.parsing_out_stride)
        # coarse_rel_coords = coarse_rel_coords.reshape(-1, self.num_coarse_classes, 2, H, W)
        # coarse_rel_coords = coarse_rel_coords.min(dim=1)[0]
        # coarse_rel_coords = coarse_rel_coords.unsqueeze(1).repeat(1, self.num_fine_classes-1, 1, 1, 1)
        coarse_rel_coords = rel_coords.unsqueeze(1).repeat(1, self.num_fine_classes-1, 1, 1, 1)
        part_parse_feats = torch.cat([
            coarse_rel_coords.view(-1, 2, H, W),
            part_parse_feats.reshape(-1, D, H, W)], dim=1)
        part_parse_feats = part_parse_feats.view(1, -1, H, W)

        part_params = part_params.reshape(-1, self.total_params)
        total_pos_part_num = part_params.shape[0]
        part_weights, part_biases = parse_dynamic_params(
            part_params,
            self.parse_h_channels,
            1,
            self.weight_nums,
            self.bias_nums)

        part_masks = dynamic_forward(
            part_parse_feats,
            part_weights,
            part_biases,
            total_pos_part_num)
        part_masks = part_masks.reshape(total_pos_num, -1, H, W)
        final_masks = torch.cat([inst_masks, part_masks], dim=1)

        return final_masks

    def get_targets(self, points, gt_bboxes_list, gt_labels_list):
        """Compute regression, classification and centerness targets for points
        in multiple images.

        Args:
            points (list[Tensor]): Points of each fpn level, each has shape
                (num_points, 2).
            gt_bboxes_list (list[Tensor]): Ground truth bboxes of each image,
                each has shape (num_gt, 4).
            gt_labels_list (list[Tensor]): Ground truth labels of each box,
                each has shape (num_gt,).

        Returns:
            tuple:
                concat_lvl_labels (list[Tensor]): Labels of each level. \
                concat_lvl_bbox_targets (list[Tensor]): BBox targets of each \
                    level.
        """
        assert len(points) == len(self.regress_ranges)
        num_levels = len(points)
        # expand regress ranges to align with points
        expanded_regress_ranges = [
            points[i].new_tensor(self.regress_ranges[i])[None].expand_as(
                points[i]) for i in range(num_levels)
        ]
        # concat all levels points and regress ranges
        concat_regress_ranges = torch.cat(expanded_regress_ranges, dim=0)
        concat_points = torch.cat(points, dim=0)

        # the number of points per img, per lvl
        num_points = [center.size(0) for center in points]

        # get labels and bbox_targets of each image
        labels_list, bbox_targets_list, pos_gt_inds_list = multi_apply(
            self._get_target_single,
            gt_bboxes_list,
            gt_labels_list,
            points=concat_points,
            regress_ranges=concat_regress_ranges,
            num_points_per_lvl=num_points)

        # concat per image
        concat_batch_labels = torch.cat(labels_list, dim=0)

        # split to per img, per level
        labels_list = [labels.split(num_points, 0) for labels in labels_list]
        bbox_targets_list = [
            bbox_targets.split(num_points, 0)
            for bbox_targets in bbox_targets_list
        ]

        # concat per level image
        concat_lvl_labels = []
        concat_lvl_bbox_targets = []
        for i in range(num_levels):
            concat_lvl_labels.append(
                torch.cat([labels[i] for labels in labels_list]))
            bbox_targets = torch.cat(
                [bbox_targets[i] for bbox_targets in bbox_targets_list])
            if self.norm_on_bbox:
                bbox_targets = bbox_targets / self.strides[i]
            concat_lvl_bbox_targets.append(bbox_targets)

        return concat_lvl_labels, concat_lvl_bbox_targets, \
               concat_batch_labels, pos_gt_inds_list

    def _get_target_single(self, gt_bboxes, gt_labels, points, regress_ranges,
                           num_points_per_lvl):
        """Compute regression and classification targets for a single image."""
        num_points = points.size(0)
        num_gts = gt_labels.size(0)
        if num_gts == 0:
            return gt_labels.new_full((num_points,), self.num_classes), \
                   gt_bboxes.new_zeros((num_points, 4))

        areas = (gt_bboxes[:, 2] - gt_bboxes[:, 0]) * (
            gt_bboxes[:, 3] - gt_bboxes[:, 1])
        # TODO: figure out why these two are different
        # areas = areas[None].expand(num_points, num_gts)
        areas = areas[None].repeat(num_points, 1)
        regress_ranges = regress_ranges[:, None, :].expand(
            num_points, num_gts, 2)
        gt_bboxes = gt_bboxes[None].expand(num_points, num_gts, 4)
        xs, ys = points[:, 0], points[:, 1]
        xs = xs[:, None].expand(num_points, num_gts)
        ys = ys[:, None].expand(num_points, num_gts)

        left = xs - gt_bboxes[..., 0]
        right = gt_bboxes[..., 2] - xs
        top = ys - gt_bboxes[..., 1]
        bottom = gt_bboxes[..., 3] - ys
        bbox_targets = torch.stack((left, top, right, bottom), -1)

        if self.center_sampling:
            # condition1: inside a `center bbox`
            radius = self.center_sample_radius
            center_xs = (gt_bboxes[..., 0] + gt_bboxes[..., 2]) / 2
            center_ys = (gt_bboxes[..., 1] + gt_bboxes[..., 3]) / 2
            center_gts = torch.zeros_like(gt_bboxes)
            stride = center_xs.new_zeros(center_xs.shape)

            # project the points on current lvl back to the `original` sizes
            lvl_begin = 0
            for lvl_idx, num_points_lvl in enumerate(num_points_per_lvl):
                lvl_end = lvl_begin + num_points_lvl
                stride[lvl_begin:lvl_end] = self.strides[lvl_idx] * radius
                lvl_begin = lvl_end

            x_mins = center_xs - stride
            y_mins = center_ys - stride
            x_maxs = center_xs + stride
            y_maxs = center_ys + stride
            center_gts[..., 0] = torch.where(x_mins > gt_bboxes[..., 0],
                                             x_mins, gt_bboxes[..., 0])
            center_gts[..., 1] = torch.where(y_mins > gt_bboxes[..., 1],
                                             y_mins, gt_bboxes[..., 1])
            center_gts[..., 2] = torch.where(x_maxs > gt_bboxes[..., 2],
                                             gt_bboxes[..., 2], x_maxs)
            center_gts[..., 3] = torch.where(y_maxs > gt_bboxes[..., 3],
                                             gt_bboxes[..., 3], y_maxs)

            cb_dist_left = xs - center_gts[..., 0]
            cb_dist_right = center_gts[..., 2] - xs
            cb_dist_top = ys - center_gts[..., 1]
            cb_dist_bottom = center_gts[..., 3] - ys
            center_bbox = torch.stack(
                (cb_dist_left, cb_dist_top, cb_dist_right, cb_dist_bottom), -1)
            inside_gt_bbox_mask = center_bbox.min(-1)[0] > 0
        else:
            # condition1: inside a gt bbox
            inside_gt_bbox_mask = bbox_targets.min(-1)[0] > 0

        # condition2: limit the regression range for each location
        max_regress_distance = bbox_targets.max(-1)[0]
        inside_regress_range = (
            (max_regress_distance >= regress_ranges[..., 0])
            & (max_regress_distance <= regress_ranges[..., 1]))

        # if there are still more than one objects for a location,
        # we choose the one with minimal area
        areas[inside_gt_bbox_mask == 0] = INF
        areas[inside_regress_range == 0] = INF
        min_area, min_area_inds = areas.min(dim=1)

        labels = gt_labels[min_area_inds]
        labels[min_area == INF] = self.num_classes  # set as BG
        bbox_targets = bbox_targets[range(num_points), min_area_inds]
        pos_gt_inds = min_area_inds[labels < self.num_classes]

        return labels, bbox_targets, pos_gt_inds

    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'centernesses',
                          'kernel_preds'))
    def get_bboxes(self,
                   cls_scores,
                   bbox_preds,
                   centernesses,
                   inst_feats,
                   img_feats,
                   parse_feats,
                   sem_logit,
                   img_metas,
                   cfg=None,
                   rescale=False,
                   with_nms=True):
        # create dirs
        save_root = self.test_cfg.get('save_root', None)
        assert save_root is not None
        save_parse_path = os.path.join(save_root, 'val_parsing')
        save_seg_path = os.path.join(save_root, 'val_seg')
        if os.path.exists(save_parse_path) == False:
            os.makedirs(save_parse_path)
        if os.path.exists(save_seg_path) == False:
            os.makedirs(save_seg_path)

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

        results_list = []

        for img_id in range(len(img_metas)):
            img_meta = img_metas[img_id]

            cls_score = select_single_mlvl(cls_scores, img_id)
            bbox_pred = select_single_mlvl(bbox_preds, img_id)
            centerness_pred = select_single_mlvl(centernesses, img_id)
            inst_feat = select_single_mlvl(inst_feats, img_id)
            img_feat = select_single_mlvl(img_feats, img_id)
            parse_feat = parse_feats[img_id]
            results = self._get_bboxes_single(
                cls_score,
                bbox_pred,
                centerness_pred,
                inst_feat,
                parse_feat,
                img_feat,
                all_level_points,
                all_level_strides,
                img_meta,
                cfg,
                rescale=rescale,
                with_nms=with_nms,
                save_parse_path=save_parse_path,
                save_seg_path=save_seg_path)

            results_list.append(results)
        return results_list

    def _get_bboxes_single(self,
                           cls_score_list,
                           bbox_pred_list,
                           centerness_pred_list,
                           inst_feats,
                           parse_feats,
                           img_feats,
                           mlvl_points,
                           mlvl_strides,
                           img_meta,
                           cfg,
                           rescale=False,
                           with_nms=True,
                           save_parse_path=None,
                           save_seg_path=None,
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
        mlvl_inst_feats = []
        mlvl_img_feats = []
        flatten_mlvl_points = []
        flatten_mlvl_strides = []
        for cls_score, bbox_pred, centerness,\
            inst_feat, img_feat, points, strides in zip(
                cls_score_list, bbox_pred_list, centerness_pred_list,
                inst_feats, img_feats, mlvl_points, mlvl_strides):

            cls_score = cls_score.permute(1, 2, 0).reshape(
                -1, self.cls_out_channels).sigmoid()
            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4)
            centerness = centerness.permute(1, 2, 0).reshape(-1).sigmoid()
            inst_feat = inst_feat.permute(1, 2, 0).reshape(-1, 9 * self.feat_channels)
            img_feat = img_feat.permute(1, 2, 0).reshape(-1, self.feat_channels)

            results = filter_scores_and_topk(
                cls_score, cfg.score_thr, nms_pre,
                dict(bbox_pred=bbox_pred,
                     centerness=centerness,
                     inst_feat=inst_feat,
                     points=points,
                     strides=strides))
            scores, labels, _, filtered_results = results

            bbox_pred = filtered_results['bbox_pred']
            centerness = filtered_results['centerness']
            inst_feat = filtered_results['inst_feat']
            points = filtered_results['points']
            strides = filtered_results['strides']

            bboxes = self.bbox_coder.decode(
                points, bbox_pred, max_shape=img_shape)
            
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
            mlvl_labels.append(labels)
            mlvl_centerness.append(centerness)
            mlvl_inst_feats.append(inst_feat)
            mlvl_img_feats.append(img_feat)
            flatten_mlvl_points.append(points)
            flatten_mlvl_strides.append(strides)

        mlvl_bboxes = torch.cat(mlvl_bboxes)
        mlvl_scores = torch.cat(mlvl_scores)
        mlvl_labels = torch.cat(mlvl_labels)
        mlvl_inst_feats = torch.cat(mlvl_inst_feats)
        mlvl_img_feats = torch.cat(mlvl_img_feats)
        mlvl_centerness = torch.cat(mlvl_centerness)
        flatten_mlvl_points = torch.cat(flatten_mlvl_points)
        flatten_mlvl_strides = torch.cat(flatten_mlvl_strides)

        if rescale:
            mlvl_bboxes /= mlvl_bboxes.new_tensor(scale_factor)

        # nms process
        mlvl_scores = mlvl_scores * mlvl_centerness
        if with_nms:
            if mlvl_bboxes.numel() == 0:
                det_bboxes = torch.cat([mlvl_bboxes, mlvl_scores[:, None]], dim=-1)
                return det_bboxes, mlvl_labels

            det_bboxes, keep_idxs = batched_nms(mlvl_bboxes, mlvl_scores,
                                                mlvl_labels, cfg.nms)
            det_bboxes = det_bboxes[:cfg.max_per_img]
            det_labels = mlvl_labels[keep_idxs][:cfg.max_per_img]
            det_points = flatten_mlvl_points[keep_idxs][:cfg.max_per_img]
            det_strides = flatten_mlvl_strides[keep_idxs][:cfg.max_per_img][:, 0]
            pos_inst_feats = mlvl_inst_feats[keep_idxs][:cfg.max_per_img]

            # part level forward
            coarse_part_offsets = self.part_offset(pos_inst_feats)
            coarse_part_offsets = coarse_part_offsets.reshape(-1, self.num_coarse_classes, 2)
            coarse_part_cls_scores = self.coarse_part_cls(pos_inst_feats).sigmoid()
            det_part_scores = self.part_cls(pos_inst_feats).sigmoid()

            scale_det_bboxes = det_bboxes[:, :4] * det_bboxes[:, :4].new_tensor(scale_factor)
            pred_coarse_part_points = decode_point(coarse_part_offsets, det_points, scale_det_bboxes)
            batch_input_shape = img_meta['batch_input_shape']
            pred_coarse_part_points[..., 0] = pred_coarse_part_points[..., 0].clamp(0, batch_input_shape[1])
            pred_coarse_part_points[..., 1] = pred_coarse_part_points[..., 1].clamp(0, batch_input_shape[0])

            # img = cv2.imread(img_meta['filename'])
            # pred_coarse_part_points /= det_bboxes[:, :4].new_tensor(scale_factor[:2])
            # for i, points in enumerate(pred_coarse_part_points):
            #     if det_bboxes[i, 4] > 0.5:
            #         for point in points:
            #             x, y = [int(i) for i in point]
            #             cv2.circle(img, (x, y), radius=5, color=(20, 255, 0), thickness=5)
            # img = cv2.imwrite('point.jpg', img)

            accum_num_points = [0]
            for i, center in enumerate(mlvl_points):
                accum_num_points.append(accum_num_points[i] + center.size(0))

            pos_part_inds = torch.zeros(
                (pred_coarse_part_points.shape[0], self.num_coarse_classes),
                dtype=torch.long,
                device=pred_coarse_part_points.device)

            for i, stride in enumerate(self.strides):
                valid_inds = (det_strides == stride)
                if not valid_inds.any():
                    continue
                num_point = accum_num_points[i]
                points_ps = mlvl_points[i].unsqueeze(0)
                pred_part_points_ps = pred_coarse_part_points[valid_inds].reshape(-1, 1, 2)
                dis = torch.sqrt(
                    (pred_part_points_ps[:, :, 0] - points_ps[:, :, 0])**2 + \
                    (pred_part_points_ps[:, :, 1] - points_ps[:, :, 1])**2)
                inds = dis.min(-1)[1]
                inds = num_point + inds
                inds = inds.reshape(-1, self.num_coarse_classes)
                pos_part_inds[valid_inds] = inds

            pos_part_inds = pos_part_inds.reshape(-1)
            pos_coarse_part_feats = mlvl_img_feats[pos_part_inds]
            pos_coarse_part_feats = pos_coarse_part_feats.reshape(
                -1, self.num_coarse_classes, self.feat_channels)
            
            # n x c x d
            num_instances = pos_coarse_part_feats.shape[0]
            pos_coarse_part_feats_sample = pos_coarse_part_feats * coarse_part_cls_scores.unsqueeze(-1)
            pos_coarse_part_feats_generate = self.coarse_part_feat(pos_inst_feats).reshape(
                num_instances, self.num_coarse_classes, -1)
            pos_coarse_part_feats = torch.cat(
                [pos_coarse_part_feats_sample, pos_coarse_part_feats_generate], dim=-1)
            pos_bch_inds = pos_coarse_part_feats.new_zeros((num_instances), dtype=torch.long)

            # rep points features sample
            rep_points_offsets = self.rep_offset(pos_coarse_part_feats).reshape(
                num_instances*self.num_coarse_classes, -1, 2)
            pred_coarse_part_points = pred_coarse_part_points.reshape(-1, 2)

            # construct factors used for rescale bboxes
            b_img_h, b_img_w = batch_input_shape
        
            img_h, img_w, _ = img_meta['img_shape']
            num_pos = (pos_bch_inds == 0).sum()
            img_factors = pos_bch_inds.new_tensor([0, 0, img_w,
                                        img_h]).unsqueeze(0).repeat(
                                            num_pos * self.num_coarse_classes, 1)
            valid_ratios = pos_bch_inds.new_tensor(
                [img_w / b_img_w, img_h / b_img_h],
                dtype=torch.float32).unsqueeze(0).repeat(num_pos * self.num_coarse_classes, 1)

            rep_points = decode_point(
                rep_points_offsets, pred_coarse_part_points, img_factors)

            # img = cv2.imread(img_meta['filename'])
            # rep_points_tmp = rep_points / det_bboxes[:, :4].new_tensor(scale_factor[:2])
            # rep_points_tmp = rep_points_tmp.reshape(num_pos, self.num_coarse_classes, -1, 2)
            # for i, rep_points_per_coarse in enumerate(rep_points_tmp):
            #     if det_bboxes[i, 4] > 0.5:
            #         for j, rep_point in enumerate(rep_points_per_coarse[2:3]):
            #             color = (j * 20, 255, 0)
            #             for point in rep_point:
            #                 x, y = [int(p) for p in point]
            #                 cv2.circle(img, (x, y), radius=3, color=color, thickness=3)
            # img = cv2.imwrite('point.jpg', img)

            norm_rep_points = rep_points / img_factors.unsqueeze(1)[..., 2:]
            norm_rep_points = norm_rep_points * valid_ratios.unsqueeze(1)
            norm_rep_points = 2 * norm_rep_points - 1
            norm_rep_points = norm_rep_points.reshape(num_instances, self.num_coarse_classes, -1, 2)

            pos_img_feats = img_feats[0][None][pos_bch_inds]
            major_points_feats = F.grid_sample(pos_img_feats, norm_rep_points, padding_mode='zeros')
            major_points_feats = major_points_feats.permute(0, 2, 3, 1)
            rep_points_feats = torch.einsum('nckd,fc->nfkd', major_points_feats, self.assign_matrix)

            # rep points scores
            rep_points_scores = self.rep_cls(pos_coarse_part_feats).reshape(
                num_instances, self.num_coarse_classes, -1, self.num_rep_points).permute(0, 2, 1, 3)
            fine_part_scores = torch.einsum('nfck,fc->nfk', rep_points_scores, self.assign_matrix).sigmoid()

            # fine parts features
            fine_part_feat = rep_points_feats * fine_part_scores.unsqueeze(-1)
            fine_part_feat = self.rep_fusion(
                fine_part_feat.reshape(num_instances, self.num_fine_classes-1, -1))

            # params generation
            part_params = self.controller(fine_part_feat)
            inst_params = self.controller(self.inst_fusion(pos_inst_feats))

            parsing_logits = self.parsing_head_forward(
                inst_params, part_params, parse_feats, det_points,
                det_strides, pos_bch_inds, pred_coarse_part_points)
            # parsing_logits = parsing_logits.softmax(dim=1)
            parsing_logits = parsing_logits.sigmoid()

            bg_scores = det_part_scores.new_ones((det_part_scores.shape[0], 1))
            det_part_scores = torch.cat((bg_scores, det_part_scores), dim=-1)
            parsing_logits = torch.einsum("qc,qchw->qchw", det_part_scores, parsing_logits)

            parsing_logits = aligned_bilinear(parsing_logits, self.mask_feat_stride // self.parsing_out_stride)
            if rescale:
                parsing_logits = aligned_bilinear(parsing_logits, self.parsing_out_stride)
                parsing_logits = parsing_logits[:, :, :img_shape[0], :img_shape[1]]
                if ori_shape[0] > 2500 or ori_shape[1] > 2500:
                    parsing_logits = parsing_logits.cpu()
                parsing_logits = F.interpolate(
                    parsing_logits,
                    size=(ori_shape[0], ori_shape[1]),
                    mode='bilinear',
                    align_corners=False)
            parsing_maps = parsing_logits.argmax(dim=1)
            parsing_maps = parsing_maps.cpu().numpy()

        # save results
        filename = img_meta['ori_filename']
        for i in range(det_bboxes.shape[0]):
            parsing_name = filename.split('.jpg')[0] + '-' + str(i) + '.png'
            cv2.imwrite(os.path.join(save_parse_path, parsing_name), parsing_maps[i])

        seg = np.zeros(parsing_maps[0].shape, dtype=np.uint8)
        for i in range(det_bboxes.shape[0]-1, -1, -1):
            if float(det_bboxes[i][4]) > 0.2:
                parsing_map = parsing_maps[i].astype(np.uint8)
                seg[parsing_map > 0] = parsing_map[parsing_map > 0]
        seg_name = filename.replace('jpg', 'png')
        cv2.imwrite(os.path.join(save_seg_path, seg_name), seg)
        return det_bboxes, det_labels
