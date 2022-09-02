import cv2
import numpy as np
import os
from tqdm import tqdm


class SemSegEvaluator:
    def __init__(self, cocoGt, cocoDt, num_classes, seg_prefix=None):
        self.cocoGt = cocoGt  # ground truth COCO API
        self.cocoDt = cocoDt  # detections COCO API
        self.num_classes = num_classes
        self.seg_prefix = seg_prefix
        self.ids = sorted(cocoGt.getImgIds())
        self.stats = dict()

    def fast_hist(self, gt, dt):
        k = (gt >= 0) & (dt < self.num_classes)
        return np.bincount(
            self.num_classes * gt[k].astype(int) + dt[k], minlength=self.num_classes ** 2
        ).reshape(self.num_classes, self.num_classes)

    def evaluate(self):
        print('Evaluating Semantic Segmentation predictions')
        hist = np.zeros((self.num_classes, self.num_classes))
        for i in tqdm(self.ids, desc='Calculating IoU ..'):
            # get gt img
            ann_ids = self.cocoGt.getAnnIds(imgIds=i, iscrowd=None)
            gt = self.cocoGt.loadAnns(ann_ids)
            image_name = gt['file_name'].replace('jpg', 'png')
            gt_png = cv2.imread(os.path.join(self.seg_prefix, image_name), 0)

            # get pred png
            ann_ids = self.cocoDt.getAnnIds(imgIds=i, iscrowd=None)
            dt = self.cocoDt.loadAnns(ann_ids)
            pre_png = dt['seg']

            assert gt_png.shape == pre_png.shape, '{} VS {}'.format(str(gt_png.shape), str(pre_png.shape))
            gt = gt_png.flatten()
            pre = pre_png.flatten()
            hist += self.fast_hist(gt, pre)

        def mean_iou(overall_h):
            iu = np.diag(overall_h) / (overall_h.sum(1) + overall_h.sum(0) - np.diag(overall_h) + 1e-10)
            return iu, np.nanmean(iu)

        def per_class_acc(overall_h):
            acc = np.diag(overall_h) / (overall_h.sum(1) + 1e-10)
            return np.nanmean(acc)

        def pixel_wise_acc(overall_h):
            return np.diag(overall_h).sum() / overall_h.sum()

        iou, miou = mean_iou(hist)
        mean_acc = per_class_acc(hist)
        pixel_acc = pixel_wise_acc(hist)
        self.stats.update(dict(IoU=iou, mIoU=miou, MeanACC=mean_acc, PixelACC=pixel_acc))

    def accumulate(self, p=None):
        pass

    def summarize(self):
        iStr = ' {:<18} @[area={:>6s}] = {:0.4f}'
        for k, v in self.stats.items():
            if k == 'IoU':
                continue
            print(iStr.format(k, 'all', v))