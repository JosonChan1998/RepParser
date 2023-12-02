import os
import mmcv
import numpy as np
import os.path as osp

from .builder import DATASETS
from .coco import CocoDataset

@DATASETS.register_module()
class CIHP(CocoDataset):

    CLASSES = ('person',)

    def __init__(self,
                 parsing_prefix=None,
                 *arg,
                 **kwargs):
        super(CIHP, self).__init__(*arg, **kwargs)

        # add parsing prefix
        self.parsing_prefix = parsing_prefix
        if self.data_root is not None:
            if not (self.parsing_prefix is None or osp.isabs(self.parsing_prefix)):
                self.parsing_prefix = osp.join(self.data_root, self.parsing_prefix)
        
        # add num_parsing class
        self.parsing_classes = self.coco.cats[1]['parsing']
    
    def pre_pipeline(self, results):
        """Prepare results dict for pipeline."""
        results['img_prefix'] = self.img_prefix
        results['seg_prefix'] = self.seg_prefix
        results['parsing_prefix'] = self.parsing_prefix
        results['proposal_file'] = self.proposal_file
        results['parsing_classes'] = self.parsing_classes
        results['bbox_fields'] = []
        results['mask_fields'] = []
        results['seg_fields'] = []
        results['parse_fields'] = []

    def _parse_ann_info(self, img_info, ann_info):
        """Parse bbox and mask annotation.

        Args:
            ann_info (list[dict]): Annotation info of an image.
            with_mask (bool): Whether to parse mask annotations.

        Returns:
            dict: A dict containing the following keys: bboxes, bboxes_ignore,\
                labels, masks, seg_map. "masks" are raw annotations and not \
                decoded into binary masks.
        """
        gt_bboxes = []
        gt_labels = []
        gt_bboxes_ignore = []
        gt_masks_ann = []
        gt_parsing_suffixs = []
        for i, ann in enumerate(ann_info):
            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            inter_w = max(0, min(x1 + w, img_info['width']) - max(x1, 0))
            inter_h = max(0, min(y1 + h, img_info['height']) - max(y1, 0))
            if inter_w * inter_h == 0:
                continue
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            if ann['category_id'] not in self.cat_ids:
                continue
            bbox = [x1, y1, x1 + w, y1 + h]
            if ann.get('iscrowd', False):
                gt_bboxes_ignore.append(bbox)
            else:
                gt_bboxes.append(bbox)
                gt_labels.append(self.cat2label[ann['category_id']])
                gt_masks_ann.append(ann.get('segmentation', None))
                gt_parsing_suffixs.append(ann.get('parsing', None))

        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)

        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

        seg_map = img_info['filename'].replace('jpg', 'png')

        ann = dict(
            bboxes=gt_bboxes,
            labels=gt_labels,
            bboxes_ignore=gt_bboxes_ignore,
            masks=gt_masks_ann,
            seg_map=seg_map,
            parsing_suffixs=gt_parsing_suffixs)

        return ann
    
    def _det2json(self, results):
        """Convert detection results to COCO json style."""
        json_results = []
        for idx in range(len(self)):
            file_name = self.data_infos[idx]['file_name']
            img_id = self.img_ids[idx]
            result = results[idx]
            for label in range(len(result)):
                bboxes = result[label]
                for i in range(bboxes.shape[0]):
                    data = dict()
                    data['image_id'] = img_id
                    data['bbox'] = self.xyxy2xywh(bboxes[i])
                    data['score'] = float(bboxes[i][4])
                    data['parsing'] = file_name.split('.jpg')[0] + '-' + str(i) + '.png'
                    data['category_id'] = self.cat_ids[label]
                    json_results.append(data)
        return json_results
