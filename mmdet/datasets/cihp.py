import cv2
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
        results['parsing_fields'] = []

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

    def _parsing2json(self, results, outfile_prefix, seg_path):
        """Convert instance segmentation results to COCO json style."""

        parsing_json_results = []
        for idx in range(len(self)):
            file_name = self.data_infos[idx]['file_name']
            img_id = self.img_ids[idx]
            if len(results[idx]) == 2:
                det, parsing = results[idx]
                score = None
            else:
                det, parsing, score = results[idx]
                score = score[0]
            det = det[0]
            parsing = parsing[0]
            for label in range(len(det)):
                # bbox results
                bboxes = det[label]
                segms = parsing[label]
                if score is not None:
                    parsing_score = score[label]
                if bboxes.shape[0] > 0:
                    seg_per_img = np.zeros((segms[0].shape[0], segms[0].shape[1]), dtype=np.uint8)
                for i in range(bboxes.shape[0]):
                    data = dict()
                    data['image_id'] = img_id
                    data['bbox'] = self.xyxy2xywh(bboxes[i])
                    data['score'] = float(bboxes[i][4])
                    data['category_id'] = self.cat_ids[label]
                    parsing_name = file_name.split('.jpg')[0] + '-' + str(i) + '.png'
                    if score is not None:
                        data['parsing_score'] = float(parsing_score[i])

                    data['parsing'] = parsing_name

                    cv2.imwrite(outfile_prefix + parsing_name, segms[i])
                    parsing_json_results.append(data)
                
                num_bbox = range(bboxes.shape[0])
                for i in reversed(num_bbox):
                    if float(bboxes[i][4]) > 0.2:
                        seg_per_img = cv2.bitwise_or(seg_per_img, segms[i].astype(np.uint8))
                if bboxes.shape[0] > 0:
                    seg_name = file_name.replace('jpg', 'png')
                    cv2.imwrite(os.path.join(seg_path, seg_name), seg_per_img)

        return parsing_json_results

    def results2json(self, results, outfile_prefix):
        """Dump the detection results to a COCO style json file.

        There are 3 types of results: proposals, bbox predictions, mask
        predictions, and they have different data types. This method will
        automatically recognize the type, and dump them to json files.

        Args:
            results (list[list | tuple | ndarray]): Testing results of the
                dataset.
            outfile_prefix (str): The filename prefix of the json files. If the
                prefix is "somepath/xxx", the json files will be named
                "somepath/xxx.bbox.json", "somepath/xxx.segm.json",
                "somepath/xxx.proposal.json".

        Returns:
            dict[str: str]: Possible keys are "bbox", "segm", "proposal", and \
                values are corresponding filenames.
        """
        result_files = dict()
        if len(results[0]) > 1:
            filename_split = outfile_prefix.split('/')
            outpath = filename_split[0]
            for i in filename_split[1:-1]:
                outpath = outpath +  '/' + i
            outpath += '/val_parsing/'
            seg_path = outpath.replace('val_parsing', 'val_seg')
            if os.path.exists(outpath) == False:
                os.makedirs(outpath)
            if os.path.exists(seg_path) == False:
                os.makedirs(seg_path)
            json_results = self._parsing2json(results, outpath, seg_path)
            result_files['bbox'] = f'{outfile_prefix}.bbox.json'
            mmcv.dump(json_results, result_files['bbox'])
        elif isinstance(results[0], list):
            json_results = self._det2json(results)
            result_files['bbox'] = f'{outfile_prefix}.bbox.json'
            result_files['proposal'] = f'{outfile_prefix}.bbox.json'
            mmcv.dump(json_results, result_files['bbox'])
        elif isinstance(results[0], tuple):
            json_results = self._segm2json(results)
            result_files['bbox'] = f'{outfile_prefix}.bbox.json'
            result_files['proposal'] = f'{outfile_prefix}.bbox.json'
            result_files['segm'] = f'{outfile_prefix}.segm.json'
            mmcv.dump(json_results[0], result_files['bbox'])
            mmcv.dump(json_results[1], result_files['segm'])
        elif isinstance(results[0], np.ndarray):
            json_results = self._proposal2json(results)
            result_files['proposal'] = f'{outfile_prefix}.proposal.json'
            mmcv.dump(json_results, result_files['proposal'])
        else:
            raise TypeError('invalid type of results')
        return result_files
