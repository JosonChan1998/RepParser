import time
import copy
from tkinter.messagebox import NO
import cv2
from matplotlib.pyplot import flag
import numpy as np
import os.path as osp
from collections import defaultdict
from pycocotools.cocoeval import COCOeval as _COCOeval, Params

class HPeval(_COCOeval):
    '''
    pass
    '''
    def __init__(self,
                 hpGT=None,
                 hpDt=None,
                 iouType=None,
                 gt_dirs=None,
                 dt_dirs=None,
                 num_classes=20):
        if not iouType:
            print('iouType not specified. use default iouType segm')
        self.cocoGt   = hpGT              # ground truth COCO API
        self.cocoDt   = hpDt              # detections COCO API
        self.evalImgs = defaultdict(list)   # per-image per-category evaluation results [KxAxI] elements
        self.eval     = {}                  # accumulated evaluation results
        self._gts = defaultdict(list)       # gt for evaluation
        self._dts = defaultdict(list)       # dt for evaluation
        self.params = HPParams(iouType=iouType)
        self._paramsEval = {}               # parameters for evaluation
        self.stats = []                     # result summarization
        self.ious = {}                      # ious between all gts and dts
        if not hpGT is None:
            self.params.imgIds = sorted(hpDt.getImgIds())
            self.params.catIds = sorted(hpDt.getCatIds())
        self.num_classes = num_classes
        self.gt_dirs = gt_dirs
        self.dt_dirs = dt_dirs

    def evaluate(self):
        '''
        Run per image evaluation on given images and store results
         (a list of dict) in self.evalImgs
        :return: None
        '''
        tic = time.time()
        print('Running per image evaluation...')
        p = self.params
        # add backward compatibility if useSegm is specified in params
        if p.useSegm is not None:
            p.iouType = 'segm' if p.useSegm == 1 else 'bbox'
            print('useSegm (deprecated) is not None. Running {} evaluation'.
                  format(p.iouType))
        print('Evaluate annotation type *{}*'.format(p.iouType))
        p.imgIds = list(np.unique(p.imgIds))
        if p.useCats:
            p.catIds = list(np.unique(p.catIds))
        p.maxDets = sorted(p.maxDets)
        self.params = p

        self._prepare()
        # loop through images, area range, max detection number
        catIds = p.catIds if p.useCats else [-1]

        if p.iouType == 'segm' or p.iouType == 'bbox':
            computeIoU = self.computeIoU
        elif p.iouType == 'keypoints':
            computeIoU = self.computeOks
        elif p.iouType == 'parse':
            computeIoU = self.computeMIoU
        self.ious = {(imgId, catId): computeIoU(imgId, catId)
                     for imgId in p.imgIds for catId in catIds}

        evaluateImg = self.evaluateImg
        maxDet = p.maxDets[-1]
        self.evalImgs = [
            evaluateImg(imgId, catId, areaRng, maxDet) for catId in catIds
            for areaRng in p.areaRng for imgId in p.imgIds
        ]
        self._paramsEval = copy.deepcopy(self.params)
        toc = time.time()
        print('DONE (t={:0.2f}s).'.format(toc - tic))
    
    def _cal_one_mean_iou(self, pre, gt):
        k = (gt >= 0) & (gt < self.num_classes)
        hist = np.bincount(
            self.num_classes * gt[k].astype(int) + pre[k], minlength=self.num_classes ** 2
        ).reshape(self.num_classes, self.num_classes).astype(np.float)
        num_cor_pix = np.diag(hist)
        num_gt_pix = hist.sum(1)
        union = num_gt_pix + hist.sum(0) - num_cor_pix
        iu = num_cor_pix / union
        return iu

    def computeMIoU(self, imgId, catId):
        p = self.params
        if p.useCats:
            gt = self._gts[imgId, catId]
            dt = self._dts[imgId, catId]
        else:
            gt = [_ for cId in p.catIds for _ in self._gts[imgId, cId]]
            dt = [_ for cId in p.catIds for _ in self._dts[imgId, cId]]
        if len(gt) == 0 and len(dt) == 0:
            return []
        inds = np.argsort([-d['score'] for d in dt], kind='mergesort')
        dt = [dt[i] for i in inds]
        if len(dt) > p.maxDets[-1]:
            dt = dt[0:p.maxDets[-1]]

        if p.iouType == 'segm':
            g = [g['segmentation'] for g in gt]
            d = [d['segmentation'] for d in dt]
        elif p.iouType == 'bbox':
            g = [g['bbox'] for g in gt]
            d = [d['bbox'] for d in dt]
        elif p.iouType == 'parse':
            gt_suffixs = [g['parsing'] for g in gt]
            dt_suffixs = [d['parsing'] for d in dt]
            g = []
            d = []
            for i, _ in enumerate(gt_suffixs):
                g_filename = osp.join(self.gt_dirs, gt_suffixs[i])
                g.append(cv2.imread(g_filename, cv2.IMREAD_GRAYSCALE))
            for i, _ in enumerate(dt_suffixs):
                d_filename = osp.join(self.dt_dirs, dt_suffixs[i])
                d.append(cv2.imread(d_filename, cv2.IMREAD_GRAYSCALE))
        else:
            raise TypeError

        # compute iou between each dt and gt region
        ious = np.zeros((len(d), len(g)))
        for j, dt in enumerate(d):
            for k, gt in enumerate(g):
                iu = self._cal_one_mean_iou(dt, gt)
                miou = np.nanmean(iu)
                ious[j, k] = miou
        return ious

class HPParams(Params):
    '''
    Params specific for Human Parsing eval
    '''
    def __init__(self, iouType='segm'):
        if iouType == 'segm' or iouType == 'bbox':
            self.setDetParams()
        elif iouType == 'keypoints':
            self.setKpParams()
        elif iouType == 'parse':
            self.setParseParams()
        else:
            raise Exception('iouType not supported')
        self.iouType = iouType
        # useSegm is deprecated
        self.useSegm = None
    
    def setParseParams(self):
        self.iouThrs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        self.pariouThrs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        self.maskiouThrs = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
        self.maxDets = [100]
        self.areaRng = [[0 ** 2, 1e5 ** 2]]
        self.areaRngLbl = ['all']
        self.useCats = 1

COCOeval = HPeval