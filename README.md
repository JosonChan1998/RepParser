# RepParser
ReSParser: Fully Convolutional Multiple Human Parsing With Representative Sets

## Installation
- pytorch 1.7.1
- python 3.7.0
- [mmdetection 2.25.2](https://mmdetection.readthedocs.io/en/latest/get_started.html#installation)

## Dataset
You need to download the datasets and annotations follwing this repo's formate


Make sure to put the files as the following structure:

```
  ├─data
  │  CIHP
  │  │  ├─train_img
  │  │  ├─train_parsing
  │  │  ├─train_seg
  │  │  ├─val_img
  │  │  ├─val_parsing
  │  │  ├─val_seg
  │  │  │─annotations
  |
  ├─work_dirs
  |  ├─resparser_r50_fpn_3x_cihp
  |  |  ├─epoch_75.pth
  ```

## Results

### CIHP

|  Backbone    |  LR  | mIOU | APvol | AP_p50 | PCP50 | download |
|--------------|:----:|:----:|:-----:|:------:|:-----:|:--------:|
|  R-50        |  1x  | 53.6 | 53.7  |  62.7  |  59.3 |[model](https://drive.google.com/file/d/1IkMpcTjqNtisBZ128AB4kqkTsnklU_04/view?usp=sharing) |
|  R-50        |  3x  | 57.0 | 55.2  |  66.2  |  62.6 |[model](https://drive.google.com/file/d/1D-R3e_76z_lP23A7W66U16v3C1DGlb9n/view?usp=sharing) |

## Evaluation
```
# inference
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ./tools/dist_test.sh configs/ReSParser/resparser_r50_fpn_3x_cihp.py work_dirs/resparser_r50_fpn_3x_cihp/epoch_75.pth 8 --eval bbox --eval-options "jsonfile_prefix=work_dirs/resparser_r50_fpn_3x_cihp/resparser_r50_fpn_3x_cihp_val_result"

# eval, noted that should change the json path produce by previous step.
python utils/eval.py
```

## Training
```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ./tools/dist_train.sh configs/ReSParser/resparser_r50_fpn_3x_cihp.py 8
```

## Citation
```
@misc{chen2022repparser,
      title={RepParser: End-to-End Multiple Human Parsing with Representative Parts}, 
      author={Xiaojia Chen and Xuanhan Wang and Lianli Gao and Jingkuan Song},
      year={2022},
      eprint={2208.12908},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}

@ARTICLE{10138431,
  author={Dai, Yan and Chen, Xiaojia and Wang, Xuanhan and Pang, Minghui and Gao, Lianli and Shen, Heng Tao},
  journal={IEEE Transactions on Multimedia}, 
  title={ReSParser: Fully Convolutional Multiple Human Parsing With Representative Sets}, 
  year={2024},
  volume={26},
  number={},
  pages={1384-1394},
  keywords={Pipelines;Task analysis;Kernel;Semantics;Pose estimation;Detectors;Crops;Multiple human parsing;human instance-level analysis;instance-aware modeling},
  doi={10.1109/TMM.2023.3281070}}
```