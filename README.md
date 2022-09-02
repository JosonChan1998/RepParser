# RepParser
RepParser: End-to-End Multiple Human Parsing with Representative Parts

## Installation
- pytorch 1.7.1
- python 3.7.0
- [mmdetection 2.20.0](https://mmdetection.readthedocs.io/en/latest/get_started.html#installation)

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
  |  ├─repparser_r50_45k_cihp
  |  |  ├─iter_45000.pth
  ```

## Results

### CIHP

|  Backbone    |  LR  | mIOU | APvol | AP_p50 | PCP50 | download |
|--------------|:----:|:----:|:-----:|:------:|:-----:|:--------:|
|  R-50        |  1x  | 52.9 | 51.9  |  57.5  |  55.7 |[model](https://drive.google.com/file/d/1-m83sJcu9fsRNE4pNTBLmkOB8cKhPyCK/view?usp=sharing) |

## Evaluation
```
# inference
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ./tools/dist_test.sh configs/repparser/repparser_r50_45k_cihp.py work_dirs/repparser_r50_45k_cihp/iter_45000.pth 8 --eval bbox --eval-options "jsonfile_prefix=work_dirs/repparser_r50_45k_cihp/repparser_r50_45k_cihp_val_result"

# eval, noted that should change the json path produce by previous step.
python utils/eval.py
```

## Training

Coming soon...

