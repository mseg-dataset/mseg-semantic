

This repo includes the **semantic segmentation pre-trained models, training and inference code** for the paper:

**MSeg: A Composite Dataset for Multi-domain Semantic Segmentation** (CVPR 2020, Official Repo) [[PDF]](http://vladlen.info/papers/MSeg.pdf)
<br>
[John Lambert*](https://johnwlambert.github.io/),
[Zhuang Liu*](https://liuzhuang13.github.io/),
[Ozan Sener](http://ozansener.net/),
[James Hays](https://www.cc.gatech.edu/~hays/),
[Vladlen Koltun](http://vladlen.info/)
<br>
Presented at [CVPR 2020](http://cvpr2018.thecvf.com/)

<p align="left">
  <img src="https://user-images.githubusercontent.com/62491525/83895683-094caa00-a721-11ea-8905-2183df60bc4f.gif" height="280">
  <img src="https://user-images.githubusercontent.com/62491525/83893966-aeb24e80-a71e-11ea-84cc-80e591f91ec0.gif" height="280">
</p>
<p align="left">
  <img src="https://user-images.githubusercontent.com/62491525/83895915-57fa4400-a721-11ea-8fa9-3c2ff0361080.gif" height="280">
  <img src="https://user-images.githubusercontent.com/62491525/83895972-73654f00-a721-11ea-8438-7bd43b695355.gif" height="280"> 
</p>

<p align="left">
  <img src="https://user-images.githubusercontent.com/62491525/83893958-abb75e00-a71e-11ea-978c-ab4080b4e718.gif" height="280">
  <img src="https://user-images.githubusercontent.com/62491525/83895490-c094f100-a720-11ea-9f85-cf4c6b030e73.gif" height="280">
</p>

<p align="left">
  <img src="https://user-images.githubusercontent.com/62491525/83895811-35682b00-a721-11ea-9641-38e3b2c1ad0e.gif" height="280">
  <img src="https://user-images.githubusercontent.com/62491525/83895963-6e080480-a721-11ea-98b6-49835d9e733a.gif" height="280">
</p>

<p align="center">
  <img src="https://user-images.githubusercontent.com/62491525/83896026-8710b580-a721-11ea-86d2-a0fff9c6e26e.gif" height="280">
  <img src="" height="280">
</p>


This repo is the second of 4 repos that introduce our work. It provides utilities to train semantic segmentation models, using a HRNet-W48 or PSPNet backbone, sufficient to train a winning entry on the [WildDash](https://wilddash.cc/benchmark/summary_tbl?hc=semantic_rob) benchmark).

Three additional repos will be introduced in April and May 2020:
- [` mseg-api`](https://github.com/mseg-dataset/mseg-api): utilities to download the MSeg dataset, prepare the data on disk in a unified taxonomy, on-the-fly mapping to a unified taxonomy during training.
- `mseg-panoptic`: provides Panoptic-FPN and Mask-RCNN training, based on Detectron2
- `mseg-mturk`: provides utilities to perform large-scale Mechanical Turk re-labeling

### Dependencies

Install the `mseg` model from [`mseg-api`](https://github.com/mseg-dataset/mseg-api)

### Install the MSeg-Semantic module:

* `mseg_semantic` can be installed as a python package using

        pip install -e /path_to_root_directory_of_the_repo/

Make sure that you can run `import mseg_semantic` in python, and you are good to go!


## MSeg Pre-trained Models

Each model is 528 MB in size. We provide download links and multi-scale testing results below:

Nicknames: VOC = PASCAL VOC, WD = WildDash, SN = ScanNet

|    Model                | Training Set    |  Training <br> Taxonomy | VOC <br> mIoU | PASCAL <br> Context <br> mIoU | CamVid <br> mIoU | WD <br> mIoU | KITTI <br> mIoU | SN <br> mIoU | h. mean | Download <br> Link        |
| :---------------------: | :------------:  | :--------------------:  | :----------:  | :---------------------------: | :--------------: | :----------: | :-------------: | :----------: | :----:  | :--------------: |
| MSeg (1M)               | MSeg train      | Universal               | 70.8          | 42.9                          | 83.1             | 63.1         | 63.7            | 48.4         | 59.0    | [Google Drive](https://drive.google.com/file/d/1g-D6PtXV-UhoIYFcQbiqcXWU2Q2M-zo9/view?usp=sharing) |
| MSeg (3M)               | MSeg train      | Universal               |               |                               |                  |              |                 |              |         | [Google Drive](https://drive.google.com/file/d/1iobea9IW2cWPF6DtM04OvKtDRVNggSb5/view?usp=sharing) |


## Inference: Using our pre-trained models

Multi-scale inference greatly improves the smoothness of predictions, therefore our demos scripts use multi-scale config by default. While we train at 1080p, our predictions are often visually better when we feed in test images at 360p resolution.

If you have video input, and you would like to make predictions on each frame in the universal taxonomy, please run:
```
python mseg_semantic/tool/universal_demo.py -- 
```
If you have a set of images, and you would like to make a prediction in the universal taxonomy for each image, please run:
```
python mseg_semantic/tool/universal_demo.py -- 
```

If you have as input a single image, and you would like to make a prediction in the universal taxonomy, please run:
```
python mseg_semantic/tool/universal_demo.py -- 
```
If you would like to make predictions in a specific dataset's taxonomy, e.g. Cityscapes, for the RVC Challenge, please run:
```
python mseg_semantic/tool/universal_demo.py -- 
```

## Citing MSeg

If you find this code useful for your research, please cite:
```
@InProceedings{MSeg_2020_CVPR,
author = {Lambert, John and Zhuang, Liu and Sener, Ozan and Hays, James and Koltun, Vladlen},
title = {{MSeg}: A Composite Dataset for Multi-domain Semantic Segmentation},
booktitle = {Computer Vision and Pattern Recognition (CVPR)},
year = {2020}
}
```

Many thanks to Hengshuang Zhao for his [semseg](https://github.com/hszhao/semseg) repo, which we've based much of this repository off of.


## Other baseline models from our paper:

Individually-trained models that serve as baselines:

Nicknames: VOC = PASCAL VOC, WD = WildDash, SN = ScanNet

|    Model                | Training Set    |  Training <br> Taxonomy | VOC <br> mIoU | PASCAL <br> Context <br> mIoU | CamVid <br> mIoU | WD <br> mIoU | KITTI <br> mIoU | SN <br> mIoU | h. mean | Download <br> Link        |
| :---------------------: | :------------:  | :--------------------:  | :----------:  | :---------------------------: | :--------------: | :----------: | :-------------: | :----------: | :----:  | :--------------: |
| COCO (1M)               | COCO train      | Universal               | 73.7          |  43.1                         | 56.6             | 38.9         | 48.2            | 33.9         | 46.0    | [Google Drive]() |
| ADE20K (1M)             | ADE20K train    | Universal               | 34.6          |  24.0                         | 53.5             | 37.0         | 44.3            | 43.8         | 37.1    | [Google Drive]() |
| Mapillary (1M)          | Mapillary train | Universal               | 22.0          |  13.5                         | 82.5             | 55.2         | 68.5            | 2.1          | 9.2     | [Google Drive]() |
| IDD (1M)                | IDD train       | Universal               | 14.5          |  6.3                          | 70.5             | 40.6         | 50.7            | 1.6          | 6.5     | [Google Drive]() |
| BDD (1M)                | BDD train       | Universal               | 13.5          |  6.9                          | 71.0             | 52.1         | 55.0            | 1.4          | 6.1     | [Google Drive]() |
| Cityscapes (1M )        | Cityscapes train| Universal               | 12.1          |  6.5                          | 65.3             | 30.1         | 58.1            | 1.7          | 6.7     | [Google Drive]() |
| SUN RGB-D (1M)          | SUN RGBD train  | Universal               | 10.2          |  4.3                          | 0.1              | 1.4          | 0.7             | 42.2         | 0.3     | [Google Drive]() |
| Naive Mix Baseline (1M) | MSeg train.     | Naive                   |               |                               |                  |              |                 |              |         | [Google Drive]() | 
| Oracle (1M)             |                 |                         | 77.0          | 46.0                          | 79.1             | –            | 57.5            | 62.2         | –       | [Google Drive]() |


## Experiment Settings
For PSPNet, we generally follow the recommendations of [Zhao et al.](https://github.com/hszhao/semseg): We use a ResNet50 or ResNet101 backbone, with a crop size of 473x473 or 713x713, with synchronized BN. Our data augmentation consists of random scaling in the range [0.5,2.0], random rotation in the range [-10,10] degrees. We use SGD with momentum 0.9, weight decay of 1e-4. We use a polynomial learning rate with power 0.9. Base learning rate is set to 1e-2. An auxiliary cross-entropy (CE) loss is added to intermediate activations, a linear combination with weight 0.4. In our data, we use 255 as an ignore/unlabeled flag for the CE loss. Logits are upsampled by a factor is 8 ("zoom factor") to match original label map resolution for loss calculation.

We use Pytorch's Distributed Data Parallel (DDP) package for multiprocessing, with the NCCL backend. Zhao et al. recommend a training batch size of 16, with different number of epochs per dataset (ADE20k: 200, Cityscapes: 200, Camvid: 100, VOC2012: 50). For inference, we use a multi-scale accumulation of probabilities: [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]. Base size (ADE20K: 512, Camvid: 512, Cityscapes: 2048, VOC: 512) roughly equivalent to the average longer side of an image.

We use apex opt_level: 'O0'

For HRNet, we follow the [original authors' suggestions](https://arxiv.org/pdf/1904.04514.pdf): a learning rate of 0.01, momentum of 0.9, and weight decay of 5e-4. As above, we use a polynomial learning rate with power 0.9. Batch size is set to...

## Training Instructions

Download the HRNet Backbone Model [here](https://1drv.ms/u/s!Aus8VCZ_C_33dKvqI6pBZlifgJk) from the original authors' OneDrive.





