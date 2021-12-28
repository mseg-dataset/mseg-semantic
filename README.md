![Python package](https://github.com/mseg-dataset/mseg-semantic/workflows/Python%20package/badge.svg?branch=master)Try out our models in [Google Colab on your own images](https://colab.research.google.com/drive/1ctyBEf74uA-7R8sidi026OvNb4WlKkG1?usp=sharing)!

This repo includes the **semantic segmentation pre-trained models, training and inference code** for the paper:

**MSeg: A Composite Dataset for Multi-domain Semantic Segmentation** (CVPR 2020, Official Repo) [[CVPR PDF]](http://vladlen.info/papers/MSeg.pdf) [[Journal PDF]](https://arxiv.org/abs/2112.13762)
<br>
[John Lambert*](https://johnwlambert.github.io/),
[Zhuang Liu*](https://liuzhuang13.github.io/),
[Ozan Sener](http://ozansener.net/),
[James Hays](https://www.cc.gatech.edu/~hays/),
[Vladlen Koltun](http://vladlen.info/)
<br>
Presented at [CVPR 2020](http://cvpr2020.thecvf.com/). Link to [MSeg Video (3min) ](https://youtu.be/PzBK6K5gyyo)

**NEWS**:
- [Dec. 2021]: An updated journal-length version of our work is now available on ArXiv [here](https://arxiv.org/abs/2112.13762).

<p align="left">
  <img src="https://user-images.githubusercontent.com/62491525/83895683-094caa00-a721-11ea-8905-2183df60bc4f.gif" height="215">
  <img src="https://user-images.githubusercontent.com/62491525/83893966-aeb24e80-a71e-11ea-84cc-80e591f91ec0.gif" height="215">
</p>
<p align="left">
  <img src="https://user-images.githubusercontent.com/62491525/83895915-57fa4400-a721-11ea-8fa9-3c2ff0361080.gif" height="215">
  <img src="https://user-images.githubusercontent.com/62491525/83895972-73654f00-a721-11ea-8438-7bd43b695355.gif" height="215"> 
</p>

<p align="left">
  <img src="https://user-images.githubusercontent.com/62491525/83893958-abb75e00-a71e-11ea-978c-ab4080b4e718.gif" height="215">
  <img src="https://user-images.githubusercontent.com/62491525/83895490-c094f100-a720-11ea-9f85-cf4c6b030e73.gif" height="215">
</p>

<p align="left">
  <img src="https://user-images.githubusercontent.com/62491525/83895811-35682b00-a721-11ea-9641-38e3b2c1ad0e.gif" height="215">
  <img src="https://user-images.githubusercontent.com/62491525/83896026-8710b580-a721-11ea-86d2-a0fff9c6e26e.gif" height="215">
</p>

This repo is the second of 4 repos that introduce our work. It provides utilities to train semantic segmentation models, using a HRNet-W48 or PSPNet backbone, sufficient to train a winning entry on the [WildDash](https://wilddash.cc/benchmark/summary_tbl?hc=semantic_rob) benchmark.

- [` mseg-api`](https://github.com/mseg-dataset/mseg-api): utilities to download the MSeg dataset, prepare the data on disk in a unified taxonomy, on-the-fly mapping to a unified taxonomy during training.
- [`mseg-mturk`](https://github.com/mseg-dataset/mseg-mturk): utilities to perform large-scale Mechanical Turk re-labeling

One additional repo will be introduced in January 2021:
- `mseg-panoptic`: provides Panoptic-FPN and Mask-RCNN training, based on Detectron2


### How fast can your models run?
Our 480p MSeg model that accepts 473x473 px crops can run at **24.04 fps** on a Quadro P5000 GPU at single-scale inference.

| Model         | Crop Size | Frame Rate (fps) <br> Quadro P5000 | Frame Rate (fps) <br> V100 |
| :-----------: | :-------: | :---: | :---: | 
| MSeg-3m-480p  | 473 x 473 | 24.04 | 8.26  |
| MSeg-3m-720p  | 593 x 593 | 16.85 | 8.18  |
| MSeg-3m-1080p | 713 x 713 | 12.37 | 7.85  |

### Dependencies

Install the `mseg` module from [`mseg-api`](https://github.com/mseg-dataset/mseg-api).

### Install the MSeg-Semantic module:

* `mseg_semantic` can be installed as a python package using

        pip install -e /path_to_root_directory_of_the_repo/

Make sure that you can run `python -c "import mseg_semantic; print('hello world')"` in python, and you are good to go!


## MSeg Pre-trained Models

Each model is 528 MB in size. We provide download links and testing results (**single-scale** inference) below:

Abbreviated Dataset Names: VOC = PASCAL VOC, PC = PASCAL Context, WD = WildDash, SN = ScanNet

|    Model                | Training Set    |  Training <br> Taxonomy | VOC <br> mIoU | PC <br> mIoU | CamVid <br> mIoU | WD <br> mIoU | KITTI <br> mIoU | SN <br> mIoU | h. mean | Download <br> Link        |
| :---------------------: | :------------:  | :--------------------:  | :----------:  | :---------------------------: | :--------------: | :----------: | :-------------: | :----------: | :----:  | :--------------: |
| MSeg (1M)               | MSeg train      | Universal               | 70.7          | 42.7                          | 83.3             | 62.0         | 67.0            | 48.2         | 59.2    | [Google Drive](https://drive.google.com/file/d/1g-D6PtXV-UhoIYFcQbiqcXWU2Q2M-zo9/view?usp=sharing) |
| MSeg (3M)-480p               | MSeg <br> train      | Universal         | 76.4 |  45.9 |  81.2 |  62.7 |  68.2 |  49.5 |  61.2  | [Google Drive](https://drive.google.com/file/d/1BeZt6QXLwVQJhOVd_NTnVTmtAO1zJYZ-/view?usp=sharing) |
| MSeg (3M)-720p               | MSeg <br> train      | Universal               | 74.7 |  44.0 |  83.5 |  60.4 |  67.9 |  47.7 |  59.8 | [Google Drive](https://drive.google.com/file/d/1Y9rHOn_8e_qLuOnl4NeOeueU-MXRi3Ft/view?usp=sharing) |
| MSeg (3M)-1080p               | MSeg <br> train      | Universal               | 72.0 |  44.0 |  84.5 |  59.9 |  66.5 |  49.5 |  59.8 | [Google Drive](https://drive.google.com/file/d/1iobea9IW2cWPF6DtM04OvKtDRVNggSb5/view?usp=sharing) |

## Inference: Using our pre-trained models

We show how to perform inference here [in our Google Colab](https://colab.research.google.com/drive/1ctyBEf74uA-7R8sidi026OvNb4WlKkG1?usp=sharing).

Multi-scale inference greatly improves the smoothness of predictions, therefore our demos scripts use multi-scale config by default. While we train at 1080p, our predictions are often visually better when we feed in test images at 360p resolution.

If you have video input, and you would like to make predictions on each frame in the universal taxonomy, please set:
```
input_file=/path/to/my/video.mp4
```
If you have a set of images in a directory, and you would like to make a prediction in the universal taxonomy for each image, please set:
```
input_file=/path/to/my/directory
```

If you have as input a single image, and you would like to make a prediction in the universal taxonomy, please set:
```
input_file=/path/to/my/image
```

Now, run our demo script:
```
model_name=mseg-3m
model_path=/path/to/downloaded/model/from/google/drive
config=mseg_semantic/config/test/default_config_360_ms.yaml
python -u mseg_semantic/tool/universal_demo.py \
  --config=${config} model_name ${model_name} model_path ${model_path} input_file ${input_file}
```



## Testing a Model Trained in the Universal Taxonomy

To compute mIoU scores on all train and test datasets, run the following (single-scale inference):

```python
cd mseg_semantic/scripts
./eval_models.sh
```
This will launch several hundred SLURM jobs, each of the following form:
```
python mseg_semantic/tool/test_universal_tax.py --config=$config_path dataset $dataset_name model_path $model_path model_name $model_name
```
Please expect this to take many hours, depending upon your SLURM cluster capacity.

## Testing a Model Trained in the Oracle Taxonomy

"Oracle" models are trained in a test dataset's taxonomy, on its train split. To compute mIoU scores on the test dataset's val or test set, please run the following:

This will launch 5 SLURM jobs, each of the following form
```
python mseg_semantic/tool/test_oracle_tax.py
```

## Citing MSeg

If you find this code useful for your research, please cite:
```
@InProceedings{MSeg_2020_CVPR,
author = {Lambert, John and Liu, Zhuang and Sener, Ozan and Hays, James and Koltun, Vladlen},
title = {{MSeg}: A Composite Dataset for Multi-domain Semantic Segmentation},
booktitle = {Computer Vision and Pattern Recognition (CVPR)},
year = {2020}
}
```

Many thanks to Hengshuang Zhao for his [semseg](https://github.com/hszhao/semseg) repo, which we've based much of this repository off of.


## Other baseline models from our paper:

Below we report the performance of individually-trained models that serve as baselines. Inference is performed at **single-scale** below.

You can obtain the following table by running 
```python
python mseg_semantic/scripts/collect_results.py --regime zero_shot --scale ss --output_format markdown
python mseg_semantic/scripts/collect_results.py --regime oracle --scale ss --output_format markdown
```
after `./mseg_semantic/scripts/eval_models.sh`:

Abbreviated Dataset Names: VOC = PASCAL VOC, PC = PASCAL Context, WD = WildDash, SN = ScanNet

|    Model                | Training Set    |  Training <br> Tax- <br> onomy | VOC <br> mIoU | PC <br> mIoU | CamVid <br> mIoU | WD <br> mIoU | KITTI <br> mIoU | SN <br> mIoU | h. mean | Download <br> Link        |
| :---------------------: | :------------:  | :--------------------:  | :----------:  | :---------------------------: | :--------------: | :----------: | :-------------: | :----------: | :----:  | :--------------: |
| ADE20K (1M)             | ADE20K train    | Universal               | 35.4 |  23.9 |  52.6 |  38.6 |  41.6 |  42.9 |  36.9   | [Google Drive](https://drive.google.com/file/d/1xZ72nDuRc53u_WBWO_MazdHxR5HsO68I/view?usp=sharing) |
| BDD (1M)                | BDD train       | Universal               | 14.4 |   7.1 |  70.7 |  52.2 |  54.5 |   1.4 |   6.1     | [Google Drive](https://drive.google.com/file/d/1lTrejH7Agg1T61igCFWBpdUrrGs7EBf-/view?usp=sharing) |
| Cityscapes (1M )        | Cityscapes train| Universal               | 13.3 |   6.8 |  76.1 |  30.1 |  57.6 |   1.7 |   6.8     | [Google Drive](https://drive.google.com/file/d/18v6yH6mx4zksozn2ap5mCgKo-4ekayCO/view?usp=sharing) |
| COCO (1M)               | COCO train      | Universal               |  73.4 |  43.3 |  58.7 |  38.2 |  47.6 |  33.4 |  45.8    | [Google Drive](https://drive.google.com/file/d/18H6lfHQTPUyDge_TU1uqoedUFRUyGTuJ/view?usp=sharing) |
| IDD (1M)                | IDD train       | Universal               | 14.6 |   6.5 |  72.1 |  41.2 |  51.0 |   1.6 |   6.5     | [Google Drive](https://drive.google.com/file/d/1I6Fo5eUXrNBhWzC0BrApGAuKMGuxsrM7/view?usp=sharing) |
| Mapillary (1M)          | Mapillary <br> train | Universal               | 22.5 |  13.6 |  82.1 |  55.4 |  67.7 |   2.1 |   9.3     | [Google Drive](https://drive.google.com/file/d/1TPIFGZWuJXipDI9ceRMS-OdPbeVOi32u/view?usp=sharing) |
| SUN RGB-D (1M)          | SUN RGBD <br> train  | Universal               | 10.0 |   4.3 |   0.1 |   1.9 |   1.1 |  42.6 |   0.3     | [Google Drive](https://drive.google.com/file/d/1YRyJGe4gz4IHAKhuDUATqvaKRdR7gyAn/view?usp=sharing) |
| MSeg (1M)              | MSeg <br> train.     | Universal                  |  70.7 |  42.7 |  83.3 |  62.0 |  67.0 |  48.2 |  59.2   | [Google Drive](https://drive.google.com/file/d/1g-D6PtXV-UhoIYFcQbiqcXWU2Q2M-zo9/view?usp=sharing) | 
| MSeg Mix w/o relabeling (1M) | MSeg train.     |Universal           | 70.2 |  42.7 |  82.0 |  62.7 |  65.5 |  43.2 |  57.6 | [Google Drive](https://drive.google.com/file/d/1PHQNttQVn2dZW7ScUvaagc-OmB4eCRRy/view?usp=sharing) | 
| MGDA Baseline (1M)       | MSeg train.     | Universal               | 68.5 |  41.7 |  82.9 |  61.1 |  66.7 |  46.7 |  58.0 | [Google Drive](https://drive.google.com/file/d/1kqbEPr7LxjMZX_f2Mtieluj1OPu4eLE2/view?usp=sharing) | 
| MSeg (3M)-480p               | MSeg <br> train      | Universal         | 76.4 |  45.9 |  81.2 |  62.7 |  68.2 |  49.5 |  61.2  | [Google Drive](https://drive.google.com/file/d/1BeZt6QXLwVQJhOVd_NTnVTmtAO1zJYZ-/view?usp=sharing) |
| MSeg (3M)-720p               | MSeg <br> train      | Universal               | 74.7 |  44.0 |  83.5 |  60.4 |  67.9 |  47.7 |  59.8 | [Google Drive](https://drive.google.com/file/d/1Y9rHOn_8e_qLuOnl4NeOeueU-MXRi3Ft/view?usp=sharing) |
| MSeg (3M)-1080p               | MSeg <br> train      | Universal               | 72.0 |  44.0 |  84.5 |  59.9 |  66.5 |  49.5 |  59.8 | [Google Drive](https://drive.google.com/file/d/1iobea9IW2cWPF6DtM04OvKtDRVNggSb5/view?usp=sharing) |
| Naive Mix Baseline (1M) | MSeg <br> train.     | Naive                   |               |                               |                  |              |                 |              |         | [Google Drive](https://drive.google.com/file/d/1t4u0C3Li6_4mxLs032EuafKgmpONo3Ws/view?usp=sharing) | 
| Oracle (1M)             |                 |                         |  77.8 |  45.8 |  78.8 | -** | 58.4 |  62.3| - | |

**WildDash has no training set, so an "oracle" model cannot be trained.

**Oracle Model Download Links** 
- VOC  2012  (1M) Model [Google Drive](https://drive.google.com/file/d/1S5DuNCiRlaqTdXJ1TGups0kYEiEohZqC/view?usp=sharing)
- PASCAL-Context (1M) Model [Google Drive](https://drive.google.com/file/d/1-V4OOst1Ud9ohPWb-tSFNY2W44fMyZ_i/view?usp=sharing)
- Camvid (1M) Model [Google Drive](https://drive.google.com/file/d/1023eornZ2LP5NjDqIeunCIH35Ue38W8d/view?usp=sharing)
- KITTI (1M) Model [Google Drive](https://drive.google.com/file/d/14OkwxoaPK5mrxyL8CeUqGQOFW5U33b8J/view?usp=sharing)
- ScanNet-20 (1M) Model [Google Drive](https://drive.google.com/file/d/1njQkFTQ6F9p0nFTBLs2C4LGjAvm0Hydd/view?usp=sharing)

Note that the output number of classes for 7 of the models listed above will be identical (194 classes). These are the models that represent a single training dataset's performance -- *ADE20K (1M), BDD (1M), Cityscapes (1M ), COCO (1M), IDD (1M), Mapillary (1M), SUN RGB-D (1M)*. When we train a baseline model on a single dataset, we train it in the universal taxonomy (w/ 194 classes). If we did not do so, we would need to specify 7*6=42 mappings (which would be unbelievably tedious and also fairly redundant) since we measure each's performance according to zero-shot cross-dataset generalization -- 7 training datasets with their own taxonomy, and each would need its own mapping to each of the 6 test sets. 

By training each baseline *that is trained on a single training dataset* within the universal taxonomy, we are able to specify just 7+6=13 mappings in [this table](https://github.com/mseg-dataset/mseg-api/blob/master/mseg/class_remapping_files/MSeg_master.tsv) (each training dataset's taxonomy->universal taxonomy, and then universal taxonomy to each test dataset).

## Results on the Training Datasets

You can obtain the following table by running 
```python
python collect_results.py --regime training_datasets --scale ss --output_format markdown
```
after `./eval_models.sh`:
|         Model Name      | COCO  | ADE20k| Mapill|  IDD  |  BDD  |Citysca|SUN-RGBD|h. mean|
| :------------------------: | :--:  | :--:  | :--:  | :--:  | :--:  | :--:  | :--:  | :--:  |
|                    COCO-1m    |  52.7 |  19.1 |  28.4 |  31.1 |  44.9 |  46.9 |  29.6 |  32.4|
|                  ADE20K-1m    |  14.6 |  45.6 |  24.2 |  26.8 |  40.7 |  44.3 |  36.0 |  28.7|
|               Mapillary-1m    |   7.0 |   6.2 |  53.0 |  50.6 |  59.3 |  71.9 |   0.3 |   1.7|
|                     IDD-1m    |   3.2 |   3.0 |  24.6 |  64.9 |  42.4 |  48.0 |   0.4 |   2.3|
|                     BDD-1m    |   3.8 |   4.2 |  23.2 |  32.3 |  63.4 |  58.1 |   0.3 |   1.6|
|              Cityscapes-1m    |   3.4 |   3.1 |  22.1 |  30.1 |  44.1 |  77.5 |   0.2 |   1.2|
|                SUN RGBD-1m    |   3.4 |   7.0 |   1.1 |   1.0 |   2.2 |   2.6 |  43.0 |   2.1|
|                 MSeg-1m    |  50.7 |  45.7 |  53.1 |  65.3 |  68.5 |  80.4 |  50.3 |  57.1|
|  MSeg-1m-w/o relabeling    |  50.4 |  45.4 |  53.1 |  65.1 |  66.5 |  79.5 |  49.9 |  56.6|
|            MSeg-MGDA-1m    |  48.1 |  43.7 |  51.6 |  64.1 |  67.2 |  78.2 |  49.9 |  55.4|
|            MSeg-3m-480p    |  56.1 |  49.6 |  53.5 |  64.5 |  67.8 |  79.9 |  49.2 |  58.5|
|            MSeg-3m-720p    |  53.3 |  48.2 |  53.5 |  64.8 |  68.6 |  79.8 |  49.3 |  57.8|
|           MSeg-3m-1080p    |  53.6 |  49.2 |  54.9 |  66.3 |  69.1 |  81.5 |  50.1 |  58.8|

## Experiment Settings
We use the [HRNetV2-W48](https://arxiv.org/pdf/1904.04514.pdf) architecture. All images are resized to 1080p (shorter size=1080) at training time before a crop is taken.

 We run inference with the shorter side of each test image at three resolutions (360p, 720p, 1080p), and take the max among these 3 possible resolutions. Note that in the original [semseg](https://github.com/hszhao/semseg) repo, the author specifies the longer side of an image, whereas we specify the shorter side. Batch size is set to 35.

We generally follow the recommendations of [Zhao et al.](https://github.com/hszhao/semseg): Our data augmentation consists of random scaling in the range [0.5,2.0], random rotation in the range [-10,10] degrees. We use SGD with momentum 0.9, weight decay of 1e-4. We use a polynomial learning rate with power 0.9. Base learning rate is set to 1e-2. An auxiliary cross-entropy (CE) loss is added to intermediate activations, a linear combination with weight 0.4. In our data, we use 255 as an ignore/unlabeled flag for the CE loss. We use Pytorch's Distributed Data Parallel (DDP) package for multiprocessing, with the NCCL backend. We use apex opt_level: 'O0' and use a crop size of 713x713, with synchronized BN.

## Training Instructions

Please refer to [training.md](https://github.com/mseg-dataset/mseg-semantic/blob/master/training.md) for detailed instructions on how to train each of our models. As a frame of reference as to the amount of compute required, we use 8 Quadro RTX 6000 cards, each w/ 24 GB of RAM, for training. The 3 million crop models took ~2-3 weeks to train on such hardware, and the 1 million crop models took ~4-7 days.


## Running unit tests and integration tests

To run the unit tests, execute
```python
pytest tests
````
All should pass. To run the integration tests, follow the instructions in the following 3 files, then run:
```python
python test_test_oracle_tax.py
python test_test_universal_tax.py
python test_universal_demo.py
```
All should also pass.

## Frequently Asked Questions (FAQ) (identical to FAQ on [`mseg-api` page](https://github.com/mseg-dataset/mseg-api))
**Q**: Do the weights include the model structure or it's just the weights? If the latter, which model do these weights refer to? Under the `models` directory, there are several model implementations.

**A**: The pre-trained models follow the HRNet-W48 architecture. The model structure is defined in the code [here](https://github.com/mseg-dataset/mseg-semantic/blob/master/mseg_semantic/model/seg_hrnet.py#L274). The saved weights provide a dictionary between keys (unique IDs for each weight identifying the corresponding layer/layer type) and values (the floating point weights).

------------------------------------------------------

**Q**: How is testing performed on the test datasets? In the paper you talk about "zero-shot transfer" -- how this is performed? Are the test dataset labels also mapped or included in the unified taxonomy? If you remapped the test dataset labels to the unified taxonomy, are the reported results the performances on the unified label space, or on each test dataset's original label space? How did you you obtain results on the WildDash dataset - which is evaluated by the server - when the MSeg taxonomy may be different from the WildDash dataset.

**A**: Regarding "zero-shot transfer", please refer to section "Using the MSeg taxonomy on a held-out dataset" on page 6 of [our paper](http://vladlen.info/papers/MSeg.pdf). This section describes how we hand-specify mappings from the unified taxonomy to each test dataset's taxonomy as a linear mapping (implemented [here](https://github.com/mseg-dataset/mseg-api/blob/master/mseg/taxonomy/taxonomy_converter.py#L220) in mseg-api). All results are in the test dataset's original label space (i.e. if WildDash expects class indices in the range [0,18] per our [names_list](https://github.com/mseg-dataset/mseg-api/blob/master/mseg/dataset_lists/wilddash-19/wilddash-19_names.txt), our testing script uses the `TaxonomyConverter` [`transform_predictions_test()`](https://github.com/mseg-dataset/mseg-api/blob/master/mseg/taxonomy/taxonomy_converter.py#L267) functionality  to produce indices in that range, remapping probabilities.

------------------------------------------------------

**Q**: Why don't indices in `MSeg_master.tsv` match the training indices in individual datasets? For example, for the *road* class: In [idd-39](https://github.com/mseg-dataset/mseg-api/blob/master/mseg/dataset_lists/idd-39/idd-39_names.txt#L1), *road* has index 0, but in [idd-39-relabeled](https://github.com/mseg-dataset/mseg-api/blob/master/mseg/dataset_lists/idd-39-relabeled/idd-39-relabeled_names.txt#L20), *road* has index 19. It is index 7 in [cityscapes-34](https://github.com/mseg-dataset/mseg-api/blob/master/mseg/dataset_lists/cityscapes-34/cityscapes-34_names.txt#L8). The [cityscapes-19-relabeled index](https://github.com/mseg-dataset/mseg-api/blob/master/mseg/dataset_lists/cityscapes-19-relabeled/cityscapes-19-relabeled_names.txt) *road* is 11. As far as I can tell, ultimately the 'MSeg_Master.tsv' file provides the final mapping to the MSeg label space. But here, the *road* class seems to have an index of 98, which is neither 19 nor 11.

**A**: Indeed, [unified taxonomy class index 98](https://github.com/mseg-dataset/mseg-api/blob/master/mseg/class_remapping_files/MSeg_master.tsv#L100) represents "road". But we use the TaxonomyConverter to accomplish the mapping on the fly from *idd-39-relabeled* to the unified/universal taxonomy (we use the terms "unified" and "universal" interchangeably). This is done by adding a transform in the training loop that calls [`TaxonomyConverter.transform_label()`](https://github.com/mseg-dataset/mseg-api/blob/master/mseg/taxonomy/taxonomy_converter.py#L250) on the fly. You can see how that transform is implemented [here](https://github.com/mseg-dataset/mseg-semantic/blob/add-dataset-eval/mseg_semantic/utils/transform.py#L52.) in `mseg-semantic`.

------------------------------------------------------

**Q**: When testing, but there are test classes that are not in the unified taxonomy (e.g. Parking, railtrack, bridge etc. in WildDash), how do you produce predictions for that class? I understand you map the predictions with a binary matrix. But what do you do when there's no one-to-one correspondence?

**A**: WildDash v1 uses the 19-class taxonomy for evaluation, just like Cityscapes. So we use [the following script](https://github.com/mseg-dataset/mseg-api/blob/master/download_scripts/mseg_remap_wilddash.sh) to remap the 34-class taxonomy to 19-class taxonomy for WildDash  for testing inference and submission. You can see how Cityscapes evaluates just 19 of the 34 classes here in the [evaluation script](https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/evaluation/evalPixelLevelSemanticLabeling.py#L301) and in [the taxonomy definition](https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py#L73). However, [bridge](https://github.com/mseg-dataset/mseg-api/blob/master/mseg/class_remapping_files/MSeg_master.tsv#L34) and [rail track](https://github.com/mseg-dataset/mseg-api/blob/master/mseg/class_remapping_files/MSeg_master.tsv#L99) are actually included in our unified taxonomy, as you’ll see in MSeg_master.tsv.

------------------------------------------------------

**Q**: How are datasets images read in for training/inference? Should I use the `dataset_apis` from `mseg-api`?

**A**: The `dataset_apis` from `mseg-api` are not for training or inference. They are purely for generating the MSeg dataset labels on disk. We read in the datasets using [`mseg_semantic/utils/dataset.py`](https://github.com/mseg-dataset/mseg-semantic/blob/master/mseg_semantic/utils/dataset.py) and then remap them to the universal space on the fly.

------------------------------------------------------

**Q**: In the training configuration file, each dataset uses one GPU each for multi-dataset training. I don't have enough hardware resources (I only have four GPUs at most)，Can I still train？

**A**: Sure, you can still train by setting to the batch size to a smaller number, but the training will take longer. Another alternative is to train at a lower input resolution (smaller input crops, see the 480p or 720p configs instead of 1080p config), or to train for fewer iterations.

------------------------------------------------------

**Q**: The purpose of using MGDA is unclear -- is it recommended for training?

**A**: Please refer to the section "Algorithms for learning from multiple domains" from our [paper](http://vladlen.info/papers/MSeg.pdf). In our ablation experiments, we found that training with MGDA does not lead to the best model, so we set it to false when training our best models.

------------------------------------------------------

**Q**: Does save_path refer to the path saved by the weights after training?

**A**: `save_path` is the directory where the model checkpoints and results will be saved. See [here](https://github.com/mseg-dataset/mseg-semantic/blob/5fd9ed3d22336005ee9f687d50188019873e67d5/mseg_semantic/tool/train.py#L587).

------------------------------------------------------

**Q**: Does the `auto_resume` param refer to the weight of breakpoint training, or the mseg-3m.pth provided by the author?

**A**: We use the `auto_resume` config parameter to allow one to continue training if training is interrupted due to a scheduler compute time limit or hardware error. You could also use it to fine-tune a model.

------------------------------------------------------

**Q**: Could I know how to map the predicted label iD to the ID on cityscapes? Do you have any code/dictionary to achieve this?

**A**: There are two Cityscape taxonomies (cityscapes-19 and cityscapes-34), although cityscapes-19 is more commonly used for evaluation. The classes in these taxonomies are enumerated in mseg-api [here](https://github.com/mseg-dataset/mseg-api/blob/master/mseg/dataset_lists/cityscapes-19/cityscapes-19_names.txt) and [here](https://github.com/mseg-dataset/mseg-api/blob/master/mseg/dataset_lists/cityscapes-34/cityscapes-34_names.txt)

We have released both unified models (trained on many datasets, list available [here](https://github.com/mseg-dataset/mseg-semantic#mseg-pre-trained-models])) and models trained on single datasets, listed [here](https://github.com/mseg-dataset/mseg-semantic#other-baseline-models-from-our-paper).

If you use a unified model for testing, our code maps class scores from the unified taxonomy to cityscapes classes. We discuss this in a section of our [paper](http://vladlen.info/papers/MSeg.pdf) (page 6, top-right under **Using the MSeg taxonomy on a held-out dataset**). The mapping is available in [MSeg_master.tsv](https://github.com/mseg-dataset/mseg-api/blob/master/mseg/class_remapping_files/MSeg_master.tsv), if you compare the `universal` and `wilddash-19` columns (wilddash-19 shares the same classes with cityscapes-19)

If instead you used a model specifically trained on cityscapes, e.g. `cityscapes-19-1m`, which we call an "oracle model" since it is trained and tested on different splits of the same dataset, then the output classes are already immediately in the desired taxonomy.

Our inference code that dumps model results in any particular taxonomy is available here:
https://github.com/mseg-dataset/mseg-semantic/blob/master/mseg_semantic/scripts/eval_models.sh







