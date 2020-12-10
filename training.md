
## Training Models

The script [`mseg_semantic/tool/train.py`](https://github.com/mseg-dataset/mseg-semantic/blob/training/mseg_semantic/tool/train.py) is the training script we use for training the majority of our models (all except the CCSA models). It merges multiple datasets at training time using our `TaxonomyConverter` class. Before training, you will need to download all the datasets as described [here](https://github.com/mseg-dataset/mseg-api/blob/master/download_scripts/README.md), and also ensure that the unit tests pass successfully at the end. You will also need to download the ImageNet-pretrained HRNet backbone model [here](https://1drv.ms/u/s!Aus8VCZ_C_33dKvqI6pBZlifgJk) from the original authors' OneDrive.

We provide a number of config files for training models. The appropriate config will depend upon 3 factors:
1. Which resolution would you like to train at? (480p, 720p, or 1080p)
2. Which datasets would you like to train on? (all of relabeled MSeg, or unrelabeled MSeg, just one particular dataset, etc)
3. In which taxonomy (output space) would you like to train the model to make predictions?

## Configs for MSeg Models for Zero-Shot Transfer
@1080p Resolution
| Dataset \ Taxonomy |  Unified |   Naive  |
|:------------------:|  :-----: |:--------:| 
| MSeg Relabeled | config/train/1080_release/mseg-relabeled-1m.yaml | --- |
| MSeg Unrelabeled | config/train/1080_release/mseg-unrelabeled.yaml | config/train/1080_release/mseg-naive-baseline.yaml |

If you want to train the Relabeled + Unified Tax. model for 3M crops instead of 1M, use `mseg_semantic/config/train/1080_release/mseg-relabeled-3m.yaml`.

@480p
| Dataset \ Taxonomy |  Unified |   Naive  |
|:------------------:|  :-----: |:--------:| 
| MSeg Relabeled | config/train/480_release/mseg-3m.yaml | --- |
| MSeg Unrelabeled | --- | --- |

Note that at 480p, we only re-train w/ our best configuration (a model trained on the Relabeled MSeg Dataset in the Unified taxonomy). The rest of our ablation experiments are carried out at 1080p.

@720p
| Dataset \ Taxonomy |  Unified |   Naive  |
|:------------------:|  :-----: |:--------:| 
| MSeg Relabeled | config/train/720_release/mseg-3m.yaml | --- |
| MSeg Unrelabeled | --- | --- |

Note that at 720p, we only re-train w/ our best configuration (a model trained on the Relabeled MSeg Dataset in the Unified taxonomy). The rest of our ablation experiments are carried out at 1080p.

## Configs for Models Trained on a Single Training Dataset

| Dataset            |   Taxonomy  |            Path to Config                       |
|:------------------:| :----------:| :---------------------------------------------: |
| ADE20K             |   Unified   | config/train/1080_release/single_universal.yaml |
| BDD                |   Unified   | config/train/1080_release/single_universal.yaml |
| COCO-Panoptic      |   Unified   | config/train/1080_release/single_universal.yaml |
| IDD                |   Unified   | config/train/1080_release/single_universal.yaml |
| Mapillary          |   Unified   | config/train/1080_release/single_universal.yaml |
| SUN RGB-D          |   Unified   | config/train/1080_release/single_universal.yaml |

vs. config/train/480/single_universal.yaml

## Configs for Oracle Models

## Training Baseline Models with Multi-Task Learning and CCSA

We also provide code to train models using multi-task learning (MGDA, specifically) and a domain generalization technique called CCSA. Please refer to [multiobjective_opt/README.md]() and [domain_generalization/README.md](), respectively.

## Apex Library
We use the 'O0' optimization level from Apex. NVIDIA Apex has 4 optimization levels, and we use the first:
- O0 (FP32 training): basically a no-op. Everything is FP32 just as before.
- O1 (Conservative Mixed Precision): only some whitelist ops are done in FP16.
- O2 (Fast Mixed Precision): this is the standard mixed precision training. 
        It maintains FP32 master weights and optimizer.step acts directly on the FP32 master weights.
- O3 (FP16 training): full FP16. Passing keep_batchnorm_fp32=True can speed 
        things up as cudnn batchnorm is faster anyway.
