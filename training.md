
## Training Models

We provide a number of config files for training models. The appropriate config will depend upon 3 factors:
1. Which resolution would you like to train at? (480p, 720p, or 1080p)
2. Which datasets would you like to train on? (all of relabeled MSeg, or unrelabeled MSeg, just one particular dataset, etc)
3. In which taxonomy (output space) would you like to train the model to make predictions?

## MSeg Models for Zero-Shot Transfer
@1080p Resolution
| Dataset \ Taxonomy |  Unified |   Naive  |
|:------------------:|  :-----: |:--------:| 
| MSeg Relabeled | | |
| MSeg Unrelabeled | config/train/1080_release/mseg-unrelabeled.yaml | config/train/1080_release/mseg-baseline.yaml |

@480p
| Dataset \ Taxonomy |  Unified |   Naive  |
|:------------------:|  :-----: |:--------:| 
| MSeg Relabeled | config/train/480_release/mseg-3m.yaml | |
| MSeg Unrelabeled |  |  |

@720p
| Dataset \ Taxonomy |  Unified |   Naive  |
|:------------------:|  :-----: |:--------:| 
| MSeg Relabeled | config/train/720_release/mseg-3m.yaml | |
| MSeg Unrelabeled |  |  |

## Models Trained on a Single Training Dataset

| Dataset            |   Taxonomy  |            Path to Config                       |
|:------------------:| :----------:| :---------------------------------------------: |
| ADE20K             |   Unified   | config/train/1080_release/single_universal.yaml |
| BDD                |   Unified   | config/train/1080_release/single_universal.yaml |
| COCO-Panoptic      |   Unified   | config/train/1080_release/single_universal.yaml |
| IDD                |   Unified   | config/train/1080_release/single_universal.yaml |
| Mapillary          |   Unified   | config/train/1080_release/single_universal.yaml |
| SUN RGB-D          |   Unified   | config/train/1080_release/single_universal.yaml |

vs. config/train/480/single_universal.yaml

## Oracle Models

## Training Baseline Models with Multi-Task Learning and CCSA

We also provide code to train models using multi-task learning (MGDA, specifically) and a domain generalization technique called CCSA. Please refer to [multiobjective_opt/README.md]() and [domain_generalization/README.md](), respectively.
