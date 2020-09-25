
## Training Models

We provide a number of config files for training models. The appropriate config will depend upon 3 factors:
1. Which resolution would you like to train at? (480p, 720p, or 1080p)
2. Which datasets would you like to train on? (all of relabeled MSeg, or unrelabeled MSeg, just one particular dataset, etc)
3. In which taxonomy (output space) would you like to train the model to make predictions?

## Models for Zero-Shot Transfer @1080p Resolution
| Dataset \ Taxonomy | Unified  | Naive  |
|:------------------:| | |
| MSeg Relabeled | | |
| MSeg Unrelabeled | | config/train/1080_release/mseg-baseline.yaml |

## Models Trained on a Single Training Dataset

| Dataset | Taxonomy | Path to Config |
|:------------------:| | |
| ADE20K | Unified | 1080_release/single_universal.yaml |
| BDD | Unified | 1080_release/single_universal.yaml |
| COCO-Panoptic | Unified | 1080_release/single_universal.yaml |
| IDD | Unified | 1080_release/single_universal.yaml |
| Mapillary | Unified | 1080_release/single_universal.yaml |
| SUN RGB-D | Unified | 1080_release/single_universal.yaml |

## Oracle Models

## Training Baseline Models with Multi-Task Learning and CCSA

We also provide code to train models using multi-task learning (MGDA, specifically) and a domain generalization technique called CCSA. Please refere to []() and [](), respectively.
