

from typing import List

import hydra
from hydra.utils import instantiate
from omegaconf import OmegaConf

from dataclasses import dataclass

@dataclass
class HRNetStageConfig:
  """
  `BLOCK' may be "BOTTLENECK" or "BASIC"

  """
  NUM_MODULES: int
  NUM_BRANCHES: int
  BLOCK: str
  NUM_BLOCKS: List[int]
  NUM_CHANNELS: List[int]
  FUSE_METHOD: str = "SUM"


@dataclass
class HRNetArchConfig:
  """ """
  STAGE1: HRNetStageConfig
  STAGE2: HRNetStageConfig
  STAGE3: HRNetStageConfig
  STAGE4: HRNetStageConfig
  FINAL_CONV_KERNEL: int











SceneOptimizer:
  _target_: gtsfm.scene_optimizer.SceneOptimizer
  save_gtsfm_data: True
  save_two_view_correspondences_viz: False
  save_3d_viz: True
  pose_angular_error_thresh: 5 # degrees

  feature_extractor:
    _target_: gtsfm.feature_extractor.FeatureExtractor
    detector_descriptor:
      _target_: gtsfm.frontend.cacher.detector_descriptor_cacher.DetectorDescriptorCacher
      detector_descriptor_obj:
        _target_: gtsfm.frontend.detector_descriptor.superpoint.SuperPointDetectorDescriptor




