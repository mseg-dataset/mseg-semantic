
"""Config definitions for the HRNet architecture."""

from typing import List

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
