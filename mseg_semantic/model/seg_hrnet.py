#!/usr/bin/python3

"""
"High-Resolution Representations for Labeling Pixels and Regions"
https://arxiv.org/pdf/1904.04514.pdf

Code adopted from https://github.com/HRNet/HRNet-Semantic-Segmentation
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
import functools
from pathlib import Path
from typing import List, Optional, Union

import hydra
import numpy as np
import torch
import torch.nn as nn
import torch._utils
import torch.nn.functional as F
from hydra.utils import instantiate
from omegaconf import OmegaConf

from mseg_semantic.model.seg_hrnet_config import HRNetArchConfig, HRNetStageConfig


BatchNorm2d = torch.nn.SyncBatchNorm
BN_MOMENTUM = 0.01
logger = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parent


def conv3x3(in_planes: int, out_planes: int, stride: int = 1) -> nn.Module:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    """
    Identical to ResNet BasicBlock, but we hardcode the
    Batchnorm variant and momentum.
    """

    expansion = 1

    def __init__(self, inplanes: int, planes: int, stride: int = 1, downsample=None) -> None:
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    """
    Identical to ResNet Bottleneck, but we hardcode the
    Batchnorm variant and momentum.
    """

    expansion = 4

    def __init__(self, inplanes: int, planes: int, stride: int = 1, downsample=None) -> None:
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = BatchNorm2d(planes * self.expansion, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class HighResolutionModule(nn.Module):
    def __init__(
        self,
        num_branches: int,
        block: Union[BasicBlock, Bottleneck],
        num_blocks: List[int],
        num_inchannels: List[int],
        num_channels: List[int],
        fuse_method: str,
        multi_scale_output: bool = True,
    ) -> None:
        """ """
        super(HighResolutionModule, self).__init__()
        self._check_branches(
            num_branches=num_branches, num_blocks=num_blocks, num_inchannels=num_inchannels, num_channels=num_channels
        )

        self.num_inchannels = num_inchannels
        self.fuse_method = fuse_method
        self.num_branches = num_branches

        self.multi_scale_output = multi_scale_output

        self.branches = self._make_branches(
            num_branches=num_branches, block=block, num_blocks=num_blocks, num_channels=num_channels
        )
        self.fuse_layers = self._make_fuse_layers()
        self.relu = nn.ReLU(inplace=True)

    def _check_branches(
        self, num_branches: int, num_blocks: List[int], num_inchannels: List[int], num_channels: List[int]
    ) -> None:
        """ """
        if num_branches != len(num_blocks):
            error_msg = "NUM_BRANCHES({}) <> NUM_BLOCKS({})".format(num_branches, len(num_blocks))
            logger.error(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_channels):
            error_msg = "NUM_BRANCHES({}) <> NUM_CHANNELS({})".format(num_branches, len(num_channels))
            logger.error(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_inchannels):
            error_msg = "NUM_BRANCHES({}) <> NUM_INCHANNELS({})".format(num_branches, len(num_inchannels))
            logger.error(error_msg)
            raise ValueError(error_msg)

    def _make_one_branch(
        self,
        branch_index: int,
        block: Union[BasicBlock, Bottleneck],
        num_blocks: List[int],
        num_channels: List[int],
        stride: int = 1,
    ) -> nn.Module:
        """ """
        downsample = None
        if stride != 1 or self.num_inchannels[branch_index] != num_channels[branch_index] * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.num_inchannels[branch_index],
                    num_channels[branch_index] * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                BatchNorm2d(num_channels[branch_index] * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(self.num_inchannels[branch_index], num_channels[branch_index], stride, downsample))
        self.num_inchannels[branch_index] = num_channels[branch_index] * block.expansion
        for i in range(1, num_blocks[branch_index]):
            layers.append(block(self.num_inchannels[branch_index], num_channels[branch_index]))

        return nn.Sequential(*layers)

    def _make_branches(
        self, num_branches: int, block: Union[BasicBlock, Bottleneck], num_blocks: List[int], num_channels: List[int]
    ) -> nn.ModuleList:
        branches = []

        for i in range(num_branches):
            branches.append(self._make_one_branch(i, block, num_blocks, num_channels))

        return nn.ModuleList(branches)

    def _make_fuse_layers(self) -> nn.ModuleList:
        """ """
        if self.num_branches == 1:
            return None

        num_branches = self.num_branches
        num_inchannels = self.num_inchannels
        fuse_layers = []
        for i in range(num_branches if self.multi_scale_output else 1):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:
                    fuse_layer.append(
                        nn.Sequential(
                            nn.Conv2d(num_inchannels[j], num_inchannels[i], 1, 1, 0, bias=False),
                            BatchNorm2d(num_inchannels[i], momentum=BN_MOMENTUM),
                        )
                    )
                elif j == i:
                    fuse_layer.append(None)
                else:
                    conv3x3s = []
                    for k in range(i - j):
                        if k == i - j - 1:
                            num_outchannels_conv3x3 = num_inchannels[i]
                            conv3x3s.append(
                                nn.Sequential(
                                    nn.Conv2d(num_inchannels[j], num_outchannels_conv3x3, 3, 2, 1, bias=False),
                                    BatchNorm2d(num_outchannels_conv3x3, momentum=BN_MOMENTUM),
                                )
                            )
                        else:
                            num_outchannels_conv3x3 = num_inchannels[j]
                            conv3x3s.append(
                                nn.Sequential(
                                    nn.Conv2d(num_inchannels[j], num_outchannels_conv3x3, 3, 2, 1, bias=False),
                                    BatchNorm2d(num_outchannels_conv3x3, momentum=BN_MOMENTUM),
                                    nn.ReLU(inplace=True),
                                )
                            )
                    fuse_layer.append(nn.Sequential(*conv3x3s))
            fuse_layers.append(nn.ModuleList(fuse_layer))

        return nn.ModuleList(fuse_layers)

    def get_num_inchannels(self) -> List[int]:
        return self.num_inchannels

    def forward(self, x: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Args:
            x: list of Pytorch tensors.

        Returns:
            x_fuse: list of Pytorch tensors.
        """
        if self.num_branches == 1:
            return [self.branches[0](x[0])]

        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])

        x_fuse = []
        for i in range(len(self.fuse_layers)):
            y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])
            for j in range(1, self.num_branches):
                if i == j:
                    y = y + x[j]
                elif j > i:
                    width_output = x[i].shape[-1]
                    height_output = x[i].shape[-2]
                    y = y + F.interpolate(
                        self.fuse_layers[i][j](x[j]), size=[height_output, width_output], mode="bilinear"
                    )
                else:
                    y = y + self.fuse_layers[i][j](x[j])
            x_fuse.append(self.relu(y))

        return x_fuse


blocks_dict = {"BASIC": BasicBlock, "BOTTLENECK": Bottleneck}


class HighResolutionNet(nn.Module):
    def __init__(self, config: HRNetArchConfig, criterion: nn.Module, n_classes: int) -> None:
        """ """
        super(HighResolutionNet, self).__init__()

        self.criterion = criterion
        self.n_classes = n_classes

        # stem net
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)

        self.stage1_cfg = config.STAGE1
        num_channels = self.stage1_cfg.NUM_CHANNELS[0]
        block = blocks_dict[self.stage1_cfg.BLOCK]
        num_blocks = self.stage1_cfg.NUM_BLOCKS[0]
        self.layer1 = self._make_layer(block, 64, num_channels, num_blocks)
        stage1_out_channel = block.expansion * num_channels

        self.stage2_cfg = config.STAGE2
        num_channels = self.stage2_cfg.NUM_CHANNELS
        block = blocks_dict[self.stage2_cfg.BLOCK]
        num_channels = [num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition1 = self._make_transition_layer([stage1_out_channel], num_channels)
        self.stage2, pre_stage_channels = self._make_stage(self.stage2_cfg, num_channels)

        self.stage3_cfg = config.STAGE3
        num_channels = self.stage3_cfg.NUM_CHANNELS
        block = blocks_dict[self.stage3_cfg.BLOCK]
        num_channels = [num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition2 = self._make_transition_layer(pre_stage_channels, num_channels)
        self.stage3, pre_stage_channels = self._make_stage(self.stage3_cfg, num_channels)

        self.stage4_cfg = config.STAGE4
        num_channels = self.stage4_cfg.NUM_CHANNELS
        block = blocks_dict[self.stage4_cfg.BLOCK]
        num_channels = [num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition3 = self._make_transition_layer(pre_stage_channels, num_channels)
        self.stage4, pre_stage_channels = self._make_stage(self.stage4_cfg, num_channels, multi_scale_output=True)

        last_inp_channels = np.int(np.sum(pre_stage_channels))

        self.last_layer = nn.Sequential(
            nn.Conv2d(
                in_channels=last_inp_channels, out_channels=last_inp_channels, kernel_size=1, stride=1, padding=0
            ),
            BatchNorm2d(last_inp_channels, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=last_inp_channels,
                out_channels=self.n_classes,
                kernel_size=config.FINAL_CONV_KERNEL,
                stride=1,
                padding=1 if config.FINAL_CONV_KERNEL == 3 else 0,
            ),
        )

    def _make_transition_layer(
        self, num_channels_pre_layer: List[int], num_channels_cur_layer: List[int]
    ) -> nn.ModuleList:
        """
        Use 3x3 convolutions, with stride 2 and padding 1.
        """
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)

        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(
                        nn.Sequential(
                            nn.Conv2d(num_channels_pre_layer[i], num_channels_cur_layer[i], 3, 1, 1, bias=False),
                            BatchNorm2d(num_channels_cur_layer[i], momentum=BN_MOMENTUM),
                            nn.ReLU(inplace=True),
                        )
                    )
                else:
                    transition_layers.append(None)
            else:
                conv3x3s = []
                for j in range(i + 1 - num_branches_pre):
                    inchannels = num_channels_pre_layer[-1]
                    outchannels = num_channels_cur_layer[i] if j == i - num_branches_pre else inchannels
                    conv3x3s.append(
                        nn.Sequential(
                            nn.Conv2d(inchannels, outchannels, 3, 2, 1, bias=False),
                            BatchNorm2d(outchannels, momentum=BN_MOMENTUM),
                            nn.ReLU(inplace=True),
                        )
                    )
                transition_layers.append(nn.Sequential(*conv3x3s))

        return nn.ModuleList(transition_layers)

    def _make_layer(
        self, block: Union[BasicBlock, Bottleneck], inplanes: int, planes: int, blocks: int, stride: int = 1
    ) -> nn.Module:
        """
        Identical to ResNet `_make_layer()`, except `inplanes` is an
        explicit argument rather than class attribute, and batch norm
        implementation and momentum are hardcoded.
        """
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(inplanes, planes))

        return nn.Sequential(*layers)

    def _make_stage(
        self, layer_config: HRNetStageConfig, num_inchannels: List[int], multi_scale_output: bool = True
    ) -> nn.Module:
        """ """
        num_modules = layer_config.NUM_MODULES

        modules = []
        for i in range(num_modules):
            # multi_scale_output is only used last module
            if not multi_scale_output and i == num_modules - 1:
                reset_multi_scale_output = False
            else:
                reset_multi_scale_output = True
            modules.append(
                HighResolutionModule(
                    num_branches=layer_config.NUM_BRANCHES,
                    block=blocks_dict[layer_config.BLOCK],
                    num_blocks=layer_config.NUM_BLOCKS,
                    num_inchannels=num_inchannels,
                    num_channels=layer_config.NUM_CHANNELS,
                    fuse_method=layer_config.FUSE_METHOD,
                    multi_scale_output=reset_multi_scale_output,
                )
            )
            num_inchannels = modules[-1].get_num_inchannels()

        return nn.Sequential(*modules), num_inchannels

    def forward(self, x: torch.Tensor, y: Optional[torch.Tensor] = None):
        """
        Network starts from a stem of two strided 3 × 3 convolutions
        decreasing the resolution to 1/4.

        Rescale the low-resolution representations through bilinear
        upsampling to the high resolution, and concatenate the subsets
        of such representations.

        At end, the segmentation maps are upsampled (4 times) to the
        input size by bilinear upsampling for both training and testing.
        """
        x_size = x.size()
        assert (x_size[2] - 1) % 8 == 0 and (x_size[3] - 1) % 8 == 0
        # h = int((x_size[2] - 1) / 8 * self.zoom_factor + 1)
        h = x_size[2]
        w = x_size[3]
        # w = int((x_size[3] - 1) / 8 * self.zoom_factor + 1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.layer1(x)

        x_list = []
        for i in range(self.stage2_cfg.NUM_BRANCHES):
            if self.transition1[i] is not None:
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)
        y_list = self.stage2(x_list)

        x_list = []
        for i in range(self.stage3_cfg.NUM_BRANCHES):
            if self.transition2[i] is not None:
                x_list.append(self.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage3(x_list)

        x_list = []
        for i in range(self.stage4_cfg.NUM_BRANCHES):
            if self.transition3[i] is not None:
                x_list.append(self.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        x = self.stage4(x_list)

        # Upsampling
        x0_h, x0_w = x[0].size(2), x[0].size(3)
        x1 = F.upsample(x[1], size=(x0_h, x0_w), mode="bilinear")
        x2 = F.upsample(x[2], size=(x0_h, x0_w), mode="bilinear")
        x3 = F.upsample(x[3], size=(x0_h, x0_w), mode="bilinear")

        x = torch.cat([x[0], x1, x2, x3], 1)
        # Perform two 1x1 convolutions on concat representation
        x = self.last_layer(x)

        # Bilinear upsampling of output
        x = F.interpolate(x, size=(h, w), mode="bilinear", align_corners=True)
        if self.training:
            main_loss = self.criterion(x, y)
            return x.max(1)[1], main_loss, main_loss * 0
        else:
            return x

        # return x

    def init_weights(self, load_imagenet_model: bool = False, imagenet_ckpt_fpath: str = "") -> None:
        """For training, we use a model pretrained on ImageNet. Irrelevant at inference.
        Args:
            load_imagenet_model:
            imagenet_ckpt_fpath: str representing path to pretrained model.
        """
        logger.info("=> init weights from normal distribution")
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if not load_imagenet_model:
            return
        if os.path.isfile(imagenet_ckpt_fpath):
            pretrained_dict = torch.load(imagenet_ckpt_fpath)
            logger.info("=> loading pretrained model {}".format(imagenet_ckpt_fpath))
            model_dict = self.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys()}
            # for k, _ in pretrained_dict.items():
            #    logger.info(
            #        '=> loading {} pretrained model {}'.format(k, pretrained))
            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict)
        else:
            # logger.info(pretrained)
            logger.info("cannot find ImageNet model path, use random initialization")
            raise Exception("no pretrained model found at {}".format(imagenet_ckpt_fpath))


def get_seg_model(
    cfg: HRNetArchConfig,
    criterion: nn.Module,
    n_classes: int,
    load_imagenet_model: bool = False,
    imagenet_ckpt_fpath: str = "",
    **kwargs
) -> nn.Module:
    model = HighResolutionNet(cfg, criterion, n_classes, **kwargs)
    model.init_weights(load_imagenet_model, imagenet_ckpt_fpath)
    assert isinstance(model, nn.Module)
    return model


def get_configured_hrnet(
    n_classes: int,
    load_imagenet_model: bool = False,
    imagenet_ckpt_fpath: str = "",
) -> nn.Module:
    """
    Args:
        n_classes: integer representing number of output classes.
        load_imagenet_model: whether to initialize from ImageNet-pretrained model.
        imagenet_ckpt_fpath: string representing path to file with weights to
            initialize model with.

    Returns:
        model: HRNet model w/ architecture configured according to model yaml,
            and with specified number of classes and weights initialized
            (at training, init using imagenet-pretrained model).
    """

    with hydra.initialize_config_module(config_module="mseg_semantic.model"):
        # config is relative to the gtsfm module
        cfg = hydra.compose(config_name="seg_hrnet.yaml")
        logger.info("Using config: ")
        logger.info(OmegaConf.to_yaml(cfg))
        config: HRNetArchConfig = instantiate(cfg.HRNetArchConfig)

    criterion = nn.CrossEntropyLoss(ignore_index=255)
    model = get_seg_model(config, criterion, n_classes, load_imagenet_model, imagenet_ckpt_fpath)
    return model


if __name__ == "__main__":
    """ """
    import pdb

    pdb.set_trace()
    imagenet_ckpt_fpath = ""
    load_imagenet_model = False
    model = get_configured_hrnet(180, load_imagenet_model, imagenet_ckpt_fpath)
    num_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(num_p)
