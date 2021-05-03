#!/usr/bin/python3

import math
import numpy as np
import os
import pdb
import random
import sys
import time
import torch
from torch import nn
import torch.nn.functional as F

from mseg_semantic.model.pspnet import PPM
import mseg_semantic.model.resnet as models

from mseg_semantic.domain_generalization.ccsa_utils import (
    paired_euclidean_distance,
    contrastive_loss,
    sample_pair_indices,
    get_merged_pair_embeddings
)
from mseg_semantic.utils.json_utils import save_json_dict

"""
Reimplementation of "Unified Deep Supervised Domain Adaptation and Generalization"

Arxiv: https://arxiv.org/pdf/1709.10190.pdf
Github: https://github.com/samotiian/CCSA

We take a PSPNet, and add a contrastive loss on its intermediate embeddings.
"""


class CCSA_PSPNet(nn.Module):
    """
    For the embedding function g, the original authors used the convolutional
    layers of the VGG-16 architecture [55] followed by 2 fully
    connected layers with output size of 1024 and 128, respectively. 
    For the prediction function h, they used a fully connected layer with 
    softmax activation.

    ResNet is our embedding function. Our classifier is PPM + Conv2d layers.
    The prediction function should include a softmax function inside of it, 
    we use 1x1 conv instead of fc layer.

    To create positive and negative pairs for training the network, for each 
    sample of a source domain the authors randomly selected 5 samples from 
    each remaining source domain, and help in this way to avoid overfitting. 
    However, to train a deeper network together with convolutional layers, the 
    authors state it is enough to create a large amount of positive and 
    negative pairs.

    We sample each minibatch uniformly from all domains, and then distribute
    among workers. Note that we 

    Since original authors compute CE loss only on sampled pairs, they compute
    CE on A first, then CE on B first, and contrastive loss A->B both times 
    (with single gradient update after both). We compute CE on all at once.

    CAN TAKE GRADIENT STEPS IN BETWEEN PAIR OF LOSSES, OR AFTER AGGREGATING BOTH LOSSES
    forward nad backward

    """
    def __init__(self, layers=50, bins=(1, 2, 3, 6), dropout=0.1, classes=2, zoom_factor=8, use_ppm=True, criterion=nn.CrossEntropyLoss(ignore_index=255), BatchNorm=nn.BatchNorm2d, pretrained=True, network_name=None):
        """

        nn.CrossEntropyLoss() combines nn.LogSoftmax() and nn.NLLLoss() in one single class.
        """
        super(CCSA_PSPNet, self).__init__()
        assert layers in [50, 101, 152]
        assert 2048 % len(bins) == 0
        assert classes > 1
        assert zoom_factor in [1, 2, 4, 8]
        self.zoom_factor = zoom_factor
        self.use_ppm = use_ppm
        self.criterion = criterion
        models.BatchNorm = BatchNorm

        if layers == 50:
            resnet = models.resnet50(pretrained=pretrained)
        elif layers == 101:
            resnet = models.resnet101(pretrained=pretrained)
        elif layers == 152:
            resnet = models.resnet152(pretrained=pretrained)


        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.conv2, resnet.bn2, resnet.relu, resnet.conv3, resnet.bn3, resnet.relu, resnet.maxpool)
        self.layer1, self.layer2, self.layer3, self.layer4 = resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4


        for n, m in self.layer3.named_modules():
            if 'conv2' in n:
                m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
            elif 'downsample.0' in n:
                m.stride = (1, 1)
        for n, m in self.layer4.named_modules():
            if 'conv2' in n:
                m.dilation, m.padding, m.stride = (4, 4), (4, 4), (1, 1)
            elif 'downsample.0' in n:
                m.stride = (1, 1)

        fea_dim = 2048
        if use_ppm:
            self.ppm = PPM(fea_dim, int(fea_dim/len(bins)), bins, BatchNorm)
            fea_dim *= 2
        self.cls = nn.Sequential(
            nn.Conv2d(fea_dim, 512, kernel_size=3, padding=1, bias=False),
            BatchNorm(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout),
            nn.Conv2d(512, classes, kernel_size=1)
        )
        if self.training:
            self.aux = nn.Sequential(
                nn.Conv2d(1024, 256, kernel_size=3, padding=1, bias=False),
                BatchNorm(256),
                nn.ReLU(inplace=True),
                nn.Dropout2d(p=dropout),
                nn.Conv2d(256, classes, kernel_size=1)
            )

    def forward(
        self, 
        x: torch.Tensor, 
        y: torch.Tensor=None, 
        batch_domain_idxs: torch.Tensor=None, 
        alpha: float = 0.25,
        num_pos_pairs: int=100):
        """
            Forward pass.
            
            Args:
            -   x: Tensor of shape (N,C,H,W)
            -   y: Tensor of shape (N,H,W) 
            -   batch_domain_idxs: Tensor of shape (N,) with domain ID
                    of each minibatch example.
            -   alpha: float acting as multiplier on contrastive loss
                    (convex combination)
            -   num_pos_pairs: number of pairs to use in contrastive loss

            Returns:
            -   logits
            -   main_loss
            -   aux_ce_loss
        """
        x_size = x.size()
        assert (x_size[2]-1) % 8 == 0 and (x_size[3]-1) % 8 == 0
        h = int((x_size[2] - 1) / 8 * self.zoom_factor + 1)
        w = int((x_size[3] - 1) / 8 * self.zoom_factor + 1)

        x = self.layer0(x) # get 128 channels, 4x downsample in H/W
        x = self.layer1(x) # get 256 channels, H/W constant
        x = self.layer2(x) # get 512 channels, 2x additional downsample in H/W
        x_tmp = self.layer3(x) # get 1024 channels, H/W constant
        x = self.layer4(x_tmp) # get 2048 channels, H/W constant

        resnet_embedding = x.clone()

        if self.use_ppm:
            x = self.ppm(x) # get 4096 channels from channel concat, H/W constant
        x = self.cls(x) # get n_classes channels, H/W constant
        if self.zoom_factor != 1: # get n_classes channels, back to input crop H/W (8x)
            x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=True)

        if self.training:
            aux = self.aux(x_tmp) # get n_classes channels, with 1/8 input crop H/W
            if self.zoom_factor != 1:
                aux = F.interpolate(aux, size=(h, w), mode='bilinear', align_corners=True)

            # ---- CCSA addition -----
            main_ce_loss = self.criterion(x, y)
            aux_ce_loss = self.criterion(aux, y)

            pos_pair_info, neg_pair_info = sample_pair_indices(
                y.type(torch.float32), # label map must be floats to use F.interpolate()
                batch_domain_idxs,
                num_pos_pairs = num_pos_pairs,
                neg_to_pos_ratio = 3,
                downsample_factor = 8)

            # y_c indicates if class indices are identical (examples are semantic pairs)
            y_c, a_embedding, b_embedding = get_merged_pair_embeddings(pos_pair_info, neg_pair_info, resnet_embedding)
 
            dists = paired_euclidean_distance(a_embedding, b_embedding)
            csa_loss = contrastive_loss(y_c, dists)

            # To balance the classification versus the contrastive semantic 
            # alignment portion of the loss (5), (7) and (8) are normalized
            # and weighted by (1-alpha) and by alpha
            main_loss = csa_loss * (alpha) + main_ce_loss * (1-alpha)
            aux_ce_loss *= (1-alpha)
            # ---- CCSA addition -----


            return x.max(1)[1], main_loss, aux_ce_loss
        else:
            return x

    
