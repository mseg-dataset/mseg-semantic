

import torch
import torch.nn as nn

from domain_generalization.ccsa_pspnet import CCSA_PSPNet


def test_CCSA_PSPNet_dims():
    """ """
    layers = 50
    classes = 183
    network_name = None
    zoom_factor = 8  # zoom factor for final prediction during training, be in [1, 2, 4, 8]
    ignore_label = 255
    criterion = nn.CrossEntropyLoss(ignore_index=ignore_label)
    BatchNorm = torch.nn.BatchNorm2d # torch.nn.SyncBatchNorm
    model = CCSA_PSPNet(
        layers=layers,
        classes=classes,
        zoom_factor=zoom_factor,
        criterion=criterion,
        BatchNorm=BatchNorm,
        network_name=network_name,
        pretrained=False) # unlike actual training time.

    x = torch.randint(high=255, size=(4,3,201,201)).type(torch.float32)
    y = torch.randint(high=10,size=(4,201,201))
    batch_domain_idxs = torch.tensor([0,1,2,1])

    out_cache = model(x,y,batch_domain_idxs)


def test_CCSA_PSPNet_dims_cuda():
    """ """
    layers = 50
    classes = 183
    network_name = None
    zoom_factor = 8  # zoom factor for final prediction during training, be in [1, 2, 4, 8]
    ignore_label = 255
    criterion = nn.CrossEntropyLoss(ignore_index=ignore_label)
    BatchNorm = torch.nn.BatchNorm2d # torch.nn.SyncBatchNorm
    model = CCSA_PSPNet(
        layers=layers,
        classes=classes,
        zoom_factor=zoom_factor,
        criterion=criterion,
        BatchNorm=BatchNorm,
        network_name=network_name,
        pretrained=False) # unlike actual training time.

    model = model.cuda()

    x = torch.randint(high=255, size=(4,3,201,201)).type(torch.float32)
    y = torch.randint(high=10,size=(4,201,201))
    batch_domain_idxs = torch.tensor([0,1,2,1])

    x = x.cuda()
    y = y.cuda()
    batch_domain_idxs = batch_domain_idxs.cuda()

    out_cache = model(x,y,batch_domain_idxs)


if __name__ == '__main__':
    """ """
    test_CCSA_PSPNet_dims()
    test_CCSA_PSPNet_dims_cuda()

