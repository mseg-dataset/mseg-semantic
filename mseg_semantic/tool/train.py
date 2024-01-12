#!/usr/bin/python3

import math
import time
start = time.time()
from typing import Dict, Union
# import numpy as np
# import os
# import pdb
# import random

import apex
import torch
import torch.nn as nn
# import cv2

import mseg_semantic
from mseg_semantic.utils import transform

"""
Script to train models on the MSeg dataset using Pytorch DDP.
"""

# cv2.ocl.setUseOpenCL(False)
# cv2.setNumThreads(0)

MAX_NUM_EPOCHS = 100000 # we let epochs run forever, then exit when max number of iters is reached

def get_parser():
    """Merge config parameters and commend line arguments into `cfg` object"""
    import argparse
    from mseg_semantic.utils import config

    parser = argparse.ArgumentParser(description='PyTorch Semantic Segmentation')
    parser.add_argument('--config', type=str, default='config/ade20k/ade20k_pspnet50.yaml', help='config file')
    parser.add_argument('opts', help='see config/ade20k/ade20k_pspnet50.yaml for all options', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    assert args.config is not None
    cfg = config.load_cfg_from_cfg_file(args.config)
    if args.opts is not None:
        cfg = config.merge_cfg_from_list(cfg, args.opts)
    return cfg


def get_logger():
    """ Configure a Python logger to the logging.INFO verbosity level"""
    import logging
    logger_name = "main-logger"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    fmt = "[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d] %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)
    return logger


def worker_init_fn(worker_id):
    import random
    random.seed(args.manual_seed + worker_id)


def main_process() -> bool:
    return not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % args.ngpus_per_node == 0)


def main() -> None:
    """
    """
    import pickle

    import torch, os, math
    import torch.backends.cudnn as cudnn
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.nn.parallel
    import torch.optim
    import torch.utils.data
    import torch.multiprocessing as mp
    import torch.distributed as dist
    from mseg.utils.dataset_config import infos
    from mseg.taxonomy.taxonomy_converter import TaxonomyConverter
    from mseg.taxonomy.naive_taxonomy_converter import NaiveTaxonomyConverter
    from mseg_semantic.utils import config
    from mseg_semantic.utils.avg_meter import AverageMeter, SegmentationAverageMeter
    from mseg_semantic.utils.verification_utils import verify_architecture


    print('Using PyTorch version: ', torch.__version__)
    args = get_parser()
    assert isinstance(args.train_gpu, list)
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in args.train_gpu)

    ###### FLAT-MIX CODE #######################
    print("CUDA_VISIBLE_DEVICES: ", os.environ["CUDA_VISIBLE_DEVICES"])

    # Randomize args.dist_url too avoid conflicts on same machine
    args.dist_url = args.dist_url[:-2] + str(os.getpid() % 100).zfill(2)

    if isinstance(args.dataset, str): # only one dataset, i.e. 'single'
        # map to a list of GPU IDs
        args.dataset_gpu_mapping = {args.dataset: args.train_gpu}
        args.dataset = [args.dataset]
        print("args.dataset=", args.dataset)

    # train with multiple datasets, must be in the universal taxonomy space
    elif len(args.dataset) > 1 and args.universal:
        args.tc = NaiveTaxonomyConverter() if args.use_naive_taxonomy else TaxonomyConverter()

        args.data_root = {dataset:infos[dataset].dataroot for dataset in args.dataset}
        args.train_list = {dataset:infos[dataset].trainlist for dataset in args.dataset}
        args.classes = args.tc.num_uclasses
        # args.save_path = args.save_path.replace("{}", '-'.join([infos[dataset].shortname for dataset in args.dataset]))

    elif (len(args.dataset) == 1) and args.universal: # single dataset on universal taxonomy training
        args.tc = TaxonomyConverter(train_datasets=args.dataset)
        args.data_root = infos[args.dataset[0]].dataroot
        args.train_list = infos[args.dataset[0]].trainlist
        args.classes = args.tc.num_uclasses
        # args.save_path = args.save_path.replace("{}", info[args.dataset].shortname)

    elif (len(args.dataset) == 1) and (not args.universal): # single dataset on self taxonomy training
        args.data_root = infos[args.dataset[0]].dataroot
        args.train_list = infos[args.dataset[0]].trainlist
        args.classes = infos[args.dataset[0]].num_classes
        # args.save_path = args.save_path.replace("{}", infos[args.dataset].shortname)
    else:
        raise RuntimeError('Incorrect training configuration, please verify your config params.')
    
    # verify arch after args.classes is populated
    verify_architecture(args)

    if args.manual_seed is not None:
        cudnn.benchmark = False
        cudnn.deterministic = True
        torch.manual_seed(args.manual_seed)
        np.random.seed(args.manual_seed)
        torch.manual_seed(args.manual_seed)
        torch.cuda.manual_seed_all(args.manual_seed)
    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])
    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    args.ngpus_per_node = len(args.train_gpu)
    if len(args.train_gpu) == 1:
        args.sync_bn = False
        args.distributed = False
        args.multiprocessing_distributed = False
    if args.multiprocessing_distributed:
        args.world_size = args.ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=args.ngpus_per_node, args=(args.ngpus_per_node, args))
    else:
        main_worker(args.train_gpu, args.ngpus_per_node, args)


def get_dataset_split_transform(
    args, split: str
) -> transform.Compose:
    """Return the input data transform (w/ data augmentations)

    Args:
        args: experiment parameters
        split: dataset split, either 'train' or 'val'

    Return:
        Runtime data transformation object that is callable
    """
    from mseg_semantic.utils.normalization_utils import get_imagenet_mean_std

    mean, std = get_imagenet_mean_std()
    if split == "train":
        transform_list = [
            transform.ResizeShort(args.short_size),
            transform.RandScale([args.scale_min, args.scale_max]),
            transform.RandRotate(
                [args.rotate_min, args.rotate_max],
                padding=mean,
                ignore_label=args.ignore_label,
            ),
            transform.RandomGaussianBlur(),
            transform.RandomHorizontalFlip(),
            transform.Crop(
                [args.train_h, args.train_w],
                crop_type="rand",
                padding=mean,
                ignore_label=args.ignore_label,
            ),
            transform.ToTensor(),
            transform.Normalize(mean=mean, std=std),
        ]
    elif split == "val":
        transform_list = [
            transform.Crop(
                [args.train_h, args.train_w],
                crop_type="center",
                padding=mean,
                ignore_label=args.ignore_label,
            ),
            transform.ToTensor(),
            transform.Normalize(mean=mean, std=std),
        ]
    else:
        raise RuntimeError("Unknown split. Quitting ...")

    if len(args.dataset) > 1 and args.universal:
        transform_list += [
            transform.ToUniversalLabel(
                args.dataset_name, use_naive_taxonomy=args.use_naive_taxonomy
            )
        ]
    elif len(args.dataset) == 1 and args.universal:
        # never run naive taxonomy baseline for training with a single dataset
        transform_list += [transform.ToUniversalLabel(args.dataset[0])]

    return transform.Compose(transform_list)




def load_pretrained_weights(args, model, optimizer): 
    """
    Returns: model (if args.resume is a model, loads the model,
    if it is a directory, find the latest model in that directory)
    """
    import torch, os, math

    resume_iter = 0

    if args.weight:
        if os.path.isfile(args.weight):
            if main_process():
                logger.info("=> loading weight '{}'".format(args.weight))
            checkpoint = torch.load(args.weight)
            model.load_state_dict(checkpoint['state_dict'])
            if main_process():
                logger.info("=> loaded weight '{}'".format(args.weight))
        else:
            if main_process():
                logger.info("=> no weight found at '{}'".format(args.weight))

    if args.resume:
        if os.path.isfile(args.resume):
            if main_process():
                logger.info("=> loading checkpoint '{}'".format(args.resume))
            # checkpoint = torch.load(args.resume)
            checkpoint = torch.load(args.resume, map_location=lambda storage, loc: storage.cuda())
            # args.start_epoch = checkpoint['epoch']
            args.start_epoch = 0 # we don't rely on this, but on resume_iter
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            resume_iter = checkpoint['current_iter']
            if main_process():
                logger.info("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            if main_process():
                logger.info("=> no checkpoint found at '{}'".format(args.resume) + ' Please check')
                exit()

    if args.auto_resume and (args.auto_resume != 'None'):
        import glob
        if main_process():
            logger.info("=> loading latest checkpoint from folder'{}'".format(args.auto_resume))

        print("Auto resume training? ", args.auto_resume)
        filelist = glob.glob(args.auto_resume + '/*.pth')
        print(os.getcwd())
        print(filelist)
        filename = [file.split('/')[-1] for file in filelist]
        filename = [file.replace('.pth', '') for file in filename]
        # epochlist = []
        if 'train_epoch_final' in filename:
            if main_process():
                logger.info("Training already finished, no need to resume!!")
                exit()
        else:
            print(filename)
            epochs = [file.split('_')[-1] for file in filename]
            epochs = [epoch for epoch in epochs if epoch.isdigit()]
            epochs = [int(epoch) for epoch in epochs]
            max_epoch = max(epochs)

            filename = 'train_epoch_{}.pth'.format(max_epoch)

            model_path = os.path.join(args.auto_resume, filename)
            logger.info(model_path)
            print(0, max_epoch, model_path, os.path.isfile(model_path))

        
        if os.path.isfile(model_path):
            if main_process():
                logger.info("=> loading checkpoint '{}'".format(model_path))

            checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage.cuda())
            # args.start_epoch = checkpoint['epoch']
            args.start_epoch = 0 # we don't rely on this, but on resume_iter
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            resume_iter = checkpoint['current_iter']

            args.epoch_history = checkpoint['epoch']

            if main_process():
                logger.info("=> loaded checkpoint '{}' (epoch history: {})".format(model_path, checkpoint['epoch']))
        else:
            if main_process():
                logger.info("=> no checkpoint found at '{}'".format(model_path) + ' Please check')
                exit()

    return model, optimizer, resume_iter

# optimizer = get_optimizer(args.model)


def get_model(
    args,
    criterion: nn.Module,
    BatchNorm: Union[
        torch.nn.SyncBatchNorm, apex.parallel.SyncBatchNorm, nn.BatchNorm2d
    ],
) -> nn.Module:
    """ Build the semantic segmentation model """
    if args.arch == "psp":
        from mseg_semantic.model.pspnet import PSPNet

        model = PSPNet(
            layers=args.layers,
            classes=args.classes,
            zoom_factor=args.zoom_factor,
            criterion=criterion,
            BatchNorm=BatchNorm,
            network_name=args.network_name,
        )
    elif args.arch == "hrnet":
        from mseg_semantic.model.seg_hrnet import get_configured_hrnet

        # note apex batchnorm is hardcoded
        model = get_configured_hrnet(args.classes)
    elif args.arch == "hrnet_ocr":
        from mseg_semantic.model.seg_hrnet_ocr import get_configured_hrnet_ocr

        model = get_configured_hrnet_ocr(args.classes)
    return model


def get_optimizer(args, model: nn.Module) -> torch.optim.Optimizer:
    """
    Create an optimizer and provide model parameters to it.

    For PSPNet, the learning rate is module-specfiic; the first 5 entries (ResNet backbone)
    have low learning rate to not clobber pre-trained weights, and later entries (PPM derivatives)
    have high learning rate.
    """
    import torch, os, math

    # HRNet settings
    if args.arch == "hrnet" or args.arch == "hrnet_ocr":
        optimizer = torch.optim.SGD(
            [
                {
                    "params": filter(lambda p: p.requires_grad, model.parameters()),
                    "lr": args.base_lr,
                }
            ],
            lr=args.base_lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
        return optimizer

    if args.arch != "psp":
        raise RuntimeError("Unknown network architecture")
    # PSPNet settings
    modules_original = [
        model.layer0,
        model.layer1,
        model.layer2,
        model.layer3,
        model.layer4,
    ]
    modules_new = [model.ppm, model.cls, model.aux]
    params_list = []
    for module in modules_original:
        params_list.append(dict(params=module.parameters(), lr=args.base_lr))

    for module in modules_new:
        params_list.append(dict(params=module.parameters(), lr=args.base_lr * 10))
    NUM_PRETRAINED_RESNET_LAYERS = 5
    args.index_split = NUM_PRETRAINED_RESNET_LAYERS
    optimizer = torch.optim.SGD(
        params_list,
        lr=args.base_lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    return optimizer


def get_rank_to_dataset_map(args) -> Dict[int, str]:
    """
    Obtain a mapping from GPU rank (index) to the name of the dataset residing on this GPU.
    """
    rank_to_dataset_map = {}
    for dataset, gpu_idxs in args.dataset_gpu_mapping.items():
        for gpu_idx in gpu_idxs:
            rank_to_dataset_map[gpu_idx] = dataset
    logger.info("Rank to dataset map: ", rank_to_dataset_map)
    return rank_to_dataset_map


def set_number_of_training_iters(args):
    """
    There are two scenarios we consider to determine number of required training iters
    when training on MSeg. We set a max number of training crops, and then subdivide the
    work between our GPUs.

    1. We are training with a single dataset. Suppose we want to train for 1 million
        crops in total (args.num_examples). Suppose our dataset has 18k images. Then
        we will train for 56 epochs. Suppose our training node has 8 GPUs. Then
        with a batch size of 32, and 8 GPUs, we need ~3906 iterations to reach 1M crops.

    2. We are mixing many datasets together. We determine which dataset this GPU
            is assigned to. Each GPU runs 1 process, and multiple GPU IDs (referred to
            as replicas) may be assigned to a single dataset. The computation is the same
            as before, except instead of counting all of the GPUs on the node, we only
            count the number of replicas counting towards this dataset.
    """
    # single dataset training
    if (len(args.dataset) == 1) and (not args.use_mgda):
        from util.txt_utils import read_txt_file
        # number of examples for 1 epoch of this dataset
        num_d_examples = len(read_txt_file(infos[args.dataset[0]].trainlist))
        # number of examples to train for in total
        num_examples_total = args.num_examples

        args.epochs = math.ceil(num_examples_total / num_d_examples)
        args.max_iters = math.floor(num_examples_total / (args.batch_size * args.ngpus_per_node))

        # on small datasets, avoid saving checkpoints too frequently in order to not waste time
        if args.epochs > 1000:
            args.save_freq = args.epochs // 100

    # multiple dataset training
    elif len(args.dataset) > 1:
        rank_to_dataset_map = get_rank_to_dataset_map(args)
        # # which dataset this gpu is for
        args.dataset_name = rank_to_dataset_map[args.rank]
        # within this dataset, its rank, i.e. 0,1,2,3 etc gpu ID assigned to this dataset
        args.dataset_rank = args.dataset_gpu_mapping[args.dataset_name].index(args.rank)
        args.num_replica_per_dataset = len(args.dataset_gpu_mapping[args.dataset_name])

        args.max_iters = math.floor(args.num_examples / (args.batch_size * args.num_replica_per_dataset))
        logger.info(f'max_iters = {args.max_iters}')

    return args


def main_worker(gpu: int, ngpus_per_node: int, argss) -> None:
    """ Each GPU process will execute this function"""
    global args
    args = argss

    import apex
    import torch, os, math
    import torch.backends.cudnn as cudnn
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.nn.parallel
    import torch.optim
    import torch.utils.data

    import torch.multiprocessing as mp
    import torch.distributed as dist

    from mseg.utils.dataset_config import infos
    from mseg.taxonomy.taxonomy_converter import TaxonomyConverter

    from mseg_semantic.multiobjective_opt.dist_mgda_utils import scale_loss_and_gradients
    from mseg_semantic.utils import config
    from mseg_semantic.utils import dataset
    from mseg_semantic.utils.avg_meter import AverageMeter, SegmentationAverageMeter
    from mseg_semantic.utils.training_utils import poly_learning_rate
    from mseg_semantic.utils.verification_utils import verify_architecture

    if args.sync_bn:
        if args.multiprocessing_distributed:
            # BatchNorm = torch.nn.SyncBatchNorm
            BatchNorm = apex.parallel.SyncBatchNorm
        else:
            raise RuntimeError("Batch norm not supported for DataParallel at this time")
    else:
        BatchNorm = nn.BatchNorm2d
    print('Using batchnorm variant: ', BatchNorm)
    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank)

    criterion = nn.CrossEntropyLoss(ignore_index=args.ignore_label)
    model = get_model(args, criterion, BatchNorm)
    optimizer = get_optimizer(args, model)

    global logger
    logger = get_logger()
    args.logger = logger
    
    if main_process():
        logger.info(args)
        logger.info("=> creating model ...")
        logger.info("Classes: {}".format(args.classes))
        logger.info(model)

    if args.distributed:
        torch.cuda.set_device(gpu)
        args.batch_size = int(args.batch_size / ngpus_per_node)
        args.batch_size_val = int(args.batch_size_val / ngpus_per_node)
        args.batch_size_val = max(1, args.batch_size_val)
        args.workers = int(args.workers / ngpus_per_node)
        if args.use_apex:
            model, optimizer = apex.amp.initialize(model.cuda(), optimizer, opt_level=args.opt_level, keep_batchnorm_fp32=args.keep_batchnorm_fp32, loss_scale=args.loss_scale)
            model = apex.parallel.DistributedDataParallel(model)
        else:
            model = torch.nn.parallel.DistributedDataParallel(model.cuda(), device_ids=[gpu])

    else:
        model = torch.nn.DataParallel(model.cuda())

    model, optimizer, args.resume_iter = load_pretrained_weights(args, model, optimizer)

    args = set_number_of_training_iters(args)
    train_transform = get_dataset_split_transform(args, split='train')  

    # Consider if a dataset has size 18,000 and is placed on a single GPU, of 4 gpus. 
    # Batch size 32. In this case, len(train_data) = 18,000 but len(train_loader) = 2250
    # Because effective batch size is 8.

    # Consider if a dataset has size 118287. If placed on 2/4 gpus with batch size 32.
    # In this case, len(train_data) = 118287 and len(train_loader) = 7393.
    if len(args.dataset) > 1:
        # FLATMIX ADDITION
        train_data = dataset.SemData(split='train', data_root=args.data_root[args.dataset_name], data_list=args.train_list[args.dataset_name], transform=train_transform)
        iters_per_epoch = math.floor((len(train_data) / (args.batch_size * args.num_replica_per_dataset)))
        args.epochs = math.ceil(args.max_iters / iters_per_epoch)
        print(f'''Rank: {args.rank}, Dataset: {args.dataset_name}, replicas: {args.num_replica_per_dataset}, length of dataset: {len(train_data)}, max_iter: {args.max_iters}, batch_size: {args.batch_size},  
            iters_per_epoch: {iters_per_epoch}, epochs: {args.epochs}, ''')
    else:
        train_data = dataset.SemData(split='train', data_root=args.data_root, data_list=args.train_list, transform=train_transform)
    
    logger.info(f'Train data has len {len(train_data)} on {args.rank}')
    if args.distributed:
        if len(args.dataset) > 1:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_data, num_replicas=args.num_replica_per_dataset, rank=args.dataset_rank)
            logger.info(f"rank: {args.rank}, dataset_rank: {args.dataset_rank}, replica: {args.num_replica_per_dataset}, actual_replica: {train_sampler.num_replicas}, length of sampler, {len(train_sampler)}")
        else:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_data, num_replicas=args.ngpus_per_node, rank=args.rank)
            logger.info(f"rank: {args.rank}, actual_replica: {train_sampler.num_replicas}, length of sampler, {len(train_sampler)}")

    else:
        train_sampler = None
    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        num_workers=args.workers,
        pin_memory=True,
        sampler=train_sampler,
        drop_last=True
    )
    logger.info(f'Train loader has len {len(train_loader)} on {args.rank}')

    if args.evaluate:
        val_transform = get_dataset_split_transform(args, split='val')
        # val_transform = transform.Compose(val_transform_list)
        val_data = dataset.SemData(split='val', data_root=args.data_root, data_list=args.val_list, transform=val_transform)
        if args.distributed:
            val_sampler = torch.utils.data.distributed.DistributedSampler(val_data)
        else:
            val_sampler = None
        val_loader = torch.utils.data.DataLoader(
            val_data,
            batch_size=args.batch_size_val,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True,
            sampler=val_sampler
        )

    for epoch in range(args.start_epoch, args.epochs + MAX_NUM_EPOCHS):

        epoch_log = epoch + 1
        if args.auto_resume != 'None': # if it is a resumed training
            epoch_log += args.epoch_history # only the main process, acting like "total_epoch"
        logger.info(f'New epoch {epoch_log} starts on rank {args.rank}')

        if args.distributed:
            train_sampler.set_epoch(epoch)
        loss_train, mIoU_train, mAcc_train, allAcc_train = train(train_loader, model, optimizer, epoch)

        if ((epoch_log % args.save_freq == 0)) and main_process():
            filename = args.save_path + '/train_epoch_' + str(epoch_log) + '.pth'
            logger.info('Saving checkpoint to: ' + filename)
            torch.save({'epoch': epoch_log, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(), 
                'current_iter': (epoch_log) * len(train_loader), 'max_iter': args.max_iters}, filename)
            # latestname = args.save_path + '/train_epoch_' + str(epoch_log) + '.pth'
            if epoch_log / args.save_freq > 2:
                # if (epoch_log - 3) % 10 != 0:
                deletename = args.save_path + '/train_epoch_' + str(epoch_log - args.save_freq * 2) + '.pth'
                os.remove(deletename)

        if (epoch_log == args.epochs) and main_process():
            filename = args.save_path + '/train_epoch_final.pth'
            logger.info('Saving checkpoint to: ' + filename)
            torch.save({'epoch': epoch_log, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(), 
                'current_iter': (epoch_log) * len(train_loader) + args.resume_iter, 'max_iter': args.max_iters}, filename)
            exit()


        # if args.evaluate:
        #     loss_val, mIoU_val, mAcc_val, allAcc_val = validate(val_loader, model, criterion)


def train(train_loader, model, optimizer: torch.optim.Optimizer, epoch: int):
    """ Run one training epoch """
    import torch, os, math, time
    import torch.distributed as dist

    from mseg_semantic.multiobjective_opt.dist_mgda_utils import scale_loss_and_gradients
    from mseg_semantic.utils.avg_meter import AverageMeter, SegmentationAverageMeter
    from mseg_semantic.utils.training_utils import poly_learning_rate

    batch_time = AverageMeter()
    data_time = AverageMeter()
    main_loss_meter = AverageMeter()
    aux_loss_meter = AverageMeter()
    loss_meter = AverageMeter()
    sam = SegmentationAverageMeter()

    model.train()

    end = time.time()
    max_iter = args.max_iters
    for i, (input, target) in enumerate(train_loader):

        data_time.update(time.time() - end)
        if args.zoom_factor != 8:
            h = int((target.size()[1] - 1) / 8 * args.zoom_factor + 1)
            w = int((target.size()[2] - 1) / 8 * args.zoom_factor + 1)
            # 'nearest' mode doesn't support align_corners mode and 'bilinear' mode is fine for downsampling
            target = F.interpolate(target.unsqueeze(1).float(), size=(h, w), mode='bilinear', align_corners=True).squeeze(1).long()
        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        if args.use_mgda:
            output, loss, main_loss, aux_loss, scales = forward_backward_mgda(input, target, model, optimizer, args)
        else:
            output, loss, main_loss, aux_loss = forward_backward_full_sync(input, target, model, optimizer, args)
        optimizer.step()

        n = input.size(0)
        if args.multiprocessing_distributed:
            main_loss, aux_loss, loss = main_loss.detach() * n, aux_loss * n, loss * n  # not considering ignore pixels
            count = target.new_tensor([n], dtype=torch.long)
            dist.all_reduce(main_loss), dist.all_reduce(aux_loss), dist.all_reduce(loss), dist.all_reduce(count)
            n = count.item()
            main_loss, aux_loss, loss = main_loss / n, aux_loss / n, loss / n

        sam.update_metrics_gpu(output, target, args.classes, args.ignore_label, args.multiprocessing_distributed)

        main_loss_meter.update(main_loss.item(), n)
        aux_loss_meter.update(aux_loss.item(), n)
        loss_meter.update(loss.item(), n)
        # if main_process():
        if i > 0:
            batch_time.update(time.time() - end)
        end = time.time()

        # print(len(train_loader))
        # logger.info(len(train_loader))

        current_iter = epoch * len(train_loader) + i + 1 + args.resume_iter
        current_lr = poly_learning_rate(args.base_lr, current_iter, max_iter, power=args.power)

        # logger.info(f'LR:{current_lr}, base_lr: {args.base_lr}, current_iter:{current_iter}, max_iter:{max_iter}, power:{args.power}')

        if args.arch == 'psp':
            for index in range(0, args.index_split):
                optimizer.param_groups[index]['lr'] = current_lr
            for index in range(args.index_split, len(optimizer.param_groups)):
                optimizer.param_groups[index]['lr'] = current_lr * 10

        elif args.arch == 'hrnet' or args.arch == 'hrnet_ocr':
            optimizer.param_groups[0]['lr'] = current_lr

        remain_iter = max_iter - current_iter
        remain_time = remain_iter * batch_time.avg
        t_m, t_s = divmod(remain_time, 60)
        t_h, t_m = divmod(t_m, 60)
        remain_time = '{:02d}:{:02d}:{:02d}'.format(int(t_h), int(t_m), int(t_s))

        if (current_iter) % args.print_freq == 0 and True:
            logger.info('Epoch: [{}/{}][{}/{}] '
                        'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                        'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                        'Remain {remain_time} '
                        'MainLoss {main_loss_meter.val:.4f} '
                        'AuxLoss {aux_loss_meter.val:.4f} '
                        'LR {current_lr:.8f} '
                        'Loss {loss_meter.val:.4f} '
                        'Accuracy {accuracy:.4f}.'.format(epoch+1, args.epochs, i + 1, len(train_loader),
                                                          batch_time=batch_time,
                                                          data_time=data_time,
                                                          remain_time=remain_time,
                                                          main_loss_meter=main_loss_meter,
                                                          aux_loss_meter=aux_loss_meter,
                                                          current_lr=current_lr,
                                                          loss_meter=loss_meter,
                                                          accuracy=sam.accuracy) + f'current_iter: {current_iter}' + f' rank: {args.rank} ')
            if args.use_mgda and main_process():
                # Scales identical in each process, so print out only in main process.
                scales_str = [f'{d}: {scale:.2f}' for d,scale in scales.items()]
                scales_str = ' , '.join(scales_str)
                logger.info(f'Scales: {scales_str}')

        if main_process() and current_iter == max_iter - 5: # early exit to prevent iter number not matching between gpus
            break

    iou_class, accuracy_class, mIoU, mAcc, allAcc = sam.get_metrics()
    logger.info('Train result at epoch [{}/{}]: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(epoch+1, args.epochs, mIoU, mAcc, allAcc))
    return main_loss_meter.avg, mIoU, mAcc, allAcc


def forward_backward_full_sync(
    input: torch.Tensor,
    target: torch.Tensor,
    model,
    optimizer: torch.optim.Optimizer,
    args,
):
    """
    Args:
        input: Tensor of size NCHW representing
        target: Tensor of size (?) representing
        model
        optimizer
        args

    Returns:
        output: Tensor of size (?) representing
        loss: Tensor of size (?) representing
        main_loss: Tensor of size (?) representing
        aux_loss: Tensor of size (?) representing
    """
    output, main_loss, aux_loss = model(input, target)
    if not args.multiprocessing_distributed:
        main_loss, aux_loss = torch.mean(main_loss), torch.mean(aux_loss)
    loss = main_loss + args.aux_weight * aux_loss

    optimizer.zero_grad()
    if args.use_apex and args.multiprocessing_distributed:
        with apex.amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
    else:
        loss.backward()
    return output, loss, main_loss, aux_loss


def forward_backward_mgda(input: torch.Tensor, target: torch.Tensor, model, optimizer, args):
    """
        We rely upon the ddp.no_sync() of gradients:
        https://github.com/pytorch/pytorch/blob/master/torch/nn/parallel/distributed.py

        Args:
        -   input: Tensor of size (?) representing
        -   target: Tensor of size (?) representing
        -   model
        -   optimizer
        -   args

        Returns:
        -   output: Tensor of size (?) representing
        -   loss: Tensor of size (?) representing
        -   main_loss: Tensor of size (?) representing
        -   aux_loss: Tensor of size (?) representing
    """
    from mseg_semantic.multiobjective_opt.dist_mgda_utils import scale_loss_and_gradients
    with model.no_sync():
        output, main_loss, aux_loss = model(input, target)
        loss = main_loss + args.aux_weight * aux_loss
        loss, scales = scale_loss_and_gradients(loss, optimizer, model, args)
        
    return output, loss, main_loss, aux_loss, scales


def validate(val_loader, model, criterion: nn.Module):
    if main_process():
        logger.info('>>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>')
    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    sam = SegmentationAverageMeter()

    model.eval()
    if main_process():
        end = time.time()
    for i, (input, target) in enumerate(val_loader):
        if main_process():
            data_time.update(time.time() - end)
        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        output = model(input)
        if args.zoom_factor != 8:
            output = F.interpolate(output, size=target.size()[1:], mode='bilinear', align_corners=True)
        loss = criterion(output, target)

        n = input.size(0)
        if args.multiprocessing_distributed:
            loss = loss * n  # not considering ignore pixels
            count = target.new_tensor([n], dtype=torch.long)
            dist.all_reduce(loss), dist.all_reduce(count)
            n = count.item()
            loss = loss / n
        else:
            loss = torch.mean(loss)

        output = output.max(1)[1]
        sam.update_metrics_gpu(output, target, args.classes, args.ignore_label, args.multiprocessing_distributed)
        loss_meter.update(loss.item(), input.size(0))
        if main_process():
            batch_time.update(time.time() - end)
            end = time.time()
        if ((i + 1) % args.print_freq == 0) and main_process():
            logger.info('Test: [{}/{}] '
                        'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                        'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                        'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f}) '
                        'Accuracy {accuracy:.4f}.'.format(i + 1, len(val_loader),
                                                          data_time=data_time,
                                                          batch_time=batch_time,
                                                          loss_meter=loss_meter,
                                                          accuracy=sam.accuracy))

    iou_class, accuracy_class, mIoU, mAcc, allAcc = sam.get_metrics()
    if main_process():
        logger.info('Val result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(mIoU, mAcc, allAcc))
        for i in range(args.classes):
            logger.info('Class_{} Result: iou/accuracy {:.4f}/{:.4f}.'.format(i, iou_class[i], accuracy_class[i]))
        logger.info('<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<')
    return loss_meter.avg, mIoU, mAcc, allAcc

end = time.time()
print(end-start)

if __name__ == '__main__':

    main()
