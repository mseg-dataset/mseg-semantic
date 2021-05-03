#!/usr/bin/python3

import time
start = time.time()
# time.sleep(2)

import apex
# import cv2



# import math
# import numpy as np
# import os
# import pdb
# import random

# from taxonomy.utils_flat import *







# end = time.time()
# print(end - start)


"""
TODO: GET THE MODELS TRAINING, THEN GO BACK LATER AND WRITE THE
UNIT TESTS FOR TAXONOMY CONVERTER

Should have fixed ratios --> then experiment with it.

Train w/ MGDA
Train w/o MGDA.
Get results on the training set as well.

Submit the jobs first -- for all training/test sets.

fix the max_iters -- 1.2 Million examples

make sure we have the right flags to evaluate on the train dataset.
"""

"""
NVIDIA Apex has 4 optimization levels:

    O0 (FP32 training): basically a no-op. Everything is FP32 just as before.
    O1 (Conservative Mixed Precision): only some whitelist ops are done in FP16.
    O2 (Fast Mixed Precision): this is the standard mixed precision training. 
        It maintains FP32 master weights and optimizer.step acts directly on the FP32 master weights.
    O3 (FP16 training): full FP16. Passing keep_batchnorm_fp32=True can speed 
        things up as cudnn batchnorm is faster anyway.
"""



class ToFlatLabel(object):
    def __init__(self, tc_init, dataset):
        self.dataset = dataset
        self.tc = tc_init
 
    def __call__(self, image, label):
        return image, self.tc.transform_label(label, self.dataset)
 
# cv2.ocl.setUseOpenCL(False)
# cv2.setNumThreads(0)


def get_parser():
    import argparse
    from util import config

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


def main_process():
    return not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % args.ngpus_per_node == 0)


def main():
    """
    """
    # with open('test_2.txt', 'a') as f:
    #     f.write('test')
    #     f.close()
    import torch, os, math
    import torch.backends.cudnn as cudnn
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.nn.parallel
    import torch.optim
    import torch.utils.data

    import torch.multiprocessing as mp
    import torch.distributed as dist
# from tensorboardX import SummaryWriter
    from util.dataset_config import infos

    from util import config
    from util.verification_utils import verify_architecture
    from util.avg_meter import AverageMeter, SegmentationAverageMeter
    from taxonomy.utils_flat import TaxonomyConverter
    from taxonomy.utils_baseline import StupidTaxonomyConverter
    import pickle


    print('Using PyTorch version: ', torch.__version__)
    args = get_parser()
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in args.train_gpu)


    ###### FLAT-MIX CODE #######################
    print(os.environ["CUDA_VISIBLE_DEVICES"])

    # Randomize args.dist_url too avoid conflicts on same machine
    args.dist_url = args.dist_url[:-2] + str(os.getpid() % 100).zfill(2)

    from util.dataset_config import infos

    
    if args.use_multiple_datasets and args.universal: # multiple datasets training, must be on universal taxononmy
        if args.tax_version == 0:
            args.tc = StupidTaxonomyConverter(version=args.tax_version)
        else:
            if args.finetune:
                args.tc = TaxonomyConverter(version=args.tax_version, finetune=True, finetune_dataset=args.finetune_dataset)
            else:
                args.tc = TaxonomyConverter(version=args.tax_version) #, train_datasets=args.dataset, test_datasets=args.test_dataset) #, train_datasets=args.dataset, test_datasets=args.test_dataset)

        args.data_root = {dataset:infos[dataset].dataroot for dataset in args.dataset}
        args.train_list = {dataset:infos[dataset].trainlist for dataset in args.dataset}
        args.classes = args.tc.classes
        # args.save_path = args.save_path.replace("{}", '-'.join([infos[dataset].shortname for dataset in args.dataset]))

    elif (not args.use_multiple_datasets) and args.universal: # single dataset on universal taxonomy training
        args.tc = TaxonomyConverter(version=args.tax_version, train_datasets=[args.dataset], test_datasets=args.test_dataset)
        args.data_root = infos[args.dataset].dataroot
        args.train_list = infos[args.dataset].trainlist
        args.classes = args.tc.classes
        # args.save_path = args.save_path.replace("{}", info[args.dataset].shortname)

    elif (not args.use_multiple_datasets) and (not args.universal): # single dataset on self taxnonmy training
        args.data_root = infos[args.dataset].dataroot
        args.train_list = infos[args.dataset].trainlist
        args.classes = infos[args.dataset].num_classes
        # args.save_path = args.save_path.replace("{}", infos[args.dataset].shortname)
    else:
        print('wrong mode, please check')
        exit()
    
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


def get_train_transform_list(args, split, dataset_name):
    """
        Args:
        -   args:
        -   split

        Return:
        -   List of transforms
    """
    from util.normalization_utils import get_imagenet_mean_std
    from util import transform

    mean, std = get_imagenet_mean_std()
    if split == 'train':
        transform_list = [
            transform.RandScale([args.scale_min, args.scale_max]),
            transform.RandRotate([args.rotate_min, args.rotate_max], padding=mean, ignore_label=args.ignore_label),
            transform.RandomGaussianBlur(),
            transform.RandomHorizontalFlip(),
            transform.Crop([args.train_h, args.train_w], crop_type='rand', padding=mean, ignore_label=args.ignore_label),
            transform.ToTensor(),
            transform.Normalize(mean=mean, std=std)
        ]
    elif split == 'val':
        transform_list = [
            transform.Crop([args.train_h, args.train_w], crop_type='center', padding=mean, ignore_label=args.ignore_label),
            transform.ToTensor(),
            transform.Normalize(mean=mean, std=std)
        ]
    else:
        print('Unknown split. Quitting ...')
        quit()

    transform_list += [ToFlatLabel(args.tc, dataset_name)]

    return transform.Compose(transform_list)


def load_pretrained_weights(args, model, optimizer):
    """
        Args:
        -   args
        -   model: Passed by reference

        Returns:
        -   None
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
            args.start_epoch = 0 # we don't really rely on this, but on resume_iter
            if args.finetune:
                args.start_epoch = 0
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            resume_iter = checkpoint['current_iter']
            if main_process():
                logger.info("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            if main_process():
                logger.info("=> no checkpoint found at '{}'".format(args.resume) + ' Please check')
                exit()

    return model, optimizer, resume_iter

            # optimizer = get_optimizer(args.model)



def get_model(args, criterion, BatchNorm):
    """
        Args:
        -   

        Returns:
        -   
    """
    if args.arch == 'psp':
        from ccsa.ccsa_pspnet import CCSA_PSPNet
        model = CCSA_PSPNet(layers=args.layers, classes=args.classes, zoom_factor=args.zoom_factor, criterion=criterion, BatchNorm=BatchNorm, network_name=args.network_name)


    elif args.arch == 'psa':
        model = PSANet(layers=args.layers, classes=args.classes, zoom_factor=args.zoom_factor, psa_type=args.psa_type,
                       compact=args.compact, shrink_factor=args.shrink_factor, mask_h=args.mask_h, mask_w=args.mask_w,
                       normalization_factor=args.normalization_factor, psa_softmax=args.psa_softmax,
                       criterion=criterion,
                       BatchNorm=BatchNorm)
    return model


def get_optimizer(args, model):
    """
    Create a parameter list, where first 5 entries (ResNet backbone) have low learning rate
    to not clobber pre-trained weights, and later entries (PPM derivatives) have high learning rate.

        Args:
        -   args
        -   model

        Returns:
        -   optimizer
    """
    import torch, os, math

    if args.arch == 'psp':
        modules_ori = [model.layer0, model.layer1, model.layer2, model.layer3, model.layer4]
        modules_new = [model.ppm, model.cls, model.aux]
    elif args.arch == 'psa':
        modules_ori = [model.layer0, model.layer1, model.layer2, model.layer3, model.layer4]
        modules_new = [model.psa, model.cls, model.aux]
    params_list = []
    for module in modules_ori:
        params_list.append(dict(params=module.parameters(), lr=args.base_lr))

    for module in modules_new:
        if args.finetune:
            params_list.append(dict(params=module.parameters(), lr=args.base_lr))
        else:
            params_list.append(dict(params=module.parameters(), lr=args.base_lr * 10))
    args.index_split = 5
    optimizer = torch.optim.SGD(params_list, lr=args.base_lr, momentum=args.momentum, weight_decay=args.weight_decay)
    return optimizer


def get_rank_to_dataset_map(args):
    """
        Obtain a mapping from GPU rank (index) to the name of the dataset residing on this GPU.

        Args:
        -   args

        Returns:
        -   rank_to_dataset_map
    """
    rank_to_dataset_map = {}
    for dataset, gpu_idxs in args.dataset_gpu_mapping.items():
        for gpu_idx in gpu_idxs:
            rank_to_dataset_map[gpu_idx] = dataset
    print('Rank to dataset map: ', rank_to_dataset_map)
    return rank_to_dataset_map


def main_worker(gpu, ngpus_per_node, argss):
    """
    Consider if a dataset has size 18,000 and is placed on a single GPU, of 4 gpus. 
    Batch size 32. In this case, len(train_data) = 18,000 but len(train_loader) = 2250
    Because effective batch size is 8.

    Consider if a dataset has size 118287. If placed on 2/4 gpus with batch size 32.
    In this case, len(train_data) = 118287 and len(train_loader) = 7393.
    """

    # with open('test_3.txt', 'a') as f:
    #     f.write('test')
    #     f.close()
    global args
    args = argss

    from ccsa.ccsa_data import CCSA_Data
    from util import dataset
    from taxonomy.utils_flat import TaxonomyConverter
    from multiobjective_opt.dist_mgda_utils import scale_loss_and_gradients
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
    from tensorboardX import SummaryWriter
    from util.dataset_config import infos

    from util import config
    from util.verification_utils import verify_architecture
    from util.avg_meter import AverageMeter, SegmentationAverageMeter
    from util.util import poly_learning_rate

    # with open('test_mainworker.txt', 'a') as f:
    #     f.write('test\t')
    #     f.close()
# os.sleep
    # time.sleep(30)
    if args.sync_bn:
        if args.multiprocessing_distributed:
            # BatchNorm = torch.nn.SyncBatchNorm
            BatchNorm = apex.parallel.SyncBatchNorm
        else:
            from lib.sync_bn.modules import BatchNorm2d
            BatchNorm = BatchNorm2d
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

    if True:
        global logger, writer
        logger = get_logger()
        writer = SummaryWriter(args.save_path)
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

    domain_idx_map = {
        'coco-panoptic-v1-qvga': 0,
        'mapillary_vistas_comm-qvga': 1,
        'ade20k-v1-qvga': 2
    }

    train_transform_dict = {}
    for dname, domain_idx in domain_idx_map.items():
        train_transform_dict[domain_idx] = get_train_transform_list(args, split='train', dataset_name=dname)

    # FLATMIX ADDITION
    train_data = CCSA_Data(split='train', data_roots=args.data_root, data_lists=args.train_list, transform_dict=train_transform_dict)
    
    from util.txt_utils import read_txt_file
    num_examples = len(train_data)

    num_examples_total = args.num_examples
    args.epochs = math.ceil(num_examples_total / num_examples)
    args.max_iters = math.floor(num_examples_total / (args.batch_size * args.ngpus_per_node))

    # avoid too frequent saving to waste time, on small datasets
    if args.epochs > 200:
        args.save_freq = args.epochs // 100

    logger.info(f'Train data has len {len(train_data)} on {args.rank}')
    if args.distributed:

        train_sampler = torch.utils.data.distributed.DistributedSampler(train_data, num_replicas=args.ngpus_per_node, rank=args.rank)
        logger.info(f"rank: {args.rank}, actual_replica: {train_sampler.num_replicas}, length of sampler, {len(train_sampler)}")

    else:
        train_sampler = None
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=(train_sampler is None), num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True)
    logger.info(f'Train loader has len {len(train_loader)} on {args.rank}')

    if args.evaluate:
        val_transform = get_train_transform_list(args, split='val')
        # val_transform = transform.Compose(val_transform_list)
        val_data = dataset.SemData(split='val', data_root=args.data_root, data_list=args.val_list, transform=val_transform)
        if args.distributed:
            val_sampler = torch.utils.data.distributed.DistributedSampler(val_data)
        else:
            val_sampler = None
        val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.batch_size_val, shuffle=False, num_workers=args.workers, pin_memory=True, sampler=val_sampler)

    for epoch in range(args.start_epoch, args.epochs):
        epoch_log = epoch + 1
        logger.info(f'New epoch {epoch_log} starts on rank {args.rank}')

        if args.distributed:
            train_sampler.set_epoch(epoch)
        print(f'On training epoch {epoch} in GPU {args.rank}')
        loss_train, mIoU_train, mAcc_train, allAcc_train = train(train_loader, model, optimizer, epoch)
        # if main_process():
        #     writer.add_scalar('loss_train', loss_train, epoch_log)
        #     writer.add_scalar('mIoU_train', mIoU_train, epoch_log)
        #     writer.add_scalar('mAcc_train', mAcc_train, epoch_log)
        #     writer.add_scalar('allAcc_train', allAcc_train, epoch_log)

        if ((epoch_log % args.save_freq == 0)) and main_process():
            filename = args.save_path + '/train_epoch_' + str(epoch_log) + '.pth'
            logger.info('Saving checkpoint to: ' + filename)
            torch.save({'epoch': epoch_log, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(), 
                'current_iter': (epoch + 1) * len(train_loader), 'max_iter': args.max_iters}, filename)
            if epoch_log / args.save_freq > 2:
                # if (epoch_log - 3) % 10 != 0:
                if not args.finetune: 
                    deletename = args.save_path + '/train_epoch_' + str(epoch_log - args.save_freq * 2) + '.pth'
                    os.remove(deletename)

        if (epoch == args.epochs - 1) and main_process():
            filename = args.save_path + '/train_epoch_final.pth'
            logger.info('Saving checkpoint to: ' + filename)
            torch.save({'epoch': epoch_log, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(), 
                'current_iter': (epoch + 1) * len(train_loader), 'max_iter': args.max_iters}, filename)
            exit()



        # if args.evaluate:
        #     loss_val, mIoU_val, mAcc_val, allAcc_val = validate(val_loader, model, criterion)
        #     if main_process():
        #         writer.add_scalar('loss_val', loss_val, epoch_log)
        #         writer.add_scalar('mIoU_val', mIoU_val, epoch_log)
        #         writer.add_scalar('mAcc_val', mAcc_val, epoch_log)
        #         writer.add_scalar('allAcc_val', allAcc_val, epoch_log)






def train(train_loader, model, optimizer, epoch):
    """
    No MGDA -- whole iteration takes 0.31 sec.
    0.24 sec to run typical backward pass (with no MGDA)

    With MGDA -- whole iteration takes 1.10 sec.
    1.05 sec to run backward pass w/ MGDA subroutine -- scale_loss_and_gradients() in every iteration.

    TODO: Profile which part of Frank-Wolfe is slow

    """

    from util.avg_meter import AverageMeter, SegmentationAverageMeter
    from util.util import poly_learning_rate

    import torch.distributed as dist
    from multiobjective_opt.dist_mgda_utils import scale_loss_and_gradients



    import torch, os, math, time


    batch_time = AverageMeter()
    data_time = AverageMeter()
    main_loss_meter = AverageMeter()
    aux_loss_meter = AverageMeter()
    loss_meter = AverageMeter()
    sam = SegmentationAverageMeter()

    model.train()
    # set bn to be eval() and see the norm
    # def set_bn_eval(m):
    #     classname = m.__class__.__name__
    #     if classname.find('BatchNorm') != -1:
    #         m.eval()
    # model.apply(set_bn_eval)
    end = time.time()
    max_iter = args.max_iters

    for i, (input, target, batch_domain_idxs) in enumerate(train_loader):
        # pass
        # if main_process():
        data_time.update(time.time() - end)
        if args.zoom_factor != 8:
            h = int((target.size()[1] - 1) / 8 * args.zoom_factor + 1)
            w = int((target.size()[2] - 1) / 8 * args.zoom_factor + 1)
            # 'nearest' mode doesn't support align_corners mode and 'bilinear' mode is fine for downsampling
            target = F.interpolate(target.unsqueeze(1).float(), size=(h, w), mode='bilinear', align_corners=True).squeeze(1).long()
        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        batch_domain_idxs = batch_domain_idxs.cuda(non_blocking=True)

        if args.use_mgda:
            output, loss, main_loss, aux_loss, scales = forward_backward_mgda(input, target, model, optimizer, args)
        else:
            #print('Batch domain idxs: ', batch_domain_idxs.shape, batch_domain_idxs.device, batch_domain_idxs)
            output, loss, main_loss, aux_loss = forward_backward_full_sync(input, target, model, optimizer, args, batch_domain_idxs)

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

        current_iter = epoch * len(train_loader) + i + 1 + args.resume_iter
        current_lr = poly_learning_rate(args.base_lr, current_iter, max_iter, power=args.power)
        # current_lr = 0
        for index in range(0, args.index_split):
            optimizer.param_groups[index]['lr'] = current_lr
        for index in range(args.index_split, len(optimizer.param_groups)):
            if args.finetune:
                optimizer.param_groups[index]['lr'] = current_lr 
            else:
                optimizer.param_groups[index]['lr'] = current_lr * 10

        remain_iter = max_iter - current_iter
        remain_time = remain_iter * batch_time.avg
        t_m, t_s = divmod(remain_time, 60)
        t_h, t_m = divmod(t_m, 60)
        remain_time = '{:02d}:{:02d}:{:02d}'.format(int(t_h), int(t_m), int(t_s))

        if (i + 1) % args.print_freq == 0 and main_process():
        # if True:
            logger.info('Epoch: [{}/{}][{}/{}] '
                        'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                        'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                        'Remain {remain_time} '
                        'MainLoss {main_loss_meter.val:.4f} '
                        'AuxLoss {aux_loss_meter.val:.4f} '
                        'Loss {loss_meter.val:.4f} '
                        'Accuracy {accuracy:.4f}.'.format(epoch+1, args.epochs, i + 1, len(train_loader),
                                                          batch_time=batch_time,
                                                          data_time=data_time,
                                                          remain_time=remain_time,
                                                          main_loss_meter=main_loss_meter,
                                                          aux_loss_meter=aux_loss_meter,
                                                          loss_meter=loss_meter,
                                                          accuracy=sam.accuracy) + f'current_iter: {current_iter}' + f' rank: {args.rank} ')
            if args.use_mgda and main_process():
                # Scales identical in each process, so print out only in main process.
                scales_str = [f'{d}: {scale:.2f}' for d,scale in scales.items()]
                scales_str = ' , '.join(scales_str)
                logger.info(f'Scales: {scales_str}')

        if main_process() and current_iter == max_iter - 5: # early exit to prevent iter number not matching between gpus
            break
        # if main_process():
        #     writer.add_scalar('loss_train_batch', main_loss_meter.val, current_iter)
        #     writer.add_scalar('mIoU_train_batch', np.mean(intersection / (union + 1e-10)), current_iter)
        #     writer.add_scalar('mAcc_train_batch', np.mean(intersection / (target + 1e-10)), current_iter)
        #     writer.add_scalar('allAcc_train_batch', accuracy, current_iter)

    iou_class, accuracy_class, mIoU, mAcc, allAcc = sam.get_metrics()
    # if main_process():
    logger.info('Train result at epoch [{}/{}]: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(epoch+1, args.epochs, mIoU, mAcc, allAcc))
    return main_loss_meter.avg, mIoU, mAcc, allAcc


def forward_backward_full_sync(input, target, model, optimizer, args, batch_domain_idxs):
    """
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
    output, main_loss, aux_loss = model(input, target, batch_domain_idxs)
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
    

def forward_backward_mgda(input, target, model, optimizer, args):
    from multiobjective_opt.dist_mgda_utils import scale_loss_and_gradients
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
    with model.no_sync():
        output, main_loss, aux_loss = model(input, target)
        loss = main_loss + args.aux_weight * aux_loss
        loss, scales = scale_loss_and_gradients(loss, optimizer, model, args)
        
    return output, loss, main_loss, aux_loss, scales




def validate(val_loader, model, criterion):
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
    print('main')


    main()