import os
import datetime
import random
import time
import cv2
import numpy as np
import logging
import argparse
import math
from visdom import Visdom
import os.path as osp

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
# from thop import profile

# from tensorboardX import SummaryWriter
from model import MetaDriver

# from util import dataset
from util import dataset_sal as dataset
import pandas as pd
from pathlib import Path
from util import transform, transform_tri, config
from util.util import AverageMeter, poly_learning_rate, intersectionAndUnionGPU, get_model_para_number, setup_seed, get_logger, get_save_path, \
                                    is_same_model, fix_bn, sum_list, check_makedirs


# ==============================================================================
# Test metrics
'''
Function:
Test metrics during testing, torch version.
This is the final test calculation.
        ------------------------  
        [↓] kldiv: smaller is better
        [↑] sim: bigger is better
        [↑] cc: bigger is better
'''

def similarity(s_map, gt):
    s_map = s_map / (np.sum(s_map) + 1e-7)
    gt = gt / (np.sum(gt) + 1e-7)
    return np.sum(np.minimum(s_map, gt))


def cc(s_map, gt):
    a = (s_map - np.mean(s_map))/(np.std(s_map) + 1e-7)
    b = (gt - np.mean(gt))/(np.std(gt) + 1e-7)
    r = (a*b).sum() / np.sqrt((a*a).sum() * (b*b).sum() + 1e-7)
    return r


def kldiv(s_map, gt):
    s_map = s_map / (np.sum(s_map) * 1.0)
    gt = gt / (np.sum(gt) * 1.0)
    eps = 2.2204e-16
    res = np.sum(gt * np.log(eps + gt / (s_map + eps)))
    return res

# ==============================================================================



cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)
val_manual_seed = 123
val_num = 10
setup_seed(val_manual_seed, False)
seed_array = np.random.randint(0,1000,val_num)    # seed->[0,999]


def get_parser():
    parser = argparse.ArgumentParser(description='PyTorch Few-Shot Semantic Segmentation')
    parser.add_argument('--arch', type=str, default='MetaDriver')
    parser.add_argument('--viz', action='store_true', default=False)
    parser.add_argument('--config', type=str, default='config/coco/coco_split3_resnet50.yaml', help='config file') # coco/coco_split0_resnet50.yaml
    parser.add_argument('--opts', help='see config/ade20k/ade20k_pspnet50.yaml for all options', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    assert args.config is not None
    cfg = config.load_cfg_from_cfg_file(args.config)
    cfg = config.merge_cfg_from_args(cfg, args)
    if args.opts is not None:
        cfg = config.merge_cfg_from_list(cfg, args.opts)
    return cfg


def get_model(args):
    model = eval(args.arch).OneModel(args, cls_type='Base')
    optimizer = model.get_optim(model, args, LR=args.base_lr)

    model = model.cuda()

    # Resume
    get_save_path(args)
    print('Model Loading', args.snapshot_path)
    
    check_makedirs(args.snapshot_path)
    check_makedirs(args.result_path)
    
    if args.weight:
        weight_path = osp.join(args.snapshot_path, args.weight)
        if os.path.isfile(weight_path):
            logger.info("=> loading checkpoint '{}'".format(weight_path))
            checkpoint = torch.load(weight_path, map_location=torch.device('cpu'))
            args.start_epoch = checkpoint['epoch']
            new_param = checkpoint['state_dict']
            try: 
                model.load_state_dict(new_param)
            except RuntimeError:                   # 1GPU loads mGPU model
                for key in list(new_param.keys()):
                    new_param[key[7:]] = new_param.pop(key)
                model.load_state_dict(new_param)
            optimizer.load_state_dict(checkpoint['optimizer'])
            logger.info("=> loaded checkpoint '{}' (epoch {})".format(weight_path, checkpoint['epoch']))
        else:
            logger.info("=> no checkpoint found at '{}'".format(weight_path))
    else:
        logger.info("=> loading default weight")
        weight = 'HDMNet_best.pth'
        # weight = 'HDMNet_epoch_4.pth'
        weight_path = osp.join(args.snapshot_path, weight)

        checkpoint = torch.load(weight_path)
        model.load_state_dict(checkpoint['state_dict'])
        logger.info("=> loaded weight '{}'".format(weight_path))


    # Get model para.
    # total_number, learnable_number = get_model_para_number(model)
    # print('Number of Parameters: %d' % (total_number))
    # print('Number of Learnable Parameters: %d' % (learnable_number))

    time.sleep(5)
    return model, optimizer

def main():
    global args, logger, writer
    args = get_parser()
    logger = get_logger()
    args.distributed = True if torch.cuda.device_count() > 1 else False
    print(args)

    if args.manual_seed is not None:
        setup_seed(args.manual_seed, args.seed_deterministic)

    # assert args.classes > 1
    # assert args.zoom_factor in [1, 2, 4, 8]
    # assert (args.train_h - 1) % 8 == 0 and (args.train_w - 1) % 8 == 0
    
    logger.info("=> creating model ...")
    model, optimizer = get_model(args)
    logger.info(model)

# ----------------------  DATASET  ----------------------
    # value_scale = 255
    # mean = [0.485, 0.456, 0.406]
    # mean = [item * value_scale for item in mean]
    # std = [0.229, 0.224, 0.225]
    # std = [item * value_scale for item in std]


    value_scale = 1
    mean = [0., 0., 0.]
    mean = [item * value_scale for item in mean]
    std = [0.229, 0.224, 0.225]
    std = [item * value_scale for item in std]

    # Val
    if args.evaluate:
        if args.resized_val:
            val_transform = transform.Compose([
                transform.Resize(size=args.val_size),
                transform.ToTensor()])
            val_transform_tri = transform_tri.Compose([
                transform_tri.Resize(size=args.val_size),
                transform_tri.ToTensor()])
        else:
            val_transform = transform.Compose([
                transform.test_Resize(size=args.val_size),
                transform.ToTensor()])
            val_transform_tri = transform_tri.Compose([
                transform_tri.test_Resize(size=args.val_size),
                transform_tri.ToTensor()])

       
        # ----------------------  VAL  ----------------------
        BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
        data_list = os.path.join(BASE_DIR, args.val_list)

        if args.data_type in ['metadada', 'metapsad']:

            test_base = args.test_base
            # test novel class
            val_data = dataset.SemData(split=args.split, shot=args.shot, data_root=args.data_root, base_data_root=args.base_data_root, data_list=data_list, \
                                    transform=val_transform, transform_tri=val_transform_tri, mode='val', \
                                    data_type=args.data_type, test_base=test_base, data_set=args.data_set, use_split_coco=args.use_split_coco)                                   
            val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.batch_size_val, shuffle=False, num_workers=args.workers, pin_memory=False, sampler=None)
            novel_metrics_info, novel_metrics = validate(val_loader, model, test_base)

            test_base = True
            # test novel class
            val_data = dataset.SemData(split=args.split, shot=args.shot, data_root=args.data_root, base_data_root=args.base_data_root, data_list=data_list, \
                                    transform=val_transform, transform_tri=val_transform_tri, mode='val', \
                                    data_type=args.data_type, test_base=test_base, data_set=args.data_set, use_split_coco=args.use_split_coco)                                   
            val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.batch_size_val, shuffle=False, num_workers=args.workers, pin_memory=False, sampler=None)
            base_metrics_info, base_metrics = validate(val_loader, model, test_base)

            print('='*70)
            print('[Final]')
            print(novel_metrics_info)
            print(base_metrics_info)
            save_metrics_to_excel(base_metrics, novel_metrics)



def save_metrics_to_excel(base_metrics, novel_metrics, excel_path='metrics.xlsx'):
    """
    将 base 和 novel 的指标分别保存到 Excel 的两个 sheet 中。

    Parameters:
        base_metrics (list of dict): Base 类指标记录，每个 dict 包含 Iter, BatchIndex, CC, SIM, KLD。
        novel_metrics (list of dict): Novel 类指标记录。
        excel_path (str): 保存的 Excel 文件路径。
    """
    excel_path = osp.join(args.snapshot_path, excel_path)
    df_base = pd.DataFrame(base_metrics)
    df_novel = pd.DataFrame(novel_metrics)

    Path(excel_path).parent.mkdir(parents=True, exist_ok=True)

    # 写入 Excel
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        df_base.to_excel(writer, sheet_name='Base', index=False)
        df_novel.to_excel(writer, sheet_name='Novel', index=False)
    

def validate(valid_loader, model, test_base):
    if test_base:
        print('===> Tesing base class...')
        cls = 'Base '
    else:
        print('===> Tesing novel class...')
        cls = 'Novel'

    losses = AverageMeter()
    metrics_list = []
    model.eval()

    start = time.time()
    metrics = [0, 0, 0]
    b = valid_loader.batch_size
    with torch.no_grad():
        for i, (input, target, target_b, s_input, s_mask, subcls, raw_query_label) in enumerate(valid_loader):

            s_input = s_input.cuda(non_blocking=True)
            s_mask = s_mask.cuda(non_blocking=True)
            input = input.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            target_b = target_b.cuda(non_blocking=True)

            # compute output
            output, _, _ = model(s_x=s_input, s_y=s_mask, x=input, y_m=target, y_b=target_b,
                                                        cat_idx=subcls)

            # valid metrics printing
            output = output.squeeze(1)
            target = target.squeeze(1)

            for index in range(output.shape[0]):
                temp_out = output[index].detach().cpu().numpy()
                temp_tar = target[index].detach().cpu().numpy()

                temp_cc = cc(temp_out, temp_tar)
                temp_sim = similarity(temp_out, temp_tar)
                temp_kld = kldiv(temp_out, temp_tar)
                metrics[0] += temp_cc
                metrics[1] += temp_sim
                metrics[2] += temp_kld

                record = {
                    'Iter': i * b + (index + 1),
                    'CC': temp_cc,
                    'SIM': temp_sim,
                    'KLD': temp_kld
                }
                metrics_list.append(record)
                msg = 'Testing Iter {:03d} || CC {:4f}  SIM {:4f}  KLD {:4f} in {:.3f}s'.format(
                    i + 1,
                    metrics[0] / (i * b + (index + 1)),
                    metrics[1] / (i * b + (index + 1)),
                    metrics[2] / (i * b + (index + 1)),
                    time.time() - start)

                if i == (len(valid_loader)-1) and index == (output.shape[0]-1) :
                    record = {
                        'Iter': 'final',
                        'CC': metrics[0] / (i * b + (index + 1)),
                        'SIM': metrics[1] / (i * b + (index + 1)),
                        'KLD': metrics[2] / (i * b + (index + 1)),
                    }
                    metrics_list.append(record)

            start = time.time()

            interval = 10
            if (i + 1) % interval == 0:
                # logger.info(msg)
                print(msg)
    res_msg = f'{cls} Final: {msg}'
    print(res_msg)
    return res_msg, metrics_list


if __name__ == '__main__':
    main()
