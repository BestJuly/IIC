import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.optim import lr_scheduler


import torchvision
import torchvision.transforms as transforms
import lib.custom_transforms as custom_transforms

import os
import argparse
import time

import models
import datasets
import math

import tensorboard_logger as tb_logger

from lib.NCEAverage import NCEAverage, NCEAverage_ori
from lib.LinearAverage import LinearAverage
from lib.NCECriterion import NCECriterion, NCESoftmaxLoss
from lib.utils import AverageMeter#, adjust_learning_rate

from datasets.ucf101 import UCF101Dataset
from datasets.hmdb51 import HMDB51Dataset
from models.c3d import C3D
from models.r21d import R2Plus1DNet
from models.r3d import R3DNet

from torch.utils.data import DataLoader, random_split

from gen_neg import preprocess
import random
import numpy as np
import ast

def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=10, help='print frequency')
    parser.add_argument('--tb_freq', type=int, default=500, help='tb frequency')
    parser.add_argument('--save_freq', type=int, default=40, help='save frequency')
    parser.add_argument('--batch_size', type=int, default=16, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=8, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=240, help='number of training epochs')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.01, help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='120,160,200', help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')

    # resume path
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')

    # model definition
    parser.add_argument('--model', type=str, default='r3d', choices=['r3d', 'c3d', 'r21d'])
    parser.add_argument('--softmax', type=ast.literal_eval, default=True)
    parser.add_argument('--nce_k', type=int, default=1024)
    parser.add_argument('--nce_t', type=float, default=0.07)
    parser.add_argument('--nce_m', type=float, default=0.5)
    parser.add_argument('--feat_dim', type=int, default=512, help='dim of feat for inner product')

    # dataset
    parser.add_argument('--dataset', type=str, default='ucf101', choices=['ucf101', 'hmdb51'])

    # specify folder
    #parser.add_argument('--data_folder', type=str, default=None, help='path to data')
    parser.add_argument('--model_path', type=str, default='./ckpt/', help='path to save model')
    parser.add_argument('--tb_path', type=str, default='./logs/', help='path to tensorboard')

    # add new views
    parser.add_argument('--debug', type=ast.literal_eval, default=False)
    parser.add_argument('--modality', type=str, default='res', choices=['rgb', 'res', 'u', 'v'])
    parser.add_argument('--intra_neg', type=ast.literal_eval, default=True)
    parser.add_argument('--neg', type=str, default='repeat', choices=['repeat', 'shuffle'])
    #parser.add_argument('--desp', type=str)
    parser.add_argument('--seed', type=int, default=632)

    opt = parser.parse_args()

    if opt.intra_neg:
        print('[Warning] using intra-negative')
        opt.model_name = 'intraneg_{}_{}_{}'.format(opt.model, opt.modality, time.strftime('%m%d'))
    else:
        print('[Warning] using baseline')
        opt.model_name = '{}_{}_{}'.format(opt.model, opt.modality, time.strftime('%m%d'))

    opt.model_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.model_folder):
        os.makedirs(opt.model_folder)

    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)

    return opt


def set_model(args, n_data):
    # set the model
    if args.model == 'c3d':
        model = C3D(with_classifier=False)
    elif args.model == 'r3d':
        model = R3DNet(layer_sizes=(1,1,1,1), with_classifier=False)
    elif args.model == 'r21d':  
        model = R2Plus1DNet(layer_sizes=(1,1,1,1), with_classifier=False)

    if args.intra_neg:
        contrast = NCEAverage(args.feat_dim, n_data, args.nce_k, args.nce_t, args.nce_m, args.softmax)
    else:
        contrast = NCEAverage_ori(args.feat_dim, n_data, args.nce_k, args.nce_t, args.nce_m, args.softmax)

    criterion_1 = NCESoftmaxLoss() if args.softmax else NCECriterion(n_data)
    criterion_2 = NCESoftmaxLoss() if args.softmax else NCECriterion(n_data)

    # GPU mode
    model = model.cuda()
    contrast = contrast.cuda()
    criterion_1 = criterion_1.cuda()
    criterion_2 = criterion_2.cuda()
    cudnn.benchmark = True

    return model, contrast, criterion_1, criterion_2


def set_optimizer(args, model):
    # return optimizer
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=args.learning_rate,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    return optimizer


def diff(x):
    shift_x = torch.roll(x, 1, 2)
    return ((x - shift_x) + 1) / 2


def train(epoch, train_loader, model, contrast, criterion_1, criterion_2, optimizer, opt):
    """
    one epoch training
    """
    model.train()
    contrast.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    view1_loss_meter = AverageMeter()
    view2_loss_meter = AverageMeter()
    view1_prob_meter = AverageMeter()
    view2_prob_meter = AverageMeter()

    end = time.time()
    for idx, (inputs, u_inputs, v_inputs, _, index) in enumerate(train_loader):
        data_time.update(time.time() - end)

        bsz = inputs.size(0)
        inputs = inputs.float().cuda()
        u_inputs = u_inputs.float().cuda()
        v_inputs = v_inputs.float().cuda()
        index = index.cuda()

        # ===================forward=====================
        feat_1 = model(inputs) # view 1 is always RGB
        if opt.modality == 'res':
            feat_2 = model(diff(inputs))
        elif opt.modality == 'u':
            feat_2 = model(u_inputs)
        elif opt.modality == 'v':
            feat_2 = model(v_inputs)
        else:
            feat_2 = feat_1
        
        if not opt.intra_neg:
            out_1, out_2 = contrast(feat_1, feat_2, index)
        else:
            feat_neg = model(preprocess(inputs, opt.neg))
            out_1, out_2 = contrast(feat_1, feat_2, feat_neg, index)

        view1_loss = criterion_1(out_1)
        view2_loss = criterion_2(out_2)
        view1_prob = out_1[:, 0].mean()
        view2_prob = out_2[:, 0].mean()

        loss = view1_loss + view2_loss

        # ===================backward=====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ===================meters=====================
        losses.update(loss.item(), bsz)
        view1_loss_meter.update(view1_loss.item(), bsz)
        view1_prob_meter.update(view1_prob.item(), bsz)
        view2_loss_meter.update(view2_loss.item(), bsz)
        view2_prob_meter.update(view2_prob.item(), bsz)

        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if (idx + 1) % opt.print_freq == 0:
            print('Train: [{0}/{1}][{2}/{3}]\t'
                'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                'loss {loss.val:.3f} ({loss.avg:.3f})\t'
                '1_p {probs1.val:.3f} ({probs1.avg:.3f})\t'
                '2_p {probs2.val:.3f} ({probs2.avg:.3f})'.format(
                 epoch, opt.epochs, idx + 1, len(train_loader), batch_time=batch_time,
                 data_time=data_time, loss=losses, probs1=view1_prob_meter,
                 probs2=view2_prob_meter), end='\r')

    return view1_loss_meter.avg, view1_prob_meter.avg, view2_loss_meter.avg, view2_prob_meter.avg


def main():
    if not torch.cuda.is_available():
        raise 'Only support GPU mode'
    # parse the args
    args = parse_option()
    print(vars(args))

    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    ''' Old version
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    '''
    # Fix all parameters for reproducibility
    seed = args.seed
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    #'''
    
    print('[Warning] The training modalities are RGB and [{}]'.format(args.modality))

    # Data
    train_transforms = transforms.Compose([
                transforms.Resize((128, 171)),  # smaller edge to 128
                transforms.RandomCrop(112),
                transforms.ToTensor()
            ]) 
    if args.dataset == 'ucf101':
        trainset = UCF101Dataset('./data/ucf101/', transforms_=train_transforms)
    else:
        trainset = HMDB51Dataset('./data/hmdb51/', transforms_=train_transforms)

    train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True,
                                        num_workers=args.num_workers, pin_memory=True, drop_last=True)

    n_data = trainset.__len__()

    # set the model
    model, contrast, criterion_1, criterion_2 = set_model(args, n_data)

    # set the optimizer
    optimizer = set_optimizer(args, model)

    # optionally resume from a checkpoint
    args.start_epoch = 1
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location='cpu')
            args.start_epoch = checkpoint['epoch'] + 1
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            contrast.load_state_dict(checkpoint['contrast'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
            del checkpoint
            torch.cuda.empty_cache()
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    # tensorboard
    logger = tb_logger.Logger(logdir=args.tb_folder, flush_secs=2)

    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[45, 90, 125, 160], gamma=0.2)
    # routine
    for epoch in range(args.start_epoch, args.epochs + 1):
        time1 = time.time()
        view1_loss, view1_prob, view2_loss, view2_prob = train(epoch, train_loader, model, contrast, 
                                                            criterion_1, criterion_2, optimizer, args)
        time2 = time.time()
        print('\nepoch {}, total time {:.2f}'.format(epoch, time2 - time1))

        # tensorboard logger
        logger.log_value('view1_loss', view1_loss, epoch)
        logger.log_value('view1_prob', view1_prob, epoch)
        logger.log_value('view2_loss', view2_loss, epoch)
        logger.log_value('view2_prob', view2_prob, epoch)

        # save model
        if epoch % args.save_freq == 0:
            print('==> Saving...')
            state = {
                'opt': args,
                'model': model.state_dict(),
                'contrast': contrast.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
            }
            save_file = os.path.join(args.model_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            torch.save(state, save_file)
            # help release GPU memory
            del state

        torch.cuda.empty_cache()
        scheduler.step()
    
    print(args.model_name)


if __name__ == '__main__':
    main()
