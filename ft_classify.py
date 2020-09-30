"""Finetune 3D CNN."""
import os
import argparse
import time
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import torch.optim as optim
from tensorboardX import SummaryWriter

from lib.utils import AverageMeter

from datasets.ucf101 import UCF101Dataset
from datasets.hmdb51 import HMDB51Dataset
from models.c3d import C3D
from models.r3d import R3DNet
from models.r21d import R2Plus1DNet

import ast



def calculate_accuracy(outputs, targets):
    batch_size = targets.size(0)

    _, pred = outputs.topk(1, 1, True)
    pred = pred.t()
    correct = pred.eq(targets.view(1, -1))
    n_correct_elems = correct.float().sum().data.item()

    return n_correct_elems / batch_size


def diff(x):
    shift_x = torch.roll(x, 1, 2)
    return x - shift_x # without rescaling
    #return ((x - shift_x) + 1) / 2


def load_pretrained_weights(ckpt_path):
    """load pretrained weights and adjust params name."""
    adjusted_weights = {}
    pretrained_weights = torch.load(ckpt_path)
    for name, params in pretrained_weights.items():
        if 'base_network' in name:
            name = name[name.find('.')+1:]
            adjusted_weights[name] = params
            #print('Pretrained weight name: [{}]'.format(name))
    return adjusted_weights


def train(args, model, criterion, optimizer, train_dataloader, epoch):
    torch.set_grad_enabled(True)
    model.train()

    losses = AverageMeter()
    accuracies = AverageMeter()
    for i, data in enumerate(train_dataloader, 1):
        # get inputs
        sampled_clips, u_clips, v_clips, targets, _ = data
        if args.modality == 'u':
            inputs = u_clips
        elif args.modality == 'v':
            inputs = v_clips
        else: # rgb and res
            inputs = sampled_clips
        inputs = inputs.cuda()
        targets = targets.cuda()
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward and backward
        if args.modality == 'res':
            outputs = model(diff(inputs))
        else:
            outputs = model(inputs)

        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        acc = calculate_accuracy(outputs, targets)
        losses.update(loss.data.item(), inputs.size(0))
        accuracies.update(acc, inputs.size(0))

        print('Train epoch: [{0:3d}/{1:3d}][{2:4d}/{3:4d}]\t'
              'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
              'Acc {acc.val:.3f} ({acc.avg:.3f})\t'
              'lr: {lr}'.format(
                epoch, args.epochs, i + 1, len(train_dataloader),
                loss=losses, acc=accuracies,
                lr=optimizer.param_groups[0]['lr']), end='\r')
    print('')


def validate(args, model, criterion, val_dataloader, epoch):
    torch.set_grad_enabled(False)
    model.eval()
    
    losses = AverageMeter()
    accuracies = AverageMeter()
    for i, data in enumerate(val_dataloader):
        # get inputs
        sampled_clips, u_clips, v_clips, targets, _ = data
        if args.modality == 'u':
            inputs = u_clips
        elif args.modality == 'v':
            inputs = v_clips
        else: # rgb and res
            inputs = sampled_clips
        inputs = inputs.cuda()
        targets = targets.cuda()
        # forward
        if args.modality == 'res':
            outputs = model(diff(inputs))
        else:
            outputs = model(inputs)

        loss = criterion(outputs, targets)
        # compute loss and acc
        acc = calculate_accuracy(outputs, targets)
        losses.update(loss.data.item(), inputs.size(0))
        accuracies.update(acc, inputs.size(0))

        print('Val epoch:   [{0:3d}/{1:3d}][{2:4d}/{3:4d}]\t'
              'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
              'Acc {acc.val:.3f} ({acc.avg:.3f})\t'
              'lr: {lr}'.format(
                epoch, args.epochs, i + 1, len(val_dataloader),
                loss=losses, acc=accuracies,
                lr=optimizer.param_groups[0]['lr']), end='\r')
    print('')
    return losses.avg


def test(args, model, criterion, test_dataloader):
    torch.set_grad_enabled(False)
    model.eval()

    accuracies = AverageMeter()

    if args.modality == 'res':
        print("[Warning]: using residual frames as input")

    total_loss = 0.0
    for i, data in enumerate(test_dataloader, 1):
        # get inputs
        rgb_clips, u_clips, v_clips, targets, _ = data
        if args.modality == 'u':
            sampled_clips = u_clips
        elif args.modality == 'v':
            sampled_clips = v_clips
        else: # rgb and res
            sampled_clips = rgb_clips
        sampled_clips = sampled_clips.cuda()
        targets = targets.cuda()
        outputs = []
        for clips in sampled_clips:
            inputs = clips.cuda()
            # forward
            if args.modality == 'res':
                o = model(diff(inputs))
            else:
                o = model(inputs)
            o = torch.mean(o, dim=0)
            outputs.append(o)
        outputs = torch.stack(outputs)
        loss = criterion(outputs, targets)
        # compute loss and acc
        total_loss += loss.item()
        acc = calculate_accuracy(outputs, targets)
        accuracies.update(acc, inputs.size(0))
        print('Test: [{}/{}], {acc.val:.3f} ({acc.avg:.3f})'.format(i, len(test_dataloader), acc=accuracies), end='\r')
    avg_loss = total_loss / len(test_dataloader)
    print('\n[TEST] loss: {:.3f}, acc: {:.3f}'.format(avg_loss, accuracies.avg))
    return avg_loss


def parse_args():
    parser = argparse.ArgumentParser(description='Finetune 3D CNN from pretrained weights')
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--cl', type=int, default=16, help='clip length')
    parser.add_argument('--model', type=str, default='r3d', help='c3d/r3d/r21d')
    parser.add_argument('--dataset', type=str, default='ucf101', help='ucf101/hmdb51')
    parser.add_argument('--split', type=str, default='1', help='dataset split')
    parser.add_argument('--gpu', type=int, default=0, help='GPU id')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--ft_lr', type=float, default=1e-3, help='finetune learning rate')
    parser.add_argument('--momentum', type=float, default=9e-1, help='momentum')
    parser.add_argument('--wd', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--model_dir', type=str, default='./ckpt/', help='path to save model')
    parser.add_argument('--ckpt', type=str, help='checkpoint path')
    parser.add_argument('--desp', type=str, help='additional description')
    parser.add_argument('--epochs', type=int, default=150, help='number of total epochs to run')
    parser.add_argument('--start-epoch', type=int, default=1, help='manual epoch number (useful on restarts)')
    parser.add_argument('--bs', type=int, default=16, help='mini-batch size')
    parser.add_argument('--workers', type=int, default=4, help='number of data loading workers')
    parser.add_argument('--seed', type=int, default=632, help='seed for initializing training.')
    parser.add_argument('--modality', default='res', type=str, help='modality from [rgb, res, u, v]') 
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    print(vars(args))

    # Uncomment to fix all parameters for reproducibility
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

    ########### model ##############
    if args.dataset == 'ucf101':
        class_num = 101
    elif args.dataset == 'hmdb51':
        class_num = 51

    if args.model == 'c3d':
        model = C3D(with_classifier=True, num_classes=class_num).cuda()
    elif args.model == 'r3d':
        model = R3DNet(layer_sizes=(1,1,1,1), with_classifier=True, num_classes=class_num).cuda()
    elif args.model == 'r21d':   
        model = R2Plus1DNet(layer_sizes=(1,1,1,1), with_classifier=True, num_classes=class_num).cuda()
    #pretrained_weights = load_pretrained_weights(args.ckpt)
    pretrained_weights = torch.load(args.ckpt)
    if args.mode == 'train':
        model.load_state_dict(pretrained_weights['model'], strict=False)
    else:
        #model.load_state_dict(pretrained_weights['model'], strict=True)
        model.load_state_dict(pretrained_weights, strict=True)


    if args.desp:
        exp_name = '{}_{}_cls_{}_{}'.format(args.model, args.modality, args.desp, time.strftime('%m%d'))
    else:
        exp_name = '{}_{}_cls_{}'.format(args.model, args.modality, time.strftime('%m%d'))
    print(exp_name)
    model_dir = os.path.join(args.model_dir, exp_name)
    if not os.path.isdir(model_dir) and args.mode == 'train':
        os.makedirs(model_dir)

    train_transforms = transforms.Compose([
        transforms.Resize((128, 171)),
        transforms.RandomCrop(112),
        transforms.ToTensor()
    ])
    test_transforms = transforms.Compose([
        transforms.Resize((128, 171)),
        transforms.CenterCrop(112),
        transforms.ToTensor()
    ])

    if args.dataset == 'ucf101':
        train_dataset = UCF101Dataset('data/ucf101', args.cl, args.split, True, train_transforms)
        test_dataset = UCF101Dataset('data/ucf101', args.cl, args.split, False, test_transforms)
        val_size = 800
    elif args.dataset == 'hmdb51':
        train_dataset = HMDB51Dataset('data/hmdb51', args.cl, args.split, True, train_transforms)
        test_dataset = HMDB51Dataset('data/hmdb51', args.cl, args.split, False, test_transforms)
        val_size = 400

    # split val for 800 videos
    train_dataset, val_dataset = random_split(train_dataset, (len(train_dataset)-val_size, val_size))
    print('TRAIN video number: {}, VAL video number: {}.'.format(len(train_dataset), len(val_dataset)))
    train_dataloader = DataLoader(train_dataset, batch_size=args.bs, shuffle=True,
                                num_workers=args.workers, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.bs, shuffle=False,
                                num_workers=args.workers, pin_memory=True)


    ### loss funciton, optimizer and scheduler ###
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD([
        {'params': [param for name, param in model.named_parameters() if 'linear' not in name and 'conv5' not in name and 'conv4' not in name]},
        {'params': [param for name, param in model.named_parameters() if 'linear' in name or 'conv5' in name or 'conv4' in name], 'lr': args.ft_lr}],
        lr=args.lr, momentum=args.momentum, weight_decay=args.wd)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', min_lr=1e-5, patience=50, factor=0.1)

    prev_best_val_loss = float('inf')
    prev_best_model_path = None
    if args.mode == 'train':
        for epoch in range(args.start_epoch, args.start_epoch+args.epochs):
            time_start = time.time()
            train(args, model, criterion, optimizer, train_dataloader, epoch)
            val_loss = validate(args, model, criterion, val_dataloader, epoch)
            print('Epoch time: {:.2f} s.'.format(time.time() - time_start))
            scheduler.step(val_loss)
            # save model every 20 epoches
            if epoch % 20 == 0:
                state = {'model': model.state_dict(),}
                torch.save(state, os.path.join(model_dir, 'model_{}.pt'.format(epoch)))
            # save model for the best val
            if val_loss < prev_best_val_loss:
                model_path = os.path.join(model_dir, 'best_model_{}.pt'.format(epoch))
                state = {'model': model.state_dict(),}
                torch.save(state, model_path)
                prev_best_val_loss = val_loss
                if prev_best_model_path:
                    os.remove(prev_best_model_path)
                prev_best_model_path = model_path

    print('start testing ...')
    if args.mode == 'train':
        model.load_state_dict(torch.load(prev_best_model_path)['model'])
    test_dataloader = DataLoader(test_dataset, batch_size=args.bs, shuffle=False,
                                num_workers=args.workers, pin_memory=True)
    test(args, model, criterion, test_dataloader)
    
