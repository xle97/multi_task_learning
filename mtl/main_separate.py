import os
import torch
import time
import shutil
import argparse
import torchvision.models as models
from loguru import logger
from utils import  AverageMeter, ProgressMeter, Summary
from torch.optim.lr_scheduler import StepLR
from model import MultiTaskModel
from loss import SingleLossWrapper
from dataset import SeparateDataset

# os.environ['CUDA_VISIBLE_DEVICES'] = "1"

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

best_acc1 =0

def parse_args():
    parser = argparse.ArgumentParser(description="Multi task classification")
    parser.add_argument('data', metavar='DIR', nargs='?', default='',
                    help='path to dataset (default: imagenet)')
    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet34',
                        choices=model_names,
                        help='model architecture: ' +
                            ' | '.join(model_names) +
                            ' (default: resnet34)')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=90, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch-size', default=256, type=int,
                        metavar='N',
                        help='mini-batch size (default: 256), this is the total '
                            'batch size of all GPUs on the current node when '
                            'using Data Parallel or Distributed Data Parallel')
    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('-p', '--print-freq', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                        help='use pre-trained model')
    # parser.add_argument('--train_head', action='store_true',
    #                     help='train only cls head')
    parser.add_argument('--head', type=str,
                        help='train which head')
    parser.add_argument('--load_from', type=str, default="./resnet34-333f7ec4.pth",
                        help='train which head')
    parser.add_argument('--save_dir',default="work_dir",
                        help='path to save model')
    parser.add_argument('--world-size', default=-1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=-1, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str,
                        help='distributed backend')
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--gpu', default=None, type=int,
                        help='GPU id to use.')
    
    args = parser.parse_args()
    return args


def setup_logger():
    loguru_format = "<level>{time:%m/%d/%Y %H:%M:%S}</level> - <level>{level: <8}</level> - <level>{message}</level>"
    logger.add("./logs/{time}.log", format= loguru_format)

def accuracy(preds, target):
    def top1(pred, target):
        with torch.no_grad():
            batch_size = target.size(0)

            _, pred = pred.topk(1, 1, True, True)
            pred = pred.t()
            # correct = pred.eq(target.view(1, -1).expand_as(pred))
            ### 注意两个tensor的shape
            correct = pred.eq(target.expand_as(pred))
            
            correct_k = correct[:1].reshape(-1).float().sum(0, keepdim=True)
            res = correct_k.mul_(100.0 / batch_size)

            return res
    return top1(preds, target)

def save_checkpoint(state, is_best, save_dir, head, filename='checkpoint.pth'):
    filename = f'checkpoint_head{head}.pth'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir,mode=0o777, exist_ok=True)
    torch.save(state, os.path.join(save_dir,filename))
    if is_best:
        shutil.copyfile(os.path.join(save_dir, filename), save_dir+f'/model_best_head{head}.pth')
        logger.info('more excellent model appears!')
        logger.info('#'*80)

def main(args):
    global best_acc1
    setup_logger()
    args.distributed = False

    if args.gpu is not None:
        logger.info("Use GPU: {} for training".format(args.gpu))
        torch.cuda.set_device(args.gpu)
        device = torch.device("cuda")
    ### 加载模型
    model = MultiTaskModel(args.arch).cuda()

    ### 训练15个epoch的分类头
    if args.head:
        logger.info("Train only cls_head")
        criterion = SingleLossWrapper().to(device)
        train_head = f"fc{args.head}"
        for name, param in model.named_parameters():
            if train_head not in name:
                param.requires_grad = False
        # for name, param in model.named_parameters():
        #     print(name, ":",param.requires_grad)
        # exit(0)
        optimizer = torch.optim.SGD(filter(lambda p : p.requires_grad, model.parameters()), 
                                    args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)
        
        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
        scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
    else:
        logger.info("Full model Training ")
        criterion = SingleLossWrapper().to(device)
        
        optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)
        
        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
        scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
    ## add dynamic weight of loss
    # optimizer.add_param_group({"params": criterion.log_vars})
    
    if args.resume:
        if os.path.isfile(args.resume):
            logger.info("=> loading checkpoint '{}'".format(args.resume))
            # if args.gpu is None:
            #     checkpoint = torch.load(args.resume)
            # elif torch.cuda.is_available():
            #     # Map model to be loaded to specified single gpu.
            #     loc = 'cuda:{}'.format(args.gpu)
                # checkpoint = torch.load(args.resume, map_location=loc)
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            # if args.gpu is not None:
            #     # best_acc1 may be from a checkpoint from a different GPU
            #     best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            logger.info("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            logger.info("=> no checkpoint found at '{}'".format(args.resume))
            raise Exception(f"no file found : {args.resume}")
    else:
        ### 从本地加载预训练权重
        ### 只加载backbone的权重
        if "resnet34" in args.load_from:
            model_dict = model.state_dict()
            pretrained_dict = torch.load(args.load_from)
            logger.info(f"load ckpt: {args.load_from}")
            pretrained_dict = {'encoder.'+k: v for k, v in pretrained_dict.items() if ('encoder.'+k in model_dict and 'fc' not in k)}
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)
        else:

            model_dict = model.state_dict()
            pretrained_dict = torch.load(args.load_from)['state_dict']
            model.load_state_dict(pretrained_dict)
            logger.info(f"load ckpt: {args.load_from}")
    if args.gpu is not None:
        
        model = model.cuda(args.gpu)
        # model = model.to(device)
        
    else:
        logger.warning('using cpu, exit')
        exit(0)
    
    train_pathlist = []
    train_labellist = []
    val_pathlist = []
    val_labellist = []
    with open(os.path.join(args.data, f'train_{args.head}.txt'), 'r') as fr:
        for line in fr:
            path, l1=line.strip().split()
            train_pathlist.append(path)
            train_labellist.append(l1)
    with open(os.path.join(args.data, f'val_{args.head}.txt'), 'r') as fr:
        for line in fr:
            path, l1=line.strip().split()
            val_pathlist.append(path)
            val_labellist.append(l1)
    
        
    train_dataset = SeparateDataset(train_pathlist, train_labellist, is_train=True)
    val_dataset = SeparateDataset(val_pathlist, val_labellist)

    train_sampler = None
    val_sampler = None
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, sampler=val_sampler)

    

    
    logger.info(f"start training head{args.head}")
    for epoch in range(args.start_epoch, args.epochs):
        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, device, args)

        # evaluate on validation set
        acc = validate(val_loader, model, criterion, args)
        scheduler.step()
        
        # remember best acc@1 and save checkpoint
        is_best = acc > best_acc1
        best_acc1 = max(acc, best_acc1)

        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_acc1': best_acc1,
            'optimizer' : optimizer.state_dict(),
            'scheduler' : scheduler.state_dict()
        }, is_best,args.save_dir, args.head)

def validate(val_loader, model, criterion, args):
    def run_validate(loader, base_progress=0):
        with torch.no_grad():
            end = time.time()
            for i, (images, target) in enumerate(loader):
                i = base_progress + i
                if args.gpu is not None and torch.cuda.is_available():
                    images = images.cuda(args.gpu, non_blocking=True)
                if torch.cuda.is_available():
                    target = target.cuda(args.gpu, non_blocking=True)

                # compute output
                output = model(images)[int(args.head)-1]
                loss = criterion(output, target)

                # measure accuracy and record loss
                # maxk = min(args.num_classes, 5)
                top1 = accuracy(output, target)
                losses.update(loss.item(), images.size(0))
                acc.update(top1.item(), images.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if i % args.print_freq == 0:
                    progress.display(i + 1)

    batch_time = AverageMeter('Time', ':6.3f', Summary.NONE)
    losses = AverageMeter('Loss', ':.4e', Summary.NONE)
    acc = AverageMeter('Acc@top1', ':6.2f', Summary.AVERAGE)
    progress = ProgressMeter(
        len(val_loader) ,
        [batch_time, losses, acc],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    run_validate(val_loader)
    progress.display_summary()

    return acc.avg / 3
    

def train(train_loader, model, criterion, optimizer, epoch, device, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    acc = AverageMeter('Acc@top1', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, acc],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()
    # global count1
    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # move data to the same device as model
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        # compute output
        output = model(images)[int(args.head)-1]
        loss = criterion(output, target)

        # measure accuracy and record loss
        top1 = accuracy(output, target)
        losses.update(loss.item(), images.size(0))
        # top1.update(acc1[0], images.size(0))
        acc.update(top1.item(), images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i + 1)



if __name__ == "__main__":
    args = parse_args()
    main(args)