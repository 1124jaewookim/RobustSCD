import datetime
import os
import time
import math
import torch
import torch.utils.data
from collections import OrderedDict
import numpy as np
import utils_ as utils
from dataset import dataset_dict
import loss
import torch.nn as nn
from loss import DiceLoss, Bicriterion
from model import model_dict
import torch.nn.functional as F
import dataset.transforms as T
import matplotlib.pyplot as plt
import random
import pickle

def draw_figure(args, target, domain1, domain2):
    assert args.train_dataset == 'VL_CMU_CD'

    title_font = {
        'fontsize': 14,
        'fontweight': 'bold'
    }

    fig = plt.figure(figsize=(10,10), constrained_layout = True)
    plt.plot(target, 'r--', label='Target (VL-CMU-CD)')
    plt.plot(domain1, 'g--', label='TSUNAMI')
    plt.plot(domain2, 'b--', label='PSCD')
    plt.legend(bbox_to_anchor=(1,1))
    plt.xlabel('Epochs', fontdict=title_font)
    plt.ylabel('F1-score (%)', fontdict=title_font)

    plt.savefig('domain_performance.png')
    
def get_scheduler_function(name, total_iters, final_lr=0):
    print("LR Scheduler: {}".format(name))
    if name == 'cosine':
        return lambda step: ((1 + math.cos(step * math.pi / total_iters)) / 2) * (1 - final_lr) + final_lr
    elif name == 'linear':
        return lambda step: 1 - (1 - final_lr) / total_iters * step
    elif name == 'exp':
        return lambda step: (1 - step / total_iters) ** 0.9
    elif name == 'none':
        return lambda step: 1
    else:
        raise ValueError(name)
    
def fix_BN_stat(module):
    classname = module.__class__.__name__
    if classname.find('BatchNorm2d') != -1:
        module.eval()
    #if classname.find('LayerNorm') != -1:
    #    module.eval()

def freeze_BN_stat(model):
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        if hasattr(model.module, 'backbone'):
            print("freeze backbone BN stat")
            model.module.backbone.apply(fix_BN_stat)

def SCD_evaluate(args, model, data_loader, device, save_imgs_dir=None):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")  
    metric_logger.add_meter('F1 (t0->t1)', utils.SmoothedValue(window_size=1, fmt='{value:.4f} ({global_avg:.4f})'))
    metric_logger.add_meter('F1 (t1->t0)', utils.SmoothedValue(window_size=1, fmt='{value:.4f} ({global_avg:.4f})'))
    header = 'Test:'

    if ('PSCD' in args.test_dataset) or ('TSUNAMI' in args.test_dataset):
        test_size = (256, 1024)
    else:
        # VL-CMU-CD
        test_size = (512, 512)

    with torch.no_grad():
        for image, target in metric_logger.log_every(data_loader, 100, header):
            image, target = image.to(device), target.to(device)

            # t0 -> t1
            output = model(image)
            if isinstance(output, OrderedDict):
                output = output['out']
            mask_pred_t0 = torch.topk(output.data, 1, dim=1)[1][:, 0]
            mask_gt_t0 = (target > 0)[:, 0]
            _,_,_, f1score_t0 = utils.CD_metric_torch(mask_pred_t0, mask_gt_t0)

            # reverse input order
            t0 = torch.split(image, 3, dim=1)[0]
            t1 = torch.split(image, 3, dim=1)[1]
            image_rev = torch.cat([t1,t0], dim=1)

            # t1 -> t0
            output = model(image_rev)
            if isinstance(output, OrderedDict):
                output = output['out']
            mask_pred_t1 = torch.topk(output.data, 1, dim=1)[1][:, 0]
            mask_gt_t1 = (target > 0)[:, 0]
            _, _, _, f1score_t1 = utils.CD_metric_torch(mask_pred_t1, mask_gt_t1)
    
            metric_logger.F1score.update(f1score_t0.mean(), n=len(f1score_t0))
            metric_logger.rev_F1score.update(f1score_t1.mean(), n=len(f1score_t1))
            
            if save_imgs_dir:
                # PSCD, TSUNAMI has test resolution of (256,1024) 
                # VL-CMU-CD has test resolution of (512,512)
                mask_pred_t0, mask_gt_t0= T.Resize((test_size))(mask_pred_t0, mask_gt_t0)
                output_pil = data_loader.dataset.get_pil(image[0], mask_gt_t0, mask_pred_t0, mask_pred_t1)
                output_pil.save(os.path.join(save_imgs_dir, "{}_{}.png".format(utils.get_rank(), metric_logger.F1score.count)))

        metric_logger.synchronize_between_processes()

    print("{} {} Total: {} Metric F1 (t0->t1): {:.4f} F1 (t1->t0): {:.4F} Avg : {:.4F}".format(
        header,
        data_loader.dataset.name,
        metric_logger.F1score.count,
        metric_logger.F1score.global_avg,
        metric_logger.rev_F1score.global_avg,
        (metric_logger.F1score.global_avg + metric_logger.rev_F1score.global_avg) / 2.0
    ))
    return metric_logger.F1score.global_avg, metric_logger.rev_F1score.global_avg

def warmup(num_iter, num_warmup, optimizer):
    if num_iter < num_warmup:
        # warm up
        xi = [0, num_warmup]
        for j, x in enumerate(optimizer.param_groups):
            # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
            #x['lr'] = np.interp(num_iter, xi, [0.1 if j == 2 else 0.0, x['initial_lr'] * lf(epoch)])
            x['lr'] = np.interp(num_iter, xi, [0.1 if j == 2 else 0.0, x['lr']])
            if 'momentum' in x:
                x['momentum'] = np.interp(num_iter, xi, [0.8, 0.9])

def SCD_train_one_epoch(model, criterion, optimizer, scaler, data_loader, lr_scheduler, num_warmup, device, epoch, print_freq, cycle):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value}'))
    metric_logger.add_meter('F1 (t0->t1)', utils.SmoothedValue(window_size=1, fmt='{value:.4f} ({global_avg:.4f})'))
    metric_logger.add_meter('F1 (t1->t0)', utils.SmoothedValue(window_size=1, fmt='{value:.4f} ({global_avg:.4f})'))
    header = 'Epoch: [{}]'.format(epoch)
    warmup(lr_scheduler._step_count, num_warmup, optimizer)
    for image, target in metric_logger.log_every(data_loader, print_freq, header):
        image, target = image.to(device), target.to(device)

        # t0 -> t1
        output = model(image)

        loss = criterion(output, target[:,0])

        # t1 -> t0
        if cycle:
            t0 = torch.split(image, 3, dim=1)[0]
            t1 = torch.split(image, 3, dim=1)[1]
            image_rev = torch.cat([t1,t0], dim=1)
            output_rev = model(image_rev)
            loss += criterion(output_rev, target[:,0])

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        #loss.backward()
        #optimizer.step()
        lr_scheduler.step()
        warmup(lr_scheduler._step_count, num_warmup, optimizer)
        if isinstance(output, OrderedDict):
            output = output['out']
            output_rev = output_rev['out']

        mask_pred_t0 = torch.topk(output.data, 1, dim=1)[1][:, 0]
        mask_gt_t0 = (target > 0)[:, 0]
        mask_pred_t1 = torch.topk(output_rev.data, 1, dim=1)[1][:, 0]
        mask_gt_t1 = (target > 0)[:, 0]

        _, _, _, f1score_t0 = utils.CD_metric_torch(mask_pred_t0, mask_gt_t0)
        _, _, _, f1score_t1 = utils.CD_metric_torch(mask_pred_t1, mask_gt_t1)
        
        metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])
        metric_logger.f1score.update(f1score_t0.mean(), n=len(f1score_t0))
        metric_logger.rev_f1score.update(f1score_t1.mean(), n=len(f1score_t1))

def create_dataloader(args):
    # our purpose is to get generalizable SCD model
    dataset = dataset_dict[args.train_dataset](args, train=True)
    dataset_test = dataset_dict[args.test_dataset](args, train=False)

    dataset_test2 = None
    dataset_test3 = None

    # use whole dataset for testing
    if args.train_dataset == 'VL_CMU_CD':
        dataset_test2 = dataset_dict['TSUNAMI_total'](args, train=False)
        dataset_test3 = dataset_dict['PSCD_total'](args, train=False)

    elif args.train_dataset == 'PSCD':
        dataset_test2 = dataset_dict['TSUNAMI_total'](args, train=False)
        dataset_test3 = dataset_dict['VL_CMU_CD_total'](args, train=False)

    train_sampler = torch.utils.data.RandomSampler(dataset)
    test_sampler = torch.utils.data.SequentialSampler(dataset_test)
    test_sampler2 = torch.utils.data.SequentialSampler(dataset_test2) if dataset_test2 else None
    test_sampler3 = torch.utils.data.SequentialSampler(dataset_test3) if dataset_test3 else None

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size,
        sampler=train_sampler, num_workers=args.workers,
        collate_fn=utils.collate_fn, drop_last=True)
    
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1,
        sampler=test_sampler, num_workers=args.workers,
        collate_fn=utils.collate_fn)
    
    data_loader_test2 = None
    data_loader_test3 = None
    if dataset_test2:
        data_loader_test2 = torch.utils.data.DataLoader(
            dataset_test2, batch_size=1,
            sampler=test_sampler2, num_workers=args.workers,
            collate_fn=utils.collate_fn)
        
    if dataset_test3:
        data_loader_test3 = torch.utils.data.DataLoader(
            dataset_test3, batch_size=1,
            sampler=test_sampler3, num_workers=args.workers,
            collate_fn=utils.collate_fn)
        
    return dataset, train_sampler, data_loader, dataset_test, data_loader_test, dataset_test2, data_loader_test2, dataset_test3, data_loader_test3

def prepare_train(args, model_without_ddp, dataset, data_loader):
    if "fcn" in args.model or "deeplabv3" in args.model:
        params_to_optimize = [
            {"params": [p for p in model_without_ddp.backbone.parameters() if p.requires_grad]},
            {"params": [p for p in model_without_ddp.classifier.parameters() if p.requires_grad]},
        ]
    else:
        params_to_optimize = model_without_ddp.parameters()

    if args.opt == 'adam':
        optimizer = torch.optim.Adam(params_to_optimize, lr=args.lr, betas=(args.momentum, 0.999), weight_decay=args.weight_decay)
    elif args.opt == 'sgd':
        optimizer = torch.optim.SGD(
            params_to_optimize,
            lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    else:
        raise ValueError(args.opt)
    
    if args.loss_weight:
        if args.train_dataset == 'VL_CMU_CD':
            loss_weight = torch.tensor([0.0645, 0.9355]).cuda()
        elif args.train_dataset == 'PSCD':
            loss_weight = torch.tensor([0.1561, 0.8439]).cuda()
    else:
        loss_weight = None
    
    criterion = loss.get_loss(args.loss, loss_weight)
    
    lambda_lr = get_scheduler_function(args.lr_scheduler, args.epochs * len(data_loader), final_lr=0.2*args.lr)
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_lr)
    if args.warmup:
        num_warmup = max(round(5 * len(data_loader)), 1000)
    else:
        num_warmup = 0
    scaler = torch.cuda.amp.GradScaler(enabled=True)

    return optimizer, criterion, lr_scheduler, scaler, num_warmup

def main(args):
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.output_dir and args.test_only:
        utils.mkdir(args.output_dir)
    
    device = torch.device(args.device)
    dataset, _ , data_loader, dataset_test, data_loader_test, dataset_test2, data_loader_test2, dataset_test3, data_loader_test3 = create_dataloader(args)
    train_one_epoch = SCD_train_one_epoch
    evaluate = SCD_evaluate

    args.num_classes = 2
    model = model_dict[args.model](args)
    model = model.to(device)
    model_without_ddp = model
    optimizer, criterion, lr_scheduler, scaler, num_warmup = prepare_train(args, model_without_ddp, dataset, data_loader)  

    if args.test_only:
        # checkpoint = torch.load(f'./output/backbone_method_trainset_res_epochs.pth')
        checkpoint_path = f'{args.checkpoint_path}'
        checkpoint = torch.load(checkpoint_path)

        sd = checkpoint['model']
        model_without_ddp.load_state_dict(sd)
        if args.save_imgs:
            save_imgs_dir = os.path.join(args.output_dir, 'img')
            os.makedirs(save_imgs_dir, exist_ok=True)
        else:
            save_imgs_dir = None

        f1score,  f1score_rev = evaluate(model, data_loader_test, device=device, save_imgs_dir=save_imgs_dir)
        print(f'F1-score (t0->t1) : {f1score:.4f}')
        print(f'F1-score (t1->t0) : {f1score_rev:.4f}')
        avg = (f1score + f1score_rev) / 2.0
        print(f'Average F1-score (t1<->t0) : {avg:.4f}')    
        return 

    best = -1
    avg_f1_target_list = []
    avg_f1_c1_list = []
    avg_f1_c2_list = []
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        train_one_epoch(model, criterion, optimizer, scaler, data_loader, lr_scheduler, num_warmup, device, epoch, args.print_freq, args.cycle)
        
        if epoch % args.eval_every == 0:
            f1score, f1score_rev = evaluate(model, data_loader_test, device=device)
            avg = (f1score + f1score_rev) / 2.0
            avg_f1_target_list.append(avg)
            if dataset_test2:
                f1score2, f1score_rev2 = evaluate(model, data_loader_test2, device=device)
                avg2 = (f1score2+f1score_rev2) / 2.0
                avg_f1_c1_list.append(avg2)
            if dataset_test3:
                f1score3, f1score_rev3 = evaluate(model, data_loader_test3, device=device)
                avg3 = (f1score3+f1score_rev3) / 2.0
                avg_f1_c2_list.append(avg3)
        
        if avg > best:
            best = avg
            # weight_name = f'{args.backbone}_{args.method}_{args.train_dataset}_{args.input_size}_{args.epochs}.pth'
            weight_name = None
            utils.save_on_master(
                checkpoint,
                os.path.join(args.output_dir, f'best.pth'))
        
    if args.save_imgs:
        save_imgs_dir = os.path.join(args.output_dir, '{}_img'.format(dataset_test.name))
        os.makedirs(save_imgs_dir, exist_ok=True)
        _ = evaluate(model, data_loader_test, device=device, save_imgs_dir=save_imgs_dir)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

    print('Best Avg F1-score {:.4f}'.format(best))
    print(f'results are saved in {args.output_dir}')
    print(f"Cylcic loss was {args.cycle}")


    with open(f"{args.output_dir}/target.pkl","wb") as f:
        pickle.dump(avg_f1_target_list, f)    
    with open(f"{args.output_dir}/cross_domain1.pkl","wb") as f:
        pickle.dump(avg_f1_c1_list, f)   
    with open(f"{args.output_dir}/cross_domain2.pkl","wb") as f:
        pickle.dump(avg_f1_c2_list, f)   

    # plot variation of f1-score
    draw_figure(avg_f1_target_list, avg_f1_c1_list, avg_f1_c2_list)

def arg_as_list(s):
    import ast
    v = ast.literal_eval(s)
    if type(v) is not list:
        print("arg type error")
        return 
    return v

def get_args_parser(add_help=True):
    import argparse
    parser = argparse.ArgumentParser(description='PyTorch change detection', add_help=add_help)
    parser.add_argument('--train-dataset', default='VL_CMU_CD', help='dataset name')
    parser.add_argument('--test-dataset', default='VL_CMU_CD', help='dataset name')
    parser.add_argument('--test-dataset2', default='PSCD_total', help='')
    parser.add_argument('--test-dataset3', default='TSUNAMI_total', help='')

    parser.add_argument('--workers', default=0, type=int, metavar='N',
                            help='number of data loading workers (default: 0)')
    parser.add_argument('--input-size', default=1024, type=int, metavar='N',
                        help='the input-size of images')
    parser.add_argument('--randomflip', default=0.5, type=float, help='random flip input')
    parser.add_argument('--randomrotate', dest="randomrotate", action="store_true", help='random rotate input')
    parser.add_argument('--randomcrop', dest="randomcrop", action="store_true", help='random crop input')
    parser.add_argument('--data-cv', default=0, type=int, metavar='N',
                        help='the number of cross validation')
    
    parser.add_argument('--backbone', default='resnet18', help='feature extractor')
    parser.add_argument('--model', default='changesam', help='model')
    parser.add_argument('--method', default='c3po', help='method')
    parser.add_argument('--freeze', default='freeze', help='freeze backbone')
    parser.add_argument('--mtf', default='id', help='choose branches to use')
    parser.add_argument('--msf', default=4, type=int, help='the number of MSF layers')
    
    parser.add_argument('--device', default='cuda', help='device')
    parser.add_argument('-b', '--batch-size', default=4, type=int)
    parser.add_argument('--epochs', default=150, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--loss', default='bi', type=str, help='the training loss')
    parser.add_argument('--loss-weight', action="store_true", help='add weight for loss')
    parser.add_argument('--opt', default='adam', type=str, help='the optimizer')
    parser.add_argument('--lr-scheduler', default='cosine', type=str, help='the lr scheduler')
    parser.add_argument('--lr', default=0.0001, type=float, help='initial learning rate')
    parser.add_argument('--warmup', dest="warmup", action="store_true", help='warmup the lr')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=0, type=float,
                        metavar='W', help='weight decay (default: 0)',
                        dest='weight_decay')
    parser.add_argument('--print-freq', default=30, type=int, help='print frequency')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval-every', default=1, type=int, metavar='N',
                        help='eval the model every n epoch')
    parser.add_argument("--test-only", dest="test_only", help="Only test the model", action="store_true")
    
    parser.add_argument("--cycle", dest="cycle", help="cycle_loss", action="store_true")
    
    parser.add_argument("--save-imgs", dest="save_imgs", action="store_true",
                        help="save the predicted mask")
    parser.add_argument('--seed', default=0, type=int, help='reproductivity')
    parser.add_argument('--layers', default=[], type=arg_as_list, help='layers to fuse')
    
    return parser
    

if __name__ == "__main__":
    args = get_args_parser().parse_args()
    # print(f"Using layer {args.layers}")
    print(f"Cylcic loss {args.cycle}")
    output_dir = 'output'
    if 'mtf' in args.model:
        save_path = "{}_{}_{}_{}_{}/{date:%Y-%m-%d_%H_%M_%S}".format(
            args.model, args.mtf, args.input_size, args.train_dataset, args.data_cv, date=datetime.datetime.now())
    else:
        save_path = "{}_{}_{}/{date:%Y-%m-%d_%H_%M_%S}".format(
            args.model, args.train_dataset, args.data_cv, date=datetime.datetime.now())
    args.output_dir = os.path.join(output_dir, save_path)
    print(f'Experiment will be saved in {args.output_dir}')
    main(args)

        
