# 1. This python file contains various backbones (Feature Extractors) for SCD.
# 2. Also, it contains methodoloy for C-3PO.
# backbone: DINO, DINOv2, MAE, SAM, CLIP

import random
import math
import os
import numpy as np
from torch import Tensor, nn
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Type
from collections import OrderedDict
from typing import List
from segment_anything import sam_model_registry
from model.resnet import ResNet
from correlation import FunctionCorrelation, CorrelationVolume, FeatureL2Norm
import torch.nn as nn
from typing import Any, Optional, Tuple, Type
from utils_ import freeze

# ResNet, DINO, DINOv2, SAM, MAE, CLIP
def get_backbone(name):
    if 'resnet18' in name:
        return ResNet(name)
    elif 'sam' in name:
        sam_checkpoint = './sam_checkpoint/sam_vit_b_01ec64.pth'
        model_type = 'vit_b'
        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        model = sam.image_encoder
        # only use strong image encoder
        del sam.prompt_encoder
        del sam.mask_decoder
        return model
    elif 'dinov2' in name:
        backbone = 'dinov2_vitb14'
        model = torch.hub.load(repo_or_dir="facebookresearch/dinov2", model=backbone)
        return model
    elif 'dinov1' in name:
        model = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16')
        return model
    elif 'mae' in name:
        pass
    elif 'clip' in name:
        pass

def get_channels(backbone_name):
    if 'resnet' in backbone_name:
        d = {
            'resnet18': 64,
            'resnet50': 256,
        }
        channel = d[backbone_name]
        return [channel, channel, channel * 2, channel * 4, channel * 8]
    
    elif 'sam' in backbone_name:
        return [768]*4
    elif 'dinov2' in backbone_name:
        return [768]*4
    elif 'dinov1' in backbone_name:
        return [768]*4
    
class MTF(nn.Module):
    def __init__(self, channel, mode='iade', kernel_size=1):
        super(MTF, self).__init__()
        assert mode in ['i', 'a', 'd', 'e', 'ia', 'id', 'ie', 'iae', 'ide', 'iad', 'iade', 'i2ade', 'iad2e', 'i2ad2e', 'i2d']
        self.mode = mode
        self.channel = channel
        self.relu = nn.ReLU(inplace=True)
        if kernel_size == 1:
            padding = 0
        elif kernel_size == 3:
            padding = 1
        if 'i2' in mode:
            self.i0 = nn.Conv2d(self.channel, self.channel, kernel_size, padding=padding, stride=1, bias=False)
            self.i1 = nn.Conv2d(self.channel, self.channel, kernel_size, padding=padding, stride=1, bias=False)
        else:
            self.conv = nn.Conv2d(self.channel, self.channel, kernel_size, padding=padding, stride=1, bias=False)
            
        if 'ad2'in mode:
            self.app = nn.Conv2d(self.channel, self.channel, kernel_size, padding=padding, stride=1, bias=False)
            self.dis = nn.Conv2d(self.channel, self.channel, kernel_size, padding=padding, stride=1, bias=False)
        else:
            self.res = nn.Conv2d(self.channel, self.channel, kernel_size, padding=padding, stride=1, bias=False)
            
        self.exchange = nn.Conv2d(self.channel, self.channel, kernel_size, padding=padding, stride=1, bias=False)
        print("MTF: mode: {} kernel_size: {}".format(self.mode, kernel_size))
        
        self.eps = 1e-5
        
    def forward(self, f0, f1):
        #t0 = self.conv(f0)
        #t1 = self.conv(f1)
        if 'i2' in self.mode:
            info = self.i0(f0) + self.i1(f1)
        else:
            info = self.conv(f0 + f1)
            
        if 'd' in self.mode:
            if 'ad2' in self.mode:
                disappear = self.dis(self.relu(f0 - f1))
            else:
                disappear = self.res(self.relu(f0 - f1))
        else:
            disappear = 0

        if 'a' in self.mode:
            if 'ad2' in self.mode:
                appear = self.app(self.relu(f1 - f0))
            else:
                appear = self.res(self.relu(f1 - f0))
        else:
            appear = 0

        if 'e' in self.mode:
            exchange = self.exchange(torch.max(f0, f1) - torch.min(f0, f1))
        else:
            exchange = 0

        if self.mode == 'i':
            f = info
        elif self.mode == 'a':
            f = appear
        elif self.mode == 'd':
            f = disappear
        elif self.mode == 'e':
            f = exchange
        elif self.mode == 'ia':
            f = info + 2 * appear
        elif self.mode in ['id', 'i2d']:
            f = info + 2 * disappear
        elif self.mode == 'ie':
            f = info + 2 * exchange
        elif self.mode == 'iae':
            f = info + appear + exchange
        elif self.mode == 'ide':
            f = info + disappear + exchange
        elif self.mode == 'iad':
            f = info + disappear + appear
        elif self.mode in ['iade', 'i2ade', 'iad2e', 'i2ad2e']:
            f = info + disappear + appear + exchange

        f = self.relu(f)
        return f

class MSF(nn.Module):
    def __init__(self, channels, total_f=5, fpn_channel=None, with_bn=False, mode='iade'):
        super(MSF, self).__init__()
        print("MSF: {}".format(channels))
        self.num_f = len(channels)
        self.total_f = total_f
        assert 0 < self.num_f <= self.total_f
        cf_list = []
        cf_inner = []
        cf_layer = []
        for i in range(self.num_f):
            cf_list.append(MTF(channels[i], mode, kernel_size=3))
            cf_inner.append(self._make_layer(channels[i], fpn_channel, 1, with_bn))
            cf_layer.append(self._make_layer(fpn_channel, fpn_channel, 3, with_bn))

        self.cfs = nn.ModuleList(cf_list)
        self.cf_inners = nn.ModuleList(cf_inner)
        self.cf_layers = nn.ModuleList(cf_layer)

        self.reduce = nn.Conv2d(fpn_channel * self.num_f, fpn_channel, 3, padding=1, stride=1, bias=False)
        self.bn   = nn.BatchNorm2d(fpn_channel)
        self.relu = nn.ReLU(inplace=True)

    def _make_layer(self, in_channel, out_channel, kernel, with_bn):
        l = []
        if kernel == 1:
            l.append(nn.Conv2d(in_channel, out_channel, 1))
        elif kernel == 3:
            l.append(nn.Conv2d(in_channel, out_channel, 3, padding=1))
        else:
            raise ValueError(kernel)
        
        if with_bn:
            l.append(nn.BatchNorm2d(out_channel))

        l.append(nn.ReLU(inplace=True))
        return nn.Sequential(*l)

    def forward(self, t0_fs, t1_fs=None):
        cfs = []
        for i in range(self.num_f):
            k = i + self.total_f - self.num_f
            if t1_fs is None:
                cfs.append(self.cfs[i](t0_fs[k], torch.zeros_like(t0_fs[k])))
            else:
                cfs.append(self.cfs[i](t0_fs[k], t1_fs[k]))

        resize_shape = cfs[0].shape[-2:]
        final_list = []
        last_inner = None
        for i in range(self.num_f - 1, -1, -1):
            cf = self.cf_inners[i](cfs[i])
            if last_inner is None:
                last_inner = cf
            else:
                inner_top_down = F.interpolate(last_inner, size=cf.shape[-2:], mode="nearest")
                last_inner = cf + inner_top_down
            cf_layer = self.cf_layers[i](last_inner)
            final_list.append(cf_layer)

        final_list = [F.interpolate(cf_layer, resize_shape, mode='bilinear') for cf_layer in final_list]
        cf = torch.cat(final_list, dim=1)
        cf = self.relu(self.bn(self.reduce(cf)))

        return cf

class Backbone_MTF_MSF(nn.Module):
    def __init__(self, backbone_name, combinefeature, share_weight=False):
        super(Backbone_MTF_MSF, self).__init__()
        self.share_weight = share_weight
        self.backbone_name = backbone_name
        
        if share_weight:
            self.encoder = get_backbone(backbone_name)
        else:
            self.encoder1 = get_backbone(backbone_name)
            self.encoder2 = get_backbone(backbone_name)
        self.combinefeature = combinefeature

    def forward(self, img):
        out = OrderedDict()
        img_t0, img_t1 = torch.split(img,3,1)
        if self.share_weight:
            t0_fs = self.encoder(img_t0)
            t1_fs = self.encoder(img_t1)
        else:
            t0_fs = self.encoder1(img_t0)
            t1_fs = self.encoder2(img_t1)

        out['out'] = self.combinefeature(t0_fs, t1_fs)
        return out
    
class Backbone_SAM(nn.Module):
    def __init__(self, backbone_name, combinefeature, layer_lst = None):
        super(Backbone_SAM, self).__init__()

        self.backbone_name = backbone_name
        self.encoder = get_backbone(backbone_name)
        self.combinefeature = combinefeature
        self.layer_lst = layer_lst
    def forward(self, img):
        out = OrderedDict()
        t0 = []
        t1 = []
        
        img_t0, img_t1 = torch.split(img,3,1)

        _, t0_fs = self.encoder(img_t0)
        _, t1_fs = self.encoder(img_t1)
        
        for i, (t0_fmap, t1_fmap) in enumerate(zip(t0_fs, t1_fs)):
            if i != 12:
                t0.append(t0_fmap.permute(0,3,1,2))
                t1.append(t1_fmap.permute(0,3,1,2))
            else:
                t0.append(t0_fmap)
                t1.append(t1_fmap)
            
        t0_fs = [] 
        t1_fs = [] 
        for num_layer in self.layer_lst:
            t0_fs.append(t0[num_layer])
            t1_fs.append(t1[num_layer])
        
        out['out'] = self.combinefeature(t0_fs, t1_fs)
        return out
    
class Backbone_dinov2(nn.Module):
    def __init__(self, backbone_name, combinefeature, layer_lst = None):
        super(Backbone_dinov2, self).__init__()

        self.backbone_name = backbone_name
        self.encoder = get_backbone(backbone_name)
        freeze(self.encoder)
        self.combinefeature = combinefeature
        self.layer_lst = layer_lst
    def forward(self, img):
        out = OrderedDict()
        t0 = []
        t1 = []
        t0_ = []
        t1_ = []
        img_t0, img_t1 = torch.split(img,3,1)

        x0 = self.encoder.patch_embed(img_t0) # (b, 3136, 768) = (b,768,56,56)
        x1 = self.encoder.patch_embed(img_t1)
        
        for i, block in enumerate(self.encoder.blocks):
            x0 =  block(x0)
            x1 =  block(x1)
            t0.append(x0.permute(0,2,1).view(-1,768,56,56))
            t1.append(x1.permute(0,2,1).view(-1,768,56,56))

        for num_layer in self.layer_lst:
            t0_.append(t0[num_layer])
            t1_.append(t1[num_layer])
            
        out['out'] = self.combinefeature(t0_, t1_)
        return out
    
    
class Backbone_dinov1(nn.Module):
    def __init__(self, backbone_name, combinefeature, layer_lst = None):
        super(Backbone_dinov1, self).__init__()

        self.backbone_name = backbone_name
        self.encoder = get_backbone(backbone_name)
        freeze(self.encoder)
        self.combinefeature = combinefeature
        self.layer_lst = layer_lst
    def forward(self, img):
        out = OrderedDict()
        t0 = []
        t1 = []
        t0_ = []
        t1_ = []
        img_t0, img_t1 = torch.split(img,3,1)

        x0 = self.encoder.patch_embed(img_t0) # (b, 3136, 768) = (b,768,56,56)
        x1 = self.encoder.patch_embed(img_t1)
        
        for i, block in enumerate(self.encoder.blocks):
            x0 =  block(x0)
            x1 =  block(x1)
            t0.append(x0.permute(0,2,1).view(-1,768,64,64))
            t1.append(x1.permute(0,2,1).view(-1,768,64,64))

        for num_layer in self.layer_lst:
            t0_.append(t0[num_layer])
            t1_.append(t1[num_layer])
            
        out['out'] = self.combinefeature(t0_, t1_)
        return out
    
def backbone_mtf_msf(backbone_name, fpn_num=4, mode='iade', layer_lst = None):
    # B -> T -> S -> H 
    channels = get_channels(backbone_name)
    combinefeature = MSF(channels[-fpn_num:], total_f=len(channels), fpn_channel=512, with_bn=False, mode=mode)
    if backbone_name == 'sam':
        model = Backbone_SAM(backbone_name, combinefeature, layer_lst = layer_lst)
    elif backbone_name == 'resnet18':
        model = Backbone_MTF_MSF(backbone_name, combinefeature, share_weight=True)
    elif backbone_name == 'dinov2':
        model = Backbone_dinov2(backbone_name, combinefeature, layer_lst = layer_lst)
    elif backbone_name == 'dinov1':
        model = Backbone_dinov1(backbone_name, combinefeature, layer_lst = layer_lst)
        
    return model