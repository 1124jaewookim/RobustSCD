import numpy as np
import torch
from torch import nn
from torchvision.models import resnet 
from typing import Any, Optional, Tuple, Type
from .common import LayerNorm2d
from collections import OrderedDict
import torch.nn.functional as F




class Backbone(nn.Module):
    def __init__(self, layer_list):
        super(Backbone, self).__init__()
        self.layers = nn.ModuleList(layer_list)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f0 = self.layers[0](x)
        f1 = self.layers[1](f0)
        f2 = self.layers[2](f1)
        f3 = self.layers[3](f2)
        f4 = self.layers[4](f3)
        return (f0, f1, f2, f3, f4)


class ResNet(Backbone):
    def __init__(self, name):
        assert name in ['resnet18', 'resnet50']
        self.name = name
        super(ResNet, self).__init__(get_layers(name))


def get_layers(name):
    if 'resnet18' == name:
        replace_stride_with_dilation=[False, False, False]
        model = resnet.__dict__[name](
                    pretrained=True,
                    replace_stride_with_dilation=replace_stride_with_dilation)
    elif 'resnet50' == name:
        replace_stride_with_dilation=[False, True, True]
        model = resnet.__dict__[name](
                    pretrained=True,
                    replace_stride_with_dilation=replace_stride_with_dilation)
    else:
        raise ValueError(name)
    
    layer0 = nn.Sequential(model.conv1, model.bn1, model.relu, model.maxpool)
    layer1 = model.layer1
    layer2 = model.layer2
    layer3 = model.layer3
    layer4 = model.layer4
    return [layer0, layer1, layer2, layer3, layer4]

def get_backbone(backbone_name):
    if 'resnet' in backbone_name:
        model = ResNet(backbone_name)
        return model
        
        
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
    def __init__(self, channels, total_f=5, fpn_channel=None, with_bn=False, mode='iade', corr= False):
        super(MSF, self).__init__()
        print("MSF: {}".format(channels))
        self.num_f = len(channels)
        self.total_f = total_f
        self.corr = corr
        assert 0 < self.num_f <= self.total_f
        cf_list = []
        cf_inner = []
        cf_layer = []
        sim_layer = []
        for i in range(self.num_f):
            cf_list.append(MTF(channels[i], mode, kernel_size=3))
            cf_inner.append(self._make_layer(channels[i], fpn_channel, 1, with_bn))
            cf_layer.append(self._make_layer(fpn_channel, fpn_channel, 3, with_bn))
            # sim_layer.append((self._make_layer(81, fpn_channel, 3, with_bn)))
            
        self.cfs = nn.ModuleList(cf_list)
        self.cf_inners = nn.ModuleList(cf_inner)
        self.cf_layers = nn.ModuleList(cf_layer)
        # self.sim_layers = nn.ModuleList(sim_layer)

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

    def forward(self, t0_fs, t1_fs=None, sim_map=None):
        # sim_map_list = [sim_map] * self.num_f
        # shape = [torch.Size([16,16]), torch.Size([32,32]), torch.Size([64,64]),torch.Size([128,128])]
        # assert self.num_f == len(shape)
        
        if sim_map is not None:
            sim_maps = [F.interpolate(s, sh, mode='bilinear') for s, sh in zip(sim_map_list, shape)] # 4 interporalted results
            sim_results = []
            for i in range(self.num_f):
                sim_results.append(self.sim_layers[i](sim_maps[i]))
            
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
            
        # out = []
        # for i,j in zip(sim_results, final_list):
        #   out.append(i * j)
          
        final_list = [F.interpolate(cf_layer, resize_shape, mode='bilinear') for cf_layer in final_list]
        cf = torch.cat(final_list, dim=1)
        cf = self.relu(self.bn(self.reduce(cf)))
        return cf
    
    

class Backbone_MTF_MSF(nn.Module):
    def __init__(self, backbone_name, combinefeature, share_weight=False):
        super(Backbone_MTF_MSF, self).__init__()
        self.share_weight = share_weight
        # self.vpr_encoder1 = get_backbone('vpr')
        # self.vpr_encoder2 = get_backbone('vpr')
        # self.l2norm = FeatureL2Norm()
        
        if share_weight:
            self.encoder = get_backbone(backbone_name)
        else:
            self.encoder1 = get_backbone(backbone_name)
            self.encoder2 = get_backbone(backbone_name)
        self.combinefeature = combinefeature
        
    def forward(self, img):
        out = OrderedDict()
        img_t0, img_t1 = torch.split(img,3,1)
        
        # vpr_t0_fs = self.vpr_encoder1(img_t0)
        # vpr_t1_fs = self.vpr_encoder2(img_t1)
        # # 32x32 resoulution feature to guide backbone
        # corr_map = FunctionCorrelation(self.l2norm(vpr_t0_fs[3]), self.l2norm(vpr_t1_fs[3]))
        # sim_map = 1 - corr_map
        
        if self.share_weight:
            t0_fs = self.encoder(img_t0)
            t1_fs = self.encoder(img_t1)
        else:
            t0_fs = self.encoder1(img_t0)
            t1_fs = self.encoder2(img_t1)
        out['out'] = self.combinefeature(t0_fs, t1_fs)
        return out
    
# ---------------------------------------------------------------------------- # 
class PromptGenerator(nn.Module):
    def __init__(self, backbone_name):
        super(PromptGenerator, self).__init__()
        
        from torchvision.models.segmentation.fcn import FCN, FCNHead
        channel = 64
        channels = [channel, channel, channel * 2, channel * 4, channel * 8]
        self.encoder = get_backbone(backbone_name)
        combinefeature = MSF(channels[-4:], total_f=5, fpn_channel=512, with_bn=False, mode='id')
        backbone = Backbone_MTF_MSF(backbone_name, combinefeature, share_weight=True)
        classifier = FCNHead(512, 2)
        
        self.seg_model = FCN(backbone, classifier, None)
        self.conv = nn.Conv2d(1,1,kernel_size=1, stride=1, padding=0, bias=False)
        
    def forward(self, imgs):
        output = self.seg_model(imgs) 
        output = output['out']
        
        # 이는 row resolution mask 여야한다. 256 사이즈여야한다.
        mask_logits = torch.sub(output[:,1,:,:], output[:,0,:,:])
        mask_logits = mask_logits.unsqueeze(1)
        low_res_mask = F.interpolate(mask_logits, (256,256))
        low_res_mask = self.conv(low_res_mask)
        
        return low_res_mask, output
    
