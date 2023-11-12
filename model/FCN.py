from collections import OrderedDict
import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torch.jit.annotations import Dict

from torchvision.models.segmentation._utils import _SimpleSegmentationModel
from torchvision.models.segmentation.fcn import FCN, FCNHead

from model.c3po import backbone_mtf_msf

def sam_mtf_msf_fcn(args):
    backbone = backbone_mtf_msf('sam', fpn_num=args.msf, mode=args.mtf, layer_lst=args.layers)
    aux_classifier = None
    classifier = FCNHead(512, args.num_classes)
    model = FCN(backbone, classifier, aux_classifier)
    return model

def dinov2_mtf_msf_fcn(args):
    backbone = backbone_mtf_msf('dinov2', fpn_num=args.msf, mode=args.mtf, layer_lst=args.layers)
    aux_classifier = None
    classifier = FCNHead(512, args.num_classes)
    model = FCN(backbone, classifier, aux_classifier)
    return model

def resnet18_mtf_msf_fcn(args):
    backbone = backbone_mtf_msf('resnet18', fpn_num=args.msf, mode=args.mtf)
    aux_classifier = None
    classifier = FCNHead(512, args.num_classes)
    model = FCN(backbone, classifier, aux_classifier)
    return model

def dinov1_mtf_msf_fcn(args):
    backbone = backbone_mtf_msf('dinov1', fpn_num=args.msf, mode=args.mtf, layer_lst=args.layers)
    aux_classifier = None
    classifier = FCNHead(512, args.num_classes)
    model = FCN(backbone, classifier, aux_classifier)
    return model