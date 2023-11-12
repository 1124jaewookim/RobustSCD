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
from c3po import get_backbone

class RobustSCD(nn.module):
    def __init__(self, backbone_name, combinefeature, layer_lst = None):
        self.backbone_name = backbone_name
        self.combinefeature = combinefeature
        self.encoder = get_backbone(backbone_name)

    def forward(self, img):
        out = OrderedDict()
        img_t0, img_t1 = torch.split(img,3,1)
        t0_fs = self.encoder(img_t0)
        t1_fs = self.encoder(img_t1)

        s
        out['out'] = self.combinefeature(t0_fs, t1_fs)
        return out

def Robust_model(backbone_name, layer_lst = None):
    model = None
        
    return model