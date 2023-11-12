# RobustSCD

from model.FCN import sam_mtf_msf_fcn, resnet18_mtf_msf_fcn, dinov2_mtf_msf_fcn, dinov1_mtf_msf_fcn
from model.TANet import dr_tanet_refine_resnet18

# from model.TANet import dr_tanet_refine_resnet18, dr_tanet_resnet18, tanet_refine_resnet18, tanet_resnet18

from model.cscdnet import cdresnet, cscdnet

model_dict = {
'sam_mtf_msf_fcn': sam_mtf_msf_fcn,
'dinov2_mtf_msf_fcn': dinov2_mtf_msf_fcn,
'dinov1_mtf_msf_fcn': dinov1_mtf_msf_fcn,
'dr_tanet_refine_resnet18': dr_tanet_refine_resnet18,
'resnet18_mtf_msf_fcn' : resnet18_mtf_msf_fcn,
'cscdnet': cscdnet,
'cdresnet': cdresnet,

}
