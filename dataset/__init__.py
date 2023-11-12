from dataset.vl_cmu_cd import get_VL_CMU_CD, get_VL_CMU_CD_Raw, get_VL_CMU_CD_total
from dataset.pcd import get_TSUNAMI, get_PSCD, get_PSCD_total, get_TSUNAMI_total

dataset_dict = {
    "VL_CMU_CD": get_VL_CMU_CD,
    'VL_CMU_CD_Raw': get_VL_CMU_CD_Raw,
    'PSCD': get_PSCD,
    'TSUNAMI': get_TSUNAMI,
    'PSCD_total': get_PSCD_total,
    'TSUNAMI_total': get_TSUNAMI_total,
    'VL_CMU_CD_total': get_VL_CMU_CD_total,
}
