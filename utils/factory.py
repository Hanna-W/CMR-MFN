from models.finetune import Finetune
from models.replay import Replay
# from models.myfinetune import MyFinetune
# from models.ARTF import ARTF
from models.cmr_mfn_wo_mixup import CMR_MFN_wo_Mixup
from models.baseline import Baseline
# from models.ARTF_R import CMDF_R
from models.CMR_MFN import CMR_MFN


def get_model(model_name, args):
    name = model_name.lower()
    if name == "finetune":
        return Finetune(args)
    elif name == "replay":
        return Replay(args)
    # elif name == "myfinetune":
    #     return MyFinetune(args)
    # elif name == 'cmdf':
    #     return ARTF(args)
    elif name == 'cmr_mfn_wo_mixup':
        return CMR_MFN_wo_Mixup(args)
    elif name == 'baseline':
        return Baseline(args)
    elif name == 'cmr_mfn':
        return CMR_MFN(args)

    else:
        assert 0
