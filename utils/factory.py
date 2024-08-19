from models.CMR_MFN import CMR_MFN

def get_model(model_name, args):
    name = model_name.lower()
    if name == 'cmr_mfn':
        return CMR_MFN(args)

    else:
        assert 0
