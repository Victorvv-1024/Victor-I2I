from models.cut_seg import CUT_SEG_model
from models.kx_cut_seg import KX_CUT
from models.cycleGAN import CycleGAN
from models.dcl_model import DCLModel
from models.distance_model import DistanceGAN
from models.ag_cycle import AG_Cycle
from models.ag_cut import AG_Cut


def create_model(opt):
    if opt.model == 'cyclegan':
        model = CycleGAN(opt)
    elif opt.model == 'cut_seg':
        model = CUT_SEG_model(opt)
    elif opt.model == 'kx_cut':
        model = KX_CUT(opt)
    elif opt.model == 'dcl':
        model = DCLModel(opt)
    elif opt.model == 'distance':
        model = DistanceGAN(opt)
    elif opt.model == 'ag_cut':
        model = AG_Cut(opt)
    elif opt.model == 'ag_cycle':
        model = AG_Cycle(opt)

    print(f'model: {opt.model} is created.')

    return model