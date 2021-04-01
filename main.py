import torch
from torch.nn import *

from utils import get_count, np_tensor
from data import get_study_data, get_dataloaders

def init():
    part_of_body = 'WRIST'
    phase_cat = ['train', 'valid']

    case_data = get_study_data('XR_' + part_of_body, 'D:/MURA-v1.1/{0}/{1}/')
    dataloaders = get_dataloaders(case_data, batch_size=1)

    tai = {x: get_count(case_data[x], 'positive') for x in phase_cat}
    tni = {x: get_count(case_data[x], 'negative') for x in phase_cat}
    Wt1 = {x: np_tensor(tni[x] / (tni[x] + tai[x])) for x in phase_cat}
    Wt0 = {x: np_tensor(tai[x] / (tni[x] + tai[x])) for x in phase_cat}


class Proc():
    class Loss(torch.nn.Module):
        def __init__(self, Wt1, Wt0):
            super(self.Loss, self).__init__()
            self.Wt1 = Wt1
            self.Wt0 = Wt0

        def forward(self, inputs, targets, phase):
            loss = - (self.Wt1[phase] * targets * inputs.log() + self.Wt0[phase] * (1-targets)*(1-inputs).log())
            return loss



if __name__ == '__main__':
    init()
    Proc()
    """
    filename = 'MURA.csv'
    data.CuntomDataset(filename)
    """