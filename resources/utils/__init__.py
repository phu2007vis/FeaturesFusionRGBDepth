
from resources.utils.augumentaion import *
from resources.utils.visualize import *
import torch.nn as nn




def duplicate(value,time):
    if isinstance(value,list):
        assert len(value) == time
        return value
    else:
        return [value]*time
def get_activation(name):
    if name == "ReLU":
        return nn.ReLU()
    elif name == "SiLU":
        return nn.SiLU()