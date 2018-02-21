import torch
import numpy as np
from properties import *


def to_tensor(array):
    return torch.from_numpy(np.array(array)).float()


def to_variable(tensor):
    if torch.cuda.is_available():
        tensor = tensor.cuda()
    return torch.autograd.Variable(tensor)

