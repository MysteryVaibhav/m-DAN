import torch
from model import mDAN

model_1 = mDAN()
model_1.load_state_dict(torch.load('model_weights.t7'))

model_2 = mDAN()
model_2.load_state_dict(torch.load('model_weights_1.t7'))
print(".")
