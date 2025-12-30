import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.models import SqueezeNet


from module_1_4 import helper_utils

dataset = helper_utils.get_dataset()
transform = transforms.ToTensor()
dataset.transform = transform