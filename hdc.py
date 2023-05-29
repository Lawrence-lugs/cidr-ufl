#%%

import torch,torchhd
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.datasets import MNIST
import torchmetrics

from torchhd import functional
from torchhd import embeddings

transform = torchvision.transforms.ToTensor()