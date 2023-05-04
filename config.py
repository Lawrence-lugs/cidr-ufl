import cidr_models
import torch

config = {
    'runname': 'test',
    'resume' : False,
    'model' : cidr_models.KL_MBV2(),
    'criterion': torch.nn.CrossEntropyLoss()
}