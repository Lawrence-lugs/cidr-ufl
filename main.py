
import cidr_models, cidr_utils
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import argparse
from train import *

if __name__ == '__main__':

    config = {
        'runname': 'original_mbv2_impl_testing'
    }
    
    torch.set_seed = 0

    import cidr_node

    from tensorboardX import SummaryWriter
    writer = SummaryWriter(f'runs/{config["runname"]}')

    mbv2_dp = cidr_node.dp_model()
    mbv2_dp.tb_writer = writer
    
    from mobilenetv2 import MobileNetV2

    mbv2_dp.model = MobileNetV2(
        num_classes = 10,
        width_mult = 1.0
    ).to(mbv2_dp.device)

    mbv2_dp.optimizer = torch.optim.SGD(
        mbv2_dp.model.parameters(),
        lr = 0.002,
        momentum = 0.9,
        weight_decay = 1e-4
    )

    mbv2_dp.train()

    breakpoint()

# %%
