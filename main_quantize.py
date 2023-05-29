
import cidr_models, cidr_utils
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import argparse

def load_vww(dpmodel):
    import pyvww
    print("Loading visual wakewords as the dataset...")
    dpmodel.train_set = pyvww.pytorch.VisualWakeWordsClassification(
        root='/home/raimarc/lawrence-workspace/MSCOCO/all2014',
        annFile='/home/raimarc/lawrence-workspace/visualwakewords/annotations/instances_train.json',
        transform = transforms.Compose([
            transforms.Resize((96,96)),
            transforms.RandomCrop(96,padding=12),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ToTensor(),
            transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225)),
        ])
    )
    dpmodel.test_set = pyvww.pytorch.VisualWakeWordsClassification(
        root='/home/raimarc/lawrence-workspace/MSCOCO/all2014',
        annFile='/home/raimarc/lawrence-workspace/visualwakewords/annotations/instances_val.json',
        transform = transforms.Compose([
            transforms.Resize((96,96)),
            transforms.ToTensor(),
            transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))
        ])
    )
    dpmodel.load_loaders()

def quantization_test(dpmodel):
    print(dpmodel.test())
    cidr_utils.print_size_of_model(dpmodel.model)
    print(cidr_utils.count_params(dpmodel.model))

    dpmodel.quantize()
    print(dpmodel.test())
    cidr_utils.print_size_of_model(dpmodel.model)
    print(cidr_utils.count_params(dpmodel.model))

if __name__ == '__main__':

    config = {
        'runname': 'mbv2_vww_rcrop_w0_35'
    }
    
    torch.set_seed = 0

    import cidr_node

    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(f'runs/{config["runname"]}')

    mbv2_dp = cidr_node.dp_model()
    mbv2_dp.tb_writer = writer
    
    from mobilenetv2 import MobileNetV2

    mbv2_dp.model = MobileNetV2(
        num_classes = 2,
        width_mult = 0.35
    ).to(mbv2_dp.device)

    mbv2_dp.optimizer = torch.optim.SGD(
        mbv2_dp.model.parameters(),
        lr = 0.002,
        momentum = 0.9,
        weight_decay = 1e-4
    )

    load_vww(mbv2_dp)

    mbv2_dp.load_params('jitted_runs/mbv2_vww_rcrop_w0_35.pth')
    print(mbv2_dp.test())
    cidr_utils.print_size_of_model(mbv2_dp.model)
    breakpoint()
    mbv2_dp.quantize()
    print(mbv2_dp.test())
    cidr_utils.print_size_of_model(mbv2_dp.model)

    save = False
    print(f'Current config runname: {config["runname"]}')
    breakpoint()
    if(save is True):
        torch.save(mbv2_dp.model.state_dict(),f'jitted_runs/{config["runname"]}.pth')

    
# %%
