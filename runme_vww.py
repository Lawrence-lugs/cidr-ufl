import dl_framework

# Name the Run

my_config = dl_framework.fw_config()

my_config.tensorboard_runs_dir = 'tb_data/vww_many_nodes'
my_config.run_name = 'fed_vww'

# Federated Simulation Settings 
# my_config.resume = False
my_config.clients_per_gpu = 1
my_config.num_nodes = 30
my_config.local_epochs = 10
my_config.federated = True
my_config.num_rounds = 10

# Set Node Class

from visual_wakewords import vww_node
my_config.node_class = vww_node.vww_node

# Make Local Datasets
import torch
def split_dataset(trainset: torch.utils.data.dataset.Dataset,num_nodes: int):

    torchseed = torch.Generator().manual_seed(42)
    dataset_shares = [1 / num_nodes] * num_nodes

    local_trainsets = torch.utils.data.random_split(trainset, dataset_shares, torchseed)

    return local_trainsets 

import pyvww
from torchvision import transforms

trainset = pyvww.pytorch.VisualWakeWordsClassification(
    root='/home/raimarc/lawrence-workspace/data/MSCOCO/all2014',
    annFile='/home/raimarc/lawrence-workspace/data/visualwakewords/annotations/instances_train.json',
    transform = transforms.Compose([
        transforms.Resize((96,96)),
        transforms.RandomCrop(96,padding=12),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ToTensor(),
        transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225)),
    ])
)
trainsets = split_dataset(trainset,my_config.num_nodes)
trainset = None
testset = pyvww.pytorch.VisualWakeWordsClassification(
    root='/home/raimarc/lawrence-workspace/data/MSCOCO/all2014',
    annFile='/home/raimarc/lawrence-workspace/data/visualwakewords/annotations/instances_val.json',
    transform = transforms.Compose([
        transforms.Resize((96,96)),
        transforms.ToTensor(),
        transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))
    ])
)

my_config.testset = testset
my_config.trainsets = trainsets

import dl_framework.framework as framework
framework.run(my_config)



