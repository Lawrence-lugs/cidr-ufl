import dl_framework

# Name the Run

my_config = dl_framework.fw_config()

my_config.tensorboard_runs_dir = 'tb_data/ks_many_nodes'
my_config.run_name = 'fed_ks'

# Federated Simulation Settings 
# my_config.resume = False
my_config.clients_per_gpu = 1
my_config.num_nodes = 30
my_config.local_epochs = 8
my_config.federated = True
my_config.num_rounds = 8

# Set Node Class

from keyword_spotting.ks_node import ks_node
my_config.node_class = ks_node

# Make Local Datasets
import torch
def split_dataset(trainset: torch.utils.data.dataset.Dataset,num_nodes: int):

    torchseed = torch.Generator().manual_seed(42)
    dataset_shares = [1 / num_nodes] * num_nodes

    local_trainsets = torch.utils.data.random_split(trainset, dataset_shares, torchseed)

    return local_trainsets 

from keyword_spotting.ks_dset import mlperftiny_ks_dset
trainset = mlperftiny_ks_dset(set='train',root='/home/raimarc/lawrence-workspace/data/mlperftiny_ks_dset')
testset = mlperftiny_ks_dset(set='test',root='/home/raimarc/lawrence-workspace/data/mlperftiny_ks_dset')

trainsets = split_dataset(trainset,num_nodes=my_config.num_nodes)

my_config.testset = testset
my_config.trainsets = trainsets

my_config.savename = "final_models/ks_final"

import dl_framework.framework as framework
framework.run(my_config)


