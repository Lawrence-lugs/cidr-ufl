#%%

from collections import OrderedDict
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10

import cidr_node, cidr_utils
import flwr as fl
from flwr.common import Metrics

import numpy as np
import os

DEVICE = torch.device("cuda")  # Try "cuda" to train on GPU
print(
    f"Training on {DEVICE} using PyTorch {torch.__version__} and Flower {fl.__version__}"
)

def split_dataset(trainset: torch.utils.data.dataset.Dataset,num_nodes: int):

    torchseed = torch.Generator().manual_seed(42)
    dataset_shares = [1 / num_nodes] * num_nodes

    local_trainsets = torch.utils.data.random_split(trainset, dataset_shares, torchseed)

    return local_trainsets 


def dp_node_creator(cid: str, trainset_list, local_learning_epochs = 10) -> cidr_node.dl_node:
    '''
    Creates a flower client for the server to play with

    The server uses this function to create a client on-demand whenever it
    needs one.

    TODO | We can probably add some metadata like remaining energy and 
    routing information on each node to dl_node(), save it to a pickle, and
    read it everytime client_fn is called here to load it to the new node object.

    For DL Framework: creates a dl_node (which inherits as a flower client object)
    '''
    node_file = f'node_states/node_{cid}.nd'
    print(f'Checking if {node_file} exists...')
    if os.path.exists(node_file):
        print('DP Node Creator: Loading DL node...')
        node = cidr_node.dl_node.load_node(cid)
    else:
        print('DP Node Creator: Creating DL node...')
        node = cidr_node.dl_node(cid)
        node.dp_model.train_set = trainset_list[int(cid)]
        node.dp_model.load_loaders()
        node.dp_model.learning_epochs = local_learning_epochs
        
    node.dp_model.epoch = 0 #reset current epoch to 0 to restart training

    print(f'DP Node Creator: Successfully created {node}')

    return node

if __name__ == '__main__':

    #only compatible with linux
    os.system('rm node_states/*')

    CIFARMAIN_train_set = torchvision.datasets.CIFAR10(
            root='data',
            train=True,
            download=False,
            transform=cidr_utils.t_cropflip_augment
        )
    CIFARMAIN_test_set = torchvision.datasets.CIFAR10(
            root='data',
            train=True,
            download=False,
            transform=cidr_utils.t_normalize
        )
    
    num_nodes = 2

    local_trainsets = split_dataset(CIFARMAIN_train_set,num_nodes)

    # Specify client resources if you need GPU (defaults to 1 CPU and 0 GPU)
    client_resources = None
    if DEVICE.type == "cuda":
        client_resources = {"num_gpus": 1}

    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,  # Sample 100% of available clients for training
        fraction_evaluate=0.5,  # Sample 50% of available clients for evaluation
        min_fit_clients=2,  # Never sample less than 10 clients for training
        min_evaluate_clients=2,  # Never sample less than 5 clients for evaluation
        min_available_clients=2,  # Wait until all 10 clients are available
        )

    import functools
    client_func = functools.partial(dp_node_creator,trainset_list = local_trainsets, local_learning_epochs = 2)

    # Start simulation
    fl.simulation.start_simulation(
        client_fn=client_func,
        num_clients=num_nodes,
        config=fl.server.ServerConfig(num_rounds=10),
        strategy=strategy,
        client_resources=client_resources,
    )

    print("Finished simulation")

# %%
