

# Sample Configuration File
# Tensorboard Naming Information

tensorboard_runs_dir = 'tb_data/unspecified'
run_name = 'default'

# Simulation Info 
resume = False
clients_per_gpu = 1
num_nodes = 2
dataset = 'ToyCar'
local_epochs = 10
federated = True

def get_node_class():
    from anomaly_detection.ad_node import toycar_ad_node
    return toycar_ad_node

import torch
def split_dataset(trainset: torch.utils.data.dataset.Dataset,num_nodes: int):

    torchseed = torch.Generator().manual_seed(42)
    dataset_shares = [1 / num_nodes] * num_nodes

    local_trainsets = torch.utils.data.random_split(trainset, dataset_shares, torchseed)

    return local_trainsets 

if federated:

    if dataset=='ToyCar':
        from anomaly_detection import toycar_dset as tc
        trainsets = [tc.toycar_dataset(set='train') for i in range(num_nodes)]

        local_dataset_size = int(len(trainsets[0].files)/num_nodes)
        import numpy as np
        indices = np.random.randint(len(trainsets[0].files),size=len(trainsets[0].files))
        for i,set in enumerate(trainsets):
            file_subset = []
            for idx in indices[local_dataset_size*i:local_dataset_size*(i+1)]:
                file_subset.append(set.files[idx])
            set.files = file_subset

        testset = tc.toycar_dataset(set='test')
        from numpy.random import randint
        indices = randint(len(testset),size=500)
        file_subset = []
        for idx in indices:
            file_subset.append(testset.files[idx])
        testset.files = file_subset

    if dataset=='CIFAR10':
        import torchvision
        import image_classification.utils
        trainset = torchvision.datasets.CIFAR10(
                root='data',
                train=True,
                download=False,
                transform=image_classification.utils.t_cropflip_augment
            )
        trainsets = split_dataset(trainset,num_nodes)
        trainset = None
        testset = torchvision.datasets.CIFAR10(
                root='data',
                train=False,
                download=False,
                transform=image_classification.utils.t_normalize
            )

    if dataset=='VWW':
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
        trainsets = split_dataset(trainset,num_nodes)
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