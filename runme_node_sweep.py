import dl_framework

### IMAGE CLASSIFICATION 
lcl_epch = 2

# Name the Run

my_config = dl_framework.fw_config()

my_config.tensorboard_runs_dir = 'tb_data/momserv/fed_ic_momfedavg'
my_config.run_name = f'fed_ic_lcl_{lcl_epch}'

# Federated Simulation Settings 
# my_config.resume = False
TOTAL_EPOCHS = 200
my_config.clients_per_gpu = 1
my_config.num_nodes = 10
my_config.local_epochs = lcl_epch
my_config.federated = True
my_config.num_rounds = TOTAL_EPOCHS//lcl_epch

# Set Node Class

import image_classification.ic_node
my_config.node_class = image_classification.ic_node.ic_node

# Make Local Datasets
import torch
print(torch.__version__)
def split_dataset(trainset: torch.utils.data.dataset.Dataset,num_nodes: int):

    torchseed = torch.Generator().manual_seed(42)
    dataset_shares = [1 / num_nodes] * num_nodes

    import numpy as np
    print(np.sum(dataset_shares))

    local_trainsets = torch.utils.data.random_split(trainset, dataset_shares, torchseed)

    return local_trainsets 

import torchvision
import image_classification.utils
trainset = torchvision.datasets.CIFAR10(
        root='data',
        train=True,
        download=False,
        transform=image_classification.utils.t_cropflip_augment
    )
trainsets = split_dataset(trainset,my_config.num_nodes)
trainset = None
testset = torchvision.datasets.CIFAR10(
        root='data',
        train=False,
        download=False,
        transform=image_classification.utils.t_normalize
    )

my_config.testset = testset
my_config.trainsets = trainsets

import dl_framework.framework as framework
framework.run(my_config)

### ANOMALY DETECTION

# Name the Run

my_config = dl_framework.fw_config()

my_config.tensorboard_runs_dir = 'tb_data/momserv/fed_ad_momfedavg'
my_config.run_name = f'fed_ad_lcl_{lcl_epch}'

# Federated Simulation Settings 
# my_config.resume = False
TOTAL_EPOCHS = 200
my_config.clients_per_gpu = 1
my_config.num_nodes = 10
my_config.local_epochs = lcl_epch
my_config.federated = True
my_config.num_rounds = TOTAL_EPOCHS//lcl_epch

# Set Node Class

import anomaly_detection.ad_node
my_config.node_class = anomaly_detection.ad_node.toycar_ad_node

# Make Local Datasets

from anomaly_detection import toycar_dset as tc
trainsets = [tc.toycar_dataset(set='train') for i in range(my_config.num_nodes)]
testset = tc.toycar_dataset(set='test')

local_dataset_size = int(len(trainsets[0].files)/my_config.num_nodes)
import numpy as np
indices = np.random.randint(len(trainsets[0].files),size=len(trainsets[0].files))
for i,set in enumerate(trainsets):
    file_subset = []
    for idx in indices[local_dataset_size*i:local_dataset_size*(i+1)]:
        file_subset.append(set.files[idx])
    set.files = file_subset
print(len(testset))

my_config.testset = testset
my_config.trainsets = trainsets

import dl_framework.framework as framework
framework.run(my_config)

n_local_epochs = [4,8]

for lcl_epch in n_local_epochs:

    ### KEYWORD SPOTTING

    # Name the Run

    my_config = dl_framework.fw_config()

    my_config.tensorboard_runs_dir = 'tb_data/momserv/fed_ks_momfedavg'
    my_config.run_name = f'fed_ks_lcl_{lcl_epch}'

    # Federated Simulation Settings 
    # my_config.resume = False

    TOTAL_EPOCHS = 128
    my_config.clients_per_gpu = 1
    my_config.num_nodes = 10
    my_config.local_epochs = lcl_epch
    my_config.federated = True
    my_config.num_rounds = TOTAL_EPOCHS // my_config.local_epochs

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

    import dl_framework.framework as framework
    framework.run(my_config)


    ### IMAGE CLASSIFICATION 

    # Name the Run

    my_config = dl_framework.fw_config()

    my_config.tensorboard_runs_dir = 'tb_data/momserv/fed_ic_momfedavg'
    my_config.run_name = f'fed_ic_lcl_{lcl_epch}'

    # Federated Simulation Settings 
    # my_config.resume = False
    TOTAL_EPOCHS = 200
    my_config.clients_per_gpu = 1
    my_config.num_nodes = 10
    my_config.local_epochs = lcl_epch
    my_config.federated = True
    my_config.num_rounds = TOTAL_EPOCHS//lcl_epch
    
    # Set Node Class

    import image_classification.ic_node
    my_config.node_class = image_classification.ic_node.ic_node

    # Make Local Datasets
    import torch
    print(torch.__version__)
    def split_dataset(trainset: torch.utils.data.dataset.Dataset,num_nodes: int):

        torchseed = torch.Generator().manual_seed(42)
        dataset_shares = [1 / num_nodes] * num_nodes

        import numpy as np
        print(np.sum(dataset_shares))

        local_trainsets = torch.utils.data.random_split(trainset, dataset_shares, torchseed)

        return local_trainsets 

    import torchvision
    import image_classification.utils
    trainset = torchvision.datasets.CIFAR10(
            root='data',
            train=True,
            download=False,
            transform=image_classification.utils.t_cropflip_augment
        )
    trainsets = split_dataset(trainset,my_config.num_nodes)
    trainset = None
    testset = torchvision.datasets.CIFAR10(
            root='data',
            train=False,
            download=False,
            transform=image_classification.utils.t_normalize
        )

    my_config.testset = testset
    my_config.trainsets = trainsets

    import dl_framework.framework as framework
    framework.run(my_config)

    ### ANOMALY DETECTION

    # Name the Run

    my_config = dl_framework.fw_config()

    my_config.tensorboard_runs_dir = 'tb_data/momserv/fed_ad_momfedavg'
    my_config.run_name = f'fed_ad_lcl_{lcl_epch}'

    # Federated Simulation Settings 
    # my_config.resume = False
    TOTAL_EPOCHS = 200
    my_config.clients_per_gpu = 1
    my_config.num_nodes = 10
    my_config.local_epochs = lcl_epch
    my_config.federated = True
    my_config.num_rounds = TOTAL_EPOCHS//lcl_epch

    # Set Node Class

    import anomaly_detection.ad_node
    my_config.node_class = anomaly_detection.ad_node.toycar_ad_node

    # Make Local Datasets

    from anomaly_detection import toycar_dset as tc
    trainsets = [tc.toycar_dataset(set='train') for i in range(my_config.num_nodes)]
    testset = tc.toycar_dataset(set='test')

    local_dataset_size = int(len(trainsets[0].files)/my_config.num_nodes)
    import numpy as np
    indices = np.random.randint(len(trainsets[0].files),size=len(trainsets[0].files))
    for i,set in enumerate(trainsets):
        file_subset = []
        for idx in indices[local_dataset_size*i:local_dataset_size*(i+1)]:
            file_subset.append(set.files[idx])
        set.files = file_subset
    print(len(testset))
    # from numpy.random import randint
    # indices = randint(len(testset),size=500)
    # file_subset = []
    # for idx in indices:
    #     file_subset.append(testset.files[idx])
    # testset.files = file_subset

    my_config.testset = testset
    my_config.trainsets = trainsets

    import dl_framework.framework as framework
    framework.run(my_config)





