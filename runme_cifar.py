import dl_framework

nums = [10]

# Name the Run

for i in nums:

    my_config = dl_framework.fw_config()

    my_config.tensorboard_runs_dir = 'tb_data/cifar_210nodes_resnet'
    my_config.run_name = f'fed_ic_{i}'

    # Federated Simulation Settings 
    # my_config.resume = False
    my_config.clients_per_gpu = 1
    my_config.num_nodes = i
    my_config.local_epochs = 10
    my_config.federated = True
    my_config.num_rounds = 10

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



