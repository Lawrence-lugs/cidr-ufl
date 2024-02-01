#%%
from image_classification.ic_node import ic_node
import dl_framework

my_config = dl_framework.fw_config()
my_config.federated = False

import image_classification, torchvision
trainset = torchvision.datasets.CIFAR10(
        root='data',
        train=True,
        download=True,
        transform=image_classification.utils.t_cropflip_augment
    )
testset = torchvision.datasets.CIFAR10(
        root='data',
        train=False,
        download=True,
        transform=image_classification.utils.t_normalize
    )


node = ic_node(my_config)
node.dp_model.test_set = testset
node.dp_model.train_set = trainset
node.dp_model.learning_epochs = 100
node.dp_model.load_loaders()

node.dp_model.sup_train()
from torch import save
save(node.dp_model.model,'final_models/ic_final')
# %%
