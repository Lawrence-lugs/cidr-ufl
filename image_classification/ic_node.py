#%%

import dl_framework.node
from image_classification import resnet
import torch
from dl_framework import fw_config

class ic_node(dl_framework.node.dl_node):

    def __init__(self, fw_config, name='default_name'):
        super(ic_node,self).__init__(fw_config, name)

        self.dp_model.model = resnet.MLPerfTiny_ResNet_Baseline(10).to(self.device)
        self.dp_model.optimizer = torch.optim.SGD(
            self.dp_model.model.parameters(),
            lr = 0.1, #set to 0.01 for momentum server, 0.1 for normal
            # momentum = 0.9,
            weight_decay = 1e-4
        )
        self.net = self.dp_model.model
        self.dp_model.scheduler = torch.optim.lr_scheduler.StepLR(self.dp_model.optimizer,60,0.1)

if __name__ == '__main__':
    my_config = fw_config()
    my_config.tensorboard_runs_dir = 'tb_data/ic_node'
    my_config.run_name = 'long_run_1'
    node = ic_node(my_config)
    node.dp_model.load_loaders()
    node.dp_model.sup_train()

