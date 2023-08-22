
import dl_framework.node
import image_classification.mbv2

import torch

class vww_node(dl_framework.node.dl_node):
    '''
    A dl node running image classification on visual wakewords
    '''
    def __init__(self, fw_config, name='default_name'):
        super(vww_node, self).__init__(fw_config,name)

        self.device = fw_config.get_device()

        self.dp_model = dl_framework.node.dp_model(fw_config)
        self.dp_model.model = image_classification.mbv2.KL_MBV2_forVWW().to(self.device)
        self.net = self.dp_model.model
        self.dp_model.name = name
        self.name = name
        self.energy = 500
        self.round = 0

        # NEED TO SET OPTIMIZER TO TARGET THE NEW MODEL AS WELL
        self.dp_model.optimizer = torch.optim.SGD(
            self.dp_model.model.parameters(),
            lr = 0.002,
            momentum = 0.9,
            weight_decay = 1e-4
        )