#%%
from keyword_spotting.ks_node import ks_node
import dl_framework

my_config = dl_framework.fw_config()
my_config.federated = False

from keyword_spotting.ks_dset import mlperftiny_ks_dset
my_config.trainset = mlperftiny_ks_dset(set='train',root='/home/raimarc/lawrence-workspace/data/mlperftiny_ks_dset')
my_config.testset = mlperftiny_ks_dset(set='test',root='/home/raimarc/lawrence-workspace/data/mlperftiny_ks_dset')

node = ks_node(my_config)
node.dp_model.test_set = my_config.testset
node.dp_model.train_set = my_config.trainset
node.dp_model.learning_epochs = 100
node.dp_model.load_loaders()

import torch
node.dp_model.model = torch.load('final_models/ks_final')

test_in = next(iter(node.dp_model.test_set))[0].to(node.dp_model.device).permute(2,0,1).unsqueeze(0)
test_out = node.dp_model.model(test_in)

torch.onnx.export(node.dp_model.model,test_in,"final_models/ks_raw.onnx")

# %%
