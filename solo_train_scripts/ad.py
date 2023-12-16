#%%
from anomaly_detection import ad_node
import dl_framework

my_config = dl_framework.fw_config()
my_config.federated = False

from anomaly_detection import toycar_dset as tc
trainset = tc.toycar_dataset(set='train')
testset = tc.toycar_dataset(set='test')

node = ad_node.toycar_ad_node(my_config)

import torch.optim

# Set it to Adam for quick training this time
node.dp_model.optimizer = torch.optim.Adam(node.dp_model.model.parameters(),lr=0.001)

node.dp_model.test_set = testset
node.dp_model.train_set = trainset
node.dp_model.learning_epochs = 100
node.dp_model.load_loaders()

node.dp_model.sup_train()
from torch import save
save(node.dp_model.model,'final_models/ad_final')

#%%

import torch.onnx
node.dp_model.model.eval()
torch.onnx.export(node.dp_model.model,next(iter(node.dp_model.test_loader))[0].to(node.dp_model.device),"final_modes/ad_raw.onnx")

# %%
node.dp_model.model.eval()

test_in = next(iter(node.dp_model.test_loader))[0][0].to(node.dp_model.device)
test_out = node.dp_model.model(test_in)
node.dp_model.criterion(test_out,test_in)

import torch.onnx
torch.onnx.export(node.dp_model.model,test_in,"final_models/ad_raw.onnx")
