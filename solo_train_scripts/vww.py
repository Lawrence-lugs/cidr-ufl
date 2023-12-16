#%%
import dl_framework

my_config = dl_framework.fw_config()
my_config.federated = False

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
testset = pyvww.pytorch.VisualWakeWordsClassification(
    root='/home/raimarc/lawrence-workspace/data/MSCOCO/all2014',
    annFile='/home/raimarc/lawrence-workspace/data/visualwakewords/annotations/instances_val.json',
    transform = transforms.Compose([
        transforms.Resize((96,96)),
        transforms.ToTensor(),
        transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))
    ])
)

from visual_wakewords import vww_node

node = vww_node.vww_node(my_config)
node.dp_model.test_set = testset
node.dp_model.train_set = trainset
node.dp_model.learning_epochs = 100
node.dp_model.load_loaders()

node.dp_model.sup_train()

#%%

import torch
torch.save(node.dp_model.model,'final_models/vww_final')

#%%

test_in = next(iter(node.dp_model.test_set))[0].to(node.dp_model.device).unsqueeze(0)
test_out = node.dp_model.model(test_in)

torch.onnx.export(node.dp_model.model,test_in,"final_models/vww_raw.onnx")

# %%
