import torch
from torch.ao.quantization import QuantStub, DeQuantStub
from cidr_utils import *
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from train import *
from config import config

device = torch.device('cpu')

myModel = load_model('jitted_runs/mbv2_kl_two2stride_relu6').to('cpu')
myModel.eval()
#myModel.fuse_model()

myModel.qconfig = torch.ao.quantization.default_qconfig
print(myModel.qconfig)

print('PTQ Prepare: Inserting Observers')
torch.ao.quantization.prepare(myModel, inplace=True)

num_calibration_batches = 32
evaluate(myModel, test_loader, neval_batches=num_calibration_batches)
print('PTQ: Calibration Done')

torch.ao.quantization.convert(myModel,inplace=True)
print('PTQ: Convert done')

print('Size of model after PTQ')
print_size_of_model(myModel)

accuracy = sup_test(myModel,device=torch.device('cpu'))
print(f'PTQ acc @ INT8 : {accuracy}')


