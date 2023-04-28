import torch
from torch.ao.quantization import QuantStub, DeQuantStub
from cidr_utils import *
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

config = {
    'quantization_api' : 'pytorch_fxgraph', # pytorch_fxgraph, pytorch_eager
    'quantization_mode' : 'static', # static, dynamic, weights
} 

def sup_test(model):
    total_correct=0
    for inputs,labels in test_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs,1)
        total_correct += torch.sum(preds == labels.data)
    acc = 100*total_correct/len(test_data)
    return acc

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

device = torch.device('cuda')
train_data = datasets.CIFAR10(root='data', train=True, download=True, transform=transform_train)
test_data = datasets.CIFAR10(root="data", train=False, download=True, transform=transform_test)

n_workers = 2
batch_size = 16
train_loader = DataLoader(train_data,batch_size=batch_size,num_workers=n_workers)
test_loader = DataLoader(test_data,batch_size,num_workers=n_workers)
criterion = torch.nn.CrossEntropyLoss()

oldModel = load_model('jitted_runs/mbv2_kl_two2stride_relu6').to('cuda') 
old_acc = sup_test(oldModel)
print_size_of_model(oldModel)
print(f'Float acc:{old_acc}')

device = torch.device('cpu')

myModel = load_model('jitted_runs/mbv2_kl_two2stride_relu6').to('cpu')
myModel.eval()
myModel.fuse_model()

myModel.qconfig = torch.ao.quantization.default_qconfig
print(myModel.qconfig)

print('PTQ Prepare: Inserting Observers')
torch.ao.quantization.prepare(myModel, inplace=True)

num_calibration_batches = 32
evaluate(myModel, criterion, test_loader, neval_batches=num_calibration_batches)
print('PTQ: Calibration Done')

torch.ao.quantization.convert(myModel,inplace=True)
print('PTQ: Convert done')

print('Size of model after PTQ')
print_size_of_model(myModel)

accuracy = sup_test(myModel)
print(f'PTQ acc @ INT8 : {accuracy}')


