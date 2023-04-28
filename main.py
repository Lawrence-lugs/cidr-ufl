
import cidr_models, cidr_utils
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import argparse

device = torch.device('cuda' if torch.cuda.is_available else 'cpu')

print("Preparing data...")

config = {
    'runname': 'mbv2_kl_two2stride_relu6',
    'resume' : True,
    'model' : cidr_models.KL_MBV2()
}


# Data augments
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


train_data = datasets.CIFAR10(root='data', train=True, download=True, transform=transform_train)
test_data = datasets.CIFAR10(root="data", train=False, download=True, transform=transform_test)

n_workers = 2
batch_size = 16
train_loader = DataLoader(train_data,batch_size=batch_size,num_workers=n_workers)
test_loader = DataLoader(test_data,batch_size,num_workers=n_workers)

from tensorboardX import SummaryWriter
writer = SummaryWriter(f'runs/{config["runname"]}')

def sup_train(model,epochs,resumefrom=0,checkpoint=None,opt='adam'):    
    
    if opt=='adam':
        optimizer = torch.optim.Adam(model.parameters(),lr=0.001)
    if opt=='sgd':
        optimizer = torch.optim.SGD(net.parameters(), lr=0.002, momentum=0.9, weight_decay=1e-4)
    
    criterion = torch.nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)

    if resumefrom != 0:
        optimizer.load_state_dict(checkpoint['opt'])
        print(f'Restarting from Epoch {resumefrom}...')

    for epoch in range( epochs - resumefrom ):

        score = 0
        runloss = 0
        model.train()
        for inputs,labels in train_loader:
            model.train()
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs,labels)
            runloss += loss

            _, preds = torch.max(outputs,1)
            score += torch.sum(preds == labels.data)

            loss.backward()
            optimizer.step()
            
        model.eval()
        #scheduler.step()
        with torch.no_grad():  
            accu = sup_test(model)
            epoch = epoch + resumefrom 
        print(f'epoch {epoch}: {accu}')
        writer.add_scalar('data/loss',runloss,epoch)
        writer.add_scalar('data/accu',accu,epoch)
        writer.add_scalar('data/trainacc',score/len(train_data),epoch)
        cidr_utils.save_progress('lastrun',epoch,model.state_dict(),optimizer.state_dict())

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

def get_parser():
    parser = argparse.ArgumentParser(description="CIDR_sims")
    parser.add_argument("--resume",type=bool,default=True)
    return parser

if __name__ == '__main__':

    torch.set_seed = 0
    
    mbv2 = config["model"]

    epoch = 0
    checkpoint=None
    if config["resume"] == True:
        print("Resuming from old checkpoint...")
        checkpoint = torch.load(f'saved_runs/{config["runname"]}')
        mbv2.load_state_dict(checkpoint['model'])
        epoch = checkpoint['epoch']

    net = mbv2.to(device)

    print('Starting...')
    sup_train(net,200,resumefrom=epoch,checkpoint=checkpoint,opt='sgd')

    breakpoint()

# %%
