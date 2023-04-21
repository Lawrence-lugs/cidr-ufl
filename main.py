
import cidr_models, cidr_utils
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import argparse

device = torch.device('cuda' if torch.cuda.is_available else 'cpu')

ToTensor = transforms.ToTensor()

train_data = datasets.CIFAR10(root='data', train=True, download=True, transform=ToTensor)
test_data = datasets.CIFAR10(root="data", train=False, download=True, transform=ToTensor)

n_workers = 0
batch_size = 16
train_loader = DataLoader(train_data,batch_size=batch_size,num_workers=n_workers)
test_loader = DataLoader(test_data,batch_size,num_workers=n_workers)

from tensorboardX import SummaryWriter
writer = SummaryWriter()

def sup_train(model,epochs,resumefrom=0,checkpoint=None):    
    
    optimizer = torch.optim.Adam(model.parameters(),lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=60, gamma=0.1)

    if resumefrom != 0:
        optimizer.load_state_dict(checkpoint['opt'])
        print(f'Restarting from Epoch {resumefrom}...')

    for epoch in range( epochs - resumefrom ):

        score = 0
        model.train()
        for inputs,labels in train_loader:
            model.train()
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs,labels)

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

    resume = False
    
    mbv2 = cidr_models.CIDR_MobileNetv2()

    epoch = 0
    checkpoint=None
    if resume == True:
        print("Resuming from old checkpoint...")
        checkpoint = torch.load('saved_runs/lastrun')
        mbv2.load_state_dict(checkpoint['model'])
        epoch = checkpoint['epoch']

    net = mbv2.to(device)

    print('Starting...')
    sup_train(net,200,resumefrom=epoch,checkpoint=checkpoint)

    breakpoint()

# %%
