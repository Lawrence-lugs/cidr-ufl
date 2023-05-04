import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import cidr_utils
from config import config

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# train_data = datasets.CIFAR10(root='data', train=True, download=False, transform=cidr_utils.t_cropflip_augment)
# test_data = datasets.CIFAR10(root="data", train=False, download=False, transform=cidr_utils.t_normalize)

# n_workers = 2
# batch_size = 16
# train_loader = DataLoader(train_data,batch_size=batch_size,num_workers=n_workers)
# test_loader = DataLoader(test_data,batch_size,num_workers=n_workers)

from tensorboardX import SummaryWriter
writer = SummaryWriter(f'runs/{config["runname"]}')

def sup_train(model,epochs,resumefrom=0,checkpoint=None,opt='adam'):    
    
    if opt=='adam':
        optimizer = torch.optim.Adam(model.parameters(),lr=0.001)
    if opt=='sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=0.002, momentum=0.9, weight_decay=1e-4)
    
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

def sup_test(model,device=device):
    total_correct=0
    for inputs,labels in test_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs,1)
        total_correct += torch.sum(preds == labels.data)
    acc = 100*total_correct/len(test_data)
    return acc
    
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def evaluate(model, data_loader, neval_batches, criterion=config["criterion"]):
    model.eval()
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    cnt = 0
    with torch.no_grad():
        for image, target in data_loader:
            output = model(image)
            loss = criterion(output, target)
            cnt += 1
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            print('.', end = '')
            top1.update(acc1[0], image.size(0))
            top5.update(acc5[0], image.size(0))
            if cnt >= neval_batches:
                 return top1, top5

    return top1, top5