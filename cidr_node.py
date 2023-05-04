import cidr_models
from train import *
from cidr_utils import *
import torch
import torchvision

class dp_model():
    '''
    A distributed processing model running inside a WSN node
    Uses mobilenetv2 on CIFAR10 by default
    '''
    def __init__(self):
        # Just for simulations
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = cidr_models.KL_MBV2().to(self.device) # TO CHANGE INTO MORE OFFICIAL MBV2 SO THAT IT CAN BE PRETRAINED
        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=0.02,
            momentum=0.9,
            weight_decay=1e-4
        )

        self.criterion = torch.nn.CrossEntropyLoss()
        self.scheduler = None
        self.learning_epochs = 200
        self.load_dataset(torchvision.datasets.CIFAR10)
        self.epoch = 0
        self.accuracy = 0
        
        
        # For the tensorboard writer
        self.tb_writer = None

    def train(self):
        print(f'Starting training using device {device}...')
        while self.epoch < self.learning_epochs:
            model = self.model
            model.train()
            epochscore = 0
            runloss = 0
            for inputs,labels in self.train_loader:
                model.train()
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = model(inputs)
                loss = self.criterion(outputs,labels)
                runloss += loss
                
                _, preds = torch.max(outputs,1)
                epochscore += torch.sum(preds == labels.data)

                loss.backward()
                self.optimizer.step()
            
            with torch.no_grad():
                self.accuracy = self.test(self.model)
            print(f'epoch {self.epoch}:\t{self.accuracy}')
            
            if self.tb_writer is not None:
                self.tb_writer.add_scalar('data/loss',runloss,self.epoch)
                self.tb_writer.add_scalar('data/testacc',self.accuracy,self.epoch)
                self.tb_writer.add_scalar('data/trainacc',epochscore/len(self.train_set),self.epoch)

            self.epoch+=1


    def test(self,num_batches = None):        
        model = self.model.to(self.device)
        if num_batches is None:
            num_batches = len(self.test_loader)
        total_correct=0
        for inputs,labels in self.test_loader:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            outputs = model(inputs)
            _, preds = torch.max(outputs,1)
            total_correct += torch.sum(preds == labels.data)
        self.accuracy = total_correct/len(self.test_set)
        return self.accuracy

    def load_dataset(self,datasetfunction: torch.utils.data.dataset):
        self.train_set = datasetfunction(
            root='data',
            train=True,
            download=False,
            transform=cidr_utils.t_cropflip_augment
        )
        self.test_set = datasetfunction(
            root='data',
            train=True,
            download=False,
            transform=cidr_utils.t_normalize
        )
        self.train_loader = torch.utils.data.DataLoader(
            self.train_set,
            batch_size=16,
            num_workers=0
        )
        self.test_loader = torch.utils.data.DataLoader(
            self.test_set,
            batch_size=16,
            num_workers=0
        )

class dl_node():
    '''
    Simulation object representing a distributed learning node
    '''
    def __init__(self):
        self.dp_model = None