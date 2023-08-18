
import torch
import torchvision
import flwr as fl
from typing import List, OrderedDict
import numpy as np
import pickle
import image_classification.mbv2
import image_classification.utils

class dp_model():
    '''
    A distributed processing model running inside a WSN node
    Uses mobilenetv2 on CIFAR10 by default
    '''
    def __init__(self, fw_config):

        self.device = fw_config.get_device()
        self.writer = fw_config.get_writer()

        self.model = image_classification.mbv2.KL_MBV2().to(self.device)

        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=0.002,
            momentum=0.9,
            weight_decay=1e-4
        )

        self.criterion = torch.nn.CrossEntropyLoss()
        self.scheduler = None
        
        self.train_set = torchvision.datasets.CIFAR10(
            root='data',
            train=True,
            download=False,
            transform=image_classification.utils.t_cropflip_augment
        )
        self.test_set = torchvision.datasets.CIFAR10(
            root='data',
            train=True,
            download=False,
            transform=image_classification.utils.t_normalize
        )
        self.load_loaders()
        
        self.learning_epochs = 150
        self.epoch = 0

        self.global_epoch = 0
        
        self.name = 'default_name'

    def load_params(self,path):
        print(f'Loading parameters from {path}...')
        self.model.load_state_dict(torch.load(path))

    def sup_train(self):
        print(f'Starting training using device {self.device}...')
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
                _,accuracy = self.test(self.model)
                trainacc = epochscore/len(self.train_set)
            print(f'[Node:{self.name}]\t epoch:{self.global_epoch}/{self.epoch}:\ttestacc:{accuracy}\ttrainacc:{trainacc}\tloss:{runloss}')

            #if self.tb_writer is not None:
            self.writer.add_scalar(f'data/node_{self.name}/loss',runloss,self.global_epoch)
            self.writer.add_scalar(f'data/node_{self.name}/testacc',accuracy,self.global_epoch)
            self.writer.add_scalar(f'data/node_{self.name}/trainacc',trainacc,self.global_epoch)
                              
            self.epoch+=1
            self.global_epoch+=1

    def test(self,num_batches = None):       
        ''' Tests the accuracy of the dp model on testset ''' 
        model = self.model.to(self.device)
        if num_batches is None:
            num_batches = len(self.test_loader)
        loss = 0
        total_correct=0
        for inputs,labels in self.test_loader:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            outputs = model(inputs)
            loss += self.criterion(outputs, labels).item()
            _, preds = torch.max(outputs,1)
            total_correct += torch.sum(preds == labels.data)
        accuracy = total_correct/len(self.test_set)
        return loss,accuracy

    def load_loaders(self):
        self.train_loader = torch.utils.data.DataLoader(
            self.train_set,
            batch_size=16,
            num_workers=2
        )
        self.test_loader = torch.utils.data.DataLoader(
            self.test_set,
            batch_size=16,
            num_workers=2
        )

class dl_node(fl.client.NumPyClient):
    '''
    Simulation object representing a distributed learning node

    Implements a mobilenetv2 on CIFAR10 by default.

    This inherits from flower client for federated learning.
    '''
    def __init__(self, name='default_name'):
        self.dp_model = dp_model()
        self.net = self.dp_model.model
        self.dp_model.name = name
        self.name = name
        self.energy = 500
        self.round = 0

    def get_parameters(self, config):
        '''
        Returns the parameters of the local net
        '''
        return [val.cpu().numpy() for _, val in self.net.state_dict().items()]

    def set_parameters(self, parameters: List[np.ndarray]):
        '''
        Sets the local net's parameters to the parameters given
        '''
        params_dict = zip(self.net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.dp_model.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        
        self.dp_model.sup_train()

        _,accuracy = self.dp_model.test()

        #decrement energy based on number of epochs performed this round; 1 energy per epoch
        self.energy-=self.dp_model.learning_epochs
        #decrement 20 energy just for sending things
        self.energy-=20
        self.writer.add_scalar(f'data/node_{self.name}/energy',self.energy,self.round)
        self.round+=1
        print(f'[Node {self.name}]\tenergy: {self.energy}\tlocal round: {self.round}')

        self.save_node()
        return self.get_parameters(self.net), len(self.dp_model.train_loader), {"accuracy": accuracy}
    
    def evaluate(self, parameters, config):
        print(f"[Node {self.name}] evaluate, config: {config}")
        self.set_parameters(parameters)
        loss, accuracy = self.dp_model.test()
        return float(loss), len(self.dp_model.test_loader), {"accuracy": float(accuracy)}

    def save_node(self):
        '''
        Saves node data to the node state directory
        '''

        path = f'node_states/node_{self.name}.nd'
        with open(path,'wb') as handle: 
            pickle.dump(self, handle)
        print(f'Dumped {self} information in {path}')

    @classmethod
    def load_node(cls,name: str):
        '''
        Loads node data to the node state directory
        '''
        path = f'node_states/node_{name}.nd'
        with open(path,'rb') as handle:
            return pickle.load(handle)