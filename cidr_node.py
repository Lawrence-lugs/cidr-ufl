import cidr_models
import cidr_utils
import torch
import torchvision
import flwr as fl
from typing import List, OrderedDict
import numpy as np
import mobilenetv2
import pickle

device = torch.device('cuda')

# Setup tensorboard
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter(f'federated_runs/fl_test_run_3')

class dp_model():
    '''
    A distributed processing model running inside a WSN node
    Uses mobilenetv2 on CIFAR10 by default
    '''
    def __init__(self):
        # Just for simulations
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # self.model = mobilenetv2.MobileNetV2(
        #     num_classes = 10,
        #     width_mult = 1
        #     ).to(self.device)

        self.model = cidr_models.KL_MBV2().to(self.device)

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
            transform=cidr_utils.t_cropflip_augment
        )
        self.test_set = torchvision.datasets.CIFAR10(
            root='data',
            train=True,
            download=False,
            transform=cidr_utils.t_normalize
        )
        self.load_loaders()
        
        self.learning_epochs = 150
        self.epoch = 0
        self.accuracy = 0

        self.global_epoch = 0
        
        self.name = 'default_name'
        self.tb_writer = None
        # For the tensorboard writer
        #from torch.utils.tensorboard import SummaryWriter
        #self.tensorboard_log_name = 'runs/dlf_flower_integration'
        #self.tb_writer = SummaryWriter(self.tensorboard_log_name)

    def quantize(self):
        ''' 
        Post-training static quantization on the loaded model
        To use this, your model must have the "quantized" attribute
        and a "PTQ_prepare" method.
        Setting this attribute to true should turn the model into its quantization
        compatible mode.
        '''
        old_model = self.model
        
        print('Device has been set to CPU for quantized inference.')
        self.device = torch.device('cpu')

        try:
            self.model.to(self.device)
            self.model.eval()
            self.model.PTQ_prepare()

            # this attribute does not originally exist
            self.model.qconfig = torch.ao.quantization.default_qconfig
            
            # insert the observers
            torch.ao.quantization.prepare(self.model, inplace=True)

            # show the observers the fmap distributions
            num_calibration_batches = 32
            self.test(num_calibration_batches)

            torch.ao.quantization.convert(self.model, inplace=True)
        except Exception as e:
            self.device = torch.device('cuda')
            self.model = old_model.to(device)
            print(f'Model could not be quantized: {e}')

    def load_params(self,path):
        print(f'Loading parameters from {path}...')
        self.model.load_state_dict(torch.load(path))

    def sup_train(self):
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
                trainacc = epochscore/len(self.train_set)
            print(f'epoch {self.global_epoch}/{self.epoch}:\ttestacc:{self.accuracy}\ttrainacc:{trainacc}\tloss:{runloss}')

            #if self.tb_writer is not None:
            writer.add_scalar(f'data/node_{self.name}/loss',runloss,self.global_epoch)
            writer.add_scalar(f'data/node_{self.name}/testacc',self.accuracy,self.global_epoch)
            writer.add_scalar(f'data/node_{self.name}/trainacc',trainacc,self.global_epoch)
                              
            self.epoch+=1
            self.global_epoch+=1

    def test(self,num_batches = None):       
        ''' Tests the accuracy of the dp model on testset ''' 
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

        accuracy = self.dp_model.test()

        #decrement energy based on number of epochs performed this round; 1 energy per epoch
        self.energy-=self.dp_model.learning_epochs
        #decrement 20 energy just for sending things
        self.energy-=20
        writer.add_scalar(f'data/node_{self.name}/energy',self.energy,self.round)
        self.round+=1
        print(f'Node {self.name}\tenergy: {self.energy}\tlocal round: {self.round}')

        self.save_node()
        return self.get_parameters(self.net), len(self.dp_model.train_loader), {"accuracy": accuracy}
    
    def save_node(self):
        '''
        Saves node data to the node state directory
        '''
        # Cannot pickle a tensorboard writer
        self.dp_model.tb_writer = None

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