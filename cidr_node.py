import cidr_models
import cidr_utils
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
        
        # For the tensorboard writer
        self.tb_writer = None

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
            print(f'epoch {self.epoch}:\ttestacc:{self.accuracy}\ttrainacc:{trainacc}\tloss:{loss}')
            
            if self.tb_writer is not None:
                self.tb_writer.add_scalar('data/loss',runloss,self.epoch)
                self.tb_writer.add_scalar('data/testacc',self.accuracy,self.epoch)
                self.tb_writer.add_scalar('data/trainacc',trainacc,self.epoch)

            self.epoch+=1

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

class dl_node():
    '''
    Simulation object representing a distributed learning node
    '''
    def __init__(self):
        self.dp_model = None