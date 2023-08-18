#%%

import dl_framework.node
import dl_framework
from anomaly_detection import fc_ae
import torch
from anomaly_detection import toycar_dset
from tqdm import tqdm

def tqdm(thing, **kwargs):
    # turn off tqdm for now
    return thing

class ad_model(dl_framework.node.dp_model):
    def __init__(self, fw_config, name='default_ad_node'):
        self.fw_config = fw_config
        self.device = fw_config.get_device()
        self.writer = fw_config.get_writer

        self.model = fc_ae.FC_AE().to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(),lr=0.001)

        self.criterion = torch.nn.MSELoss()
        
        self.learning_epochs = 100
        self.epoch = 0
        self.global_epoch = 0
        self.name = name

    def load_params(self,path):
        print(f'Loading parameters from {path}...')
        self.model.load_state_dict(torch.load(path))

    def sup_train(self):
        print(f'Starting training using self.device {self.device}...')
        print(f'Trainset length : {len(self.train_set)}. Training...')
        while self.epoch < self.learning_epochs:
            self.model.train()
            runloss = 0
            for inputs in tqdm(self.train_loader,desc='Training'):
                inputs = inputs.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(inputs, outputs)
                runloss += loss
                loss.backward()
                self.optimizer.step()

            self.epoch += 1
            self.global_epoch += 1

            auc, _, _, _ = self.test(500)

            self.writer().add_scalar(f'data/node_{self.name}/loss',runloss,self.global_epoch)
            self.writer().add_scalar(f'data/node_{self.name}/auc',auc,self.global_epoch)

            print(f'[Node:{self.name}]\t epoch:{self.global_epoch}/{self.epoch}\tloss:{runloss}\tauc:{auc}')

        auc, fpr, tpr, _ = self.test()
        print(f'[Node:{self.name}]\t EndofRound:{self.global_epoch}/{self.epoch}\tloss:{runloss}\tauc:{auc}')


    def eval(self,num_elem = 500):       
        ''' Tests for ROC & AUC ''' 

        from sklearn.metrics import roc_curve,roc_auc_score
        mses = []
        lbls = []
        self.model.eval()
        for input,label in tqdm(self.eval_set,desc='Eval'):
            input = input.to(self.device)
            avgmse = 0
            outputs = self.model(input)
            avgmse += self.criterion(outputs,input)
            avgmse /= input.shape[0]
            mses.append(avgmse)
            lbls.append(label)
        mses = [i.detach().cpu() for i in mses]
        lbls = [not i for i in lbls]
        fpr,tpr,thresholds = roc_curve(lbls,mses)
        auc = roc_auc_score(lbls,mses)
        self.plot_stats(lbls,mses,fpr,tpr,auc)
        return auc, fpr, tpr
    
    def test(self,num_batches = None, plot_roc = False, test_set = None):       
        ''' Tests for ROC & AUC ''' 

        from sklearn.metrics import roc_curve,roc_auc_score
        #from tqdm import tqdm
        import numpy as np

        if test_set is None:
            test_set = self.test_set

        print(f'Size of test set: {len(test_set)}. Testing...')

        mses = []
        lbls = []
        self.model.eval()
        for input,label in tqdm(test_set,desc='Testing'):
            input = input.to(self.device)
            avgmse = 0
            outputs = self.model(input)
            avgmse += self.criterion(outputs,input)
            avgmse /= input.shape[0]
            mses.append(avgmse)
            lbls.append(label)
        mses = [i.detach().cpu() for i in mses]
        lbls = [not i for i in lbls]
        fpr,tpr,thresholds = roc_curve(lbls,mses)
        auc = roc_auc_score(lbls,mses)
        loss = np.sum(mses)
        
        if plot_roc is True:
            self.plot_stats(lbls,mses,fpr,tpr,auc)

        return auc, fpr, tpr, loss


    def load_loaders(self, train_batch = 8192, test_batch = 16, train_workers = 4, test_workers = 2):
        
        if self.fw_config.federated:
            print('Setting dataloader workers to 1 for federated learning')
            train_workers = 1
            test_workers = 1
        
        self.train_loader = torch.utils.data.DataLoader(
            self.train_set,
            batch_size=train_batch,
            num_workers=train_workers
        )
        # self.test_loader = torch.utils.data.DataLoader(
        #     self.test_set,
        #     batch_size=test_batch,
        #     num_workers=test_workers
        # )        
        # from numpy.random import randint
        # indices = randint(len(self.test_set),size=500)
        # self.eval_set = torch.utils.data.Subset(self.test_set,indices)


    def plot_stats(self,lbls,mses,fpr,tpr,auc):
        from sklearn.metrics import roc_curve,roc_auc_score
        import matplotlib.pyplot as plt
        import io, PIL
        from torchvision.transforms import ToTensor

        anoms = [val for i,val in enumerate(mses) if lbls[i] == 1]
        norms = [val for i,val in enumerate(mses) if lbls[i] == 0]

        plt.figure(figsize=(9,4))
        plt.subplot(1,2,1)
        plt.plot(fpr,tpr,label=f'AUC = {auc.round(3)}')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend()
        plt.subplot(1,2,2)
        plt.hist([norms,anoms],bins=100,label=['norms','anoms'])
        plt.legend()
        buf = io.BytesIO()
        plt.savefig(buf,format='png')
        img = PIL.Image.open(buf)
        image = ToTensor()(img)
        self.writer().add_image(f'Epoch {self.global_epoch}',image,self.global_epoch)
        plt.close()

class toycar_ad_node(dl_framework.node.dl_node):
    '''
    A dl node running anomaly detection on the DCASE2020 Task 2 Dataset (ToyCar)
    '''
    def __init__(self, fw_config, name='default_ad_node'):
        self.writer = fw_config.get_writer
        self.energy = 500
        self.round = 0
        self.name = name

        self.dp_model = ad_model(fw_config)
        self.net = self.dp_model.model
        self.dp_model.name = name

    # Redefine evaluate
    def evaluate(self, parameters, config):
        print(f"[Node {self.name}] evaluate, config: {config}")
        self.set_parameters(parameters)
        auc, fpr, tpr, loss = self.dp_model.test()
        return float(loss), len(self.dp_model.test_loader), {"accuracy": float(auc)}

    # Redefine Fit
    def fit(self, parameters, config):
        self.set_parameters(parameters)
        
        self.dp_model.sup_train()

        auc, fpr, tpr, loss = self.dp_model.test(plot_roc = True)

        #decrement energy based on number of epochs performed this round; 1 energy per epoch
        self.energy-=self.dp_model.learning_epochs
        #decrement 20 energy just for sending things
        self.energy-=20
        self.writer().add_scalar(f'data/node_{self.name}/energy',self.energy,self.round)
        self.round+=1
        print(f'[Node {self.name}]\tenergy: {self.energy}\tlocal round: {self.round}')

        self.save_node()
        return self.get_parameters(self.net), len(self.dp_model.train_loader), {"accuracy": auc}


if __name__ == '__main__':
    my_node = toycar_ad_node()
    my_node.dp_model.test_set = toycar_dset.toycar_dataset(set='test',type='wav')
    my_node.dp_model.train_set = toycar_dset.toycar_dataset(set='train',type='wav')
    my_node.dp_model.load_loaders()
    my_node.dp_model.sup_train()

# %%
