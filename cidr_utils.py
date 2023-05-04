import torch
import cidr_models
import os
from torchvision import transforms

def save_progress(filename,epoch,model_state,optimizer_state):
    '''
    Saves the training progress of a model in training.
    Restarts from the last successful epoch.
    '''
    path = f'saved_runs/{filename}'
    torch.save({
        'epoch':epoch,
        'model':model_state,
        'opt':optimizer_state
        },path)
    return

def load_model(model_file):
    model = cidr_models.KL_MBV2_Q()
    state_dict = torch.load(model_file)
    model.load_state_dict(state_dict)
    model.to('cpu')
    return model

def print_size_of_model(model):
    torch.save(model.state_dict(), "temp.p")
    print('Size (MB):', os.path.getsize("temp.p")/1e6)
    os.remove('temp.p')

t_cropflip_augment = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

t_normalize = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])