import torch

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