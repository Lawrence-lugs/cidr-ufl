
import uuid
import torch
import tensorboard

class fw_config(object):
    def __init__(self):

        self.tensorboard_runs_dir = 'tb_data/unspecified'
        self.run_name = 'default'
        
        self.run_hash = str(uuid.uuid4())[:5]

        # Simulation Info 
        self.resume = False
        self.clients_per_gpu = 1
        self.num_nodes = 2

        # 
        self.num_rounds = 10
        self.local_epochs = 10

        self.federated = True
        self.node_class = None

        self.announced = False

    def get_writer(self):

        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(f'{self.tensorboard_runs_dir}/{self.run_name}_{self.run_hash}')
        
        if self.announced is False:
            self.announced = True
            print(f'Find the tensorboard run data in : {self.tensorboard_runs_dir}/{self.run_name}_{self.run_hash}')
        
        return writer

    def get_device(self):
        import torch
        device = torch.device('cuda') if torch.cuda.is_available else torch.device('cpu')
        return device