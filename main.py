
import torch
import torchhd
import cidr_node, cidr_datasets

class hdc_classifier(torch.nn.Module):
    def __init__(self, num_classes, size, dimensions, num_levels):
        super(hdc_classifier, self).__init__()

        self.flatten = torch.nn.Flatten()

        # position are unrelated to each other
        self.position = torchhd.embeddings.Random(size * size, dimensions)
        # value is an ordered set of levels 
        self.value = torchhd.embeddings.Level(num_levels, dimensions)

        self.classify = torch.nn.Linear(dimensions, num_classes, bias=False)
        for param in self.classify.parameters():
            param.requires_grad = False

        self.classify.weight.data.fill_(0.0)

    def encode(self,x):
        x = self.flatten(x)
        sample_hv = torchhd.functional.bind(self.position.weight, self.value(x))
        sample_hv = torchhd.functional.multiset(sample_hv)
        return torchhd.functional.hard_quantize(sample_hv) 
    
    def forward(self,x):
        enc = self.encode(x)
        logit = self.classify(enc)
        return logit

if __name__ == '__main__':

    config = {
        'runname': 'hdc_mnist'
    }
    torch.set_seed = 0

    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(f'runs/{config["runname"]}')

    hdc_dp = cidr_node.dp_model()
    hdc_dp.tb_writer = writer

    hdc_dp.model = hdc_classifier(10,28,10000,16).to(torch.device('cuda'))
    cidr_datasets.load_mnist(hdc_dp)

    print('Starting training...')
    for samples, labels in hdc_dp.train_loader:
        device = torch.device('cuda')
        samples = samples.to(device)
        label = labels.to(device)
        samples_hv = hdc_dp.model.encode(samples)
        with(torch.no_grad()):
            hdc_dp.model.classify.weight[labels] += samples_hv

    hdc_dp.model.classify.weight[:] = torch.nn.functional.normalize(hdc_dp.model.classify.weight)

    print('Starting classification...')
    print(hdc_dp.test())


    
# %%
