
#%%

import torch
import cidr_node, cidr_datasets
from torchvision import transforms
import torchvision

resize_normalize = transforms.Compose([
            transforms.Resize((96,96)),
            transforms.ToTensor(),
            #transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))
        ])

SPICE = torchvision.datasets.ImageFolder('../data/Racho_SPICE_Datasets',transform=resize_normalize)


if __name__ == '__main__':

    config = {
        'runname': 'mbv2_spice_clustering'
    }
    
    torch.set_seed = 0

    import cidr_node

    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(f'runs/{config["runname"]}')

    mbv2_dp = cidr_node.dp_model()
    mbv2_dp.tb_writer = writer
    
    from mobilenetv2 import MobileNetV2

    mbv2_dp.model = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=True).to(mbv2_dp.device)
    mbv2_dp.model.features[0][0].stride=(1,1)
    mbv2_dp.model.features[2].conv[1][0].stride=(1,1)

    mbv2_dp.optimizer = torch.optim.SGD(
        mbv2_dp.model.parameters(),
        lr = 0.002,
        momentum = 0.9,
        weight_decay = 1e-4
    )

    #%%

    spice_loader = torch.utils.data.DataLoader(
        SPICE,
        batch_size=32,
        num_workers=1
    )

    spice_iter = iter(spice_loader)


    crunched_set = mbv2_dp.model.features(next(spice_iter)[0]).mean(-1).mean(-1).cpu().detach().numpy()

    #%%

    from sklearn.cluster import KMeans

    kmodel = KMeans(n_clusters = 16)
    out = kmodel.fit(crunched_set)

    print(out)

    
# %%

    kpredictions = kmodel.predict(crunched_set)
    print(kpredictions)
