from torchvision import transforms
import torchvision

def load_vww(dpmodel):
    import pyvww
    print("Loading visual wakewords as the dataset...")
    dpmodel.train_set = pyvww.pytorch.VisualWakeWordsClassification(
        root='/home/raimarc/lawrence-workspace/MSCOCO/all2014',
        annFile='/home/raimarc/lawrence-workspace/visualwakewords/annotations/instances_train.json',
        transform = transforms.Compose([
            transforms.Resize((96,96)),
            transforms.RandomCrop(96,padding=12),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ToTensor(),
            transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225)),
        ])
    )
    dpmodel.test_set = pyvww.pytorch.VisualWakeWordsClassification(
        root='/home/raimarc/lawrence-workspace/MSCOCO/all2014',
        annFile='/home/raimarc/lawrence-workspace/visualwakewords/annotations/instances_val.json',
        transform = transforms.Compose([
            transforms.Resize((96,96)),
            transforms.ToTensor(),
            transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))
        ])
    )
    dpmodel.load_loaders()