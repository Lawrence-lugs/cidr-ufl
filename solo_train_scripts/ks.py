#%%
from keyword_spotting.ks_node import ks_node
import dl_framework

my_config = dl_framework.fw_config()
my_config.federated = False


from keyword_spotting.ks_dset import mlperftiny_ks_dset
my_config.trainset = mlperftiny_ks_dset(set='train',root='/home/raimarc/lawrence-workspace/data/mlperftiny_ks_dset')
my_config.testset = mlperftiny_ks_dset(set='test',root='/home/raimarc/lawrence-workspace/data/mlperftiny_ks_dset')

node = ks_node(my_config)
node.dp_model.test_set = my_config.testset
node.dp_model.train_set = my_config.trainset
node.dp_model.learning_epochs = 100
node.dp_model.load_loaders()

node.dp_model.sup_train()
from torch import save
save(node.dp_model.model,'final_model/ks_final')