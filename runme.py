import dl_framework

# Name the Run

my_config = dl_framework.fw_config()

my_config.tensorboard_runs_dir = 'tb_data/ad_tennodes'
my_config.run_name = 'fed_ad'

# Federated Simulation Settings 
# my_config.resume = False
my_config.clients_per_gpu = 1
my_config.num_nodes = 10
my_config.local_epochs = 10
my_config.federated = True
my_config.num_rounds = 10

# Set Node Class

import anomaly_detection.ad_node
my_config.node_class = anomaly_detection.ad_node.toycar_ad_node

# Make Local Datasets

from anomaly_detection import toycar_dset as tc
trainsets = [tc.toycar_dataset(set='train') for i in range(my_config.num_nodes)]
testset = tc.toycar_dataset(set='test')

local_dataset_size = int(len(trainsets[0].files)/my_config.num_nodes)
import numpy as np
indices = np.random.randint(len(trainsets[0].files),size=len(trainsets[0].files))
for i,set in enumerate(trainsets):
    file_subset = []
    for idx in indices[local_dataset_size*i:local_dataset_size*(i+1)]:
        file_subset.append(set.files[idx])
    set.files = file_subset
print(len(testset))
# from numpy.random import randint
# indices = randint(len(testset),size=500)
# file_subset = []
# for idx in indices:
#     file_subset.append(testset.files[idx])
# testset.files = file_subset

my_config.testset = testset
my_config.trainsets = trainsets

import dl_framework.framework as framework
framework.run(my_config)



