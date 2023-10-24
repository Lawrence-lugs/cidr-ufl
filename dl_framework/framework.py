import flwr as fl
import os
import dl_framework.node
import functools

def dp_node_creator(cid: str,fw_config) -> dl_framework.node.dl_node:
    '''
    Creates a flower client for the server to play with

    The server uses this function to create a client on-demand whenever it
    needs one.

    For DL Framework: creates a dl_node (which inherits as a flower client object)
    '''
    node_file = f'node_states/node_{cid}.nd'

    print(f'Checking if {node_file} exists...')
    if os.path.exists(node_file):
        print('DP Node Creator: Loading DL node...')
        node = dl_framework.node.dl_node.load_node(cid)
    else:
        print('DP Node Creator: Creating DL node...')
        node = fw_config.node_class(fw_config,name=cid)
        node.dp_model.train_set = fw_config.trainsets[int(cid)]
        node.dp_model.test_set = fw_config.testset
        node.dp_model.load_loaders()
        node.dp_model.learning_epochs = fw_config.local_epochs
        
    node.dp_model.epoch = 0 #reset current epoch to 0 to restart training

    print(f'DP Node Creator: Successfully created {node}')

    return node

def run(fw_config):

    if fw_config.federated is True:

        if not fw_config.resume:
            os.system('rm node_states/*')
        
        gpus_per_client = 1/fw_config.clients_per_gpu
        
        client_resources = {"num_gpus": gpus_per_client}

        from fed_opt.momentum_fed_avg import MomFedAvg
        FedAvg = fl.server.strategy.FedAvg
        strat = MomFedAvg

        strategy = strat(
            fraction_fit=1.0,  # Sample 100% of available clients for training
            fraction_evaluate=0.5,  # Sample 50% of available clients for evaluation
            min_fit_clients=2,  # Never sample less than 10 clients for training
            min_evaluate_clients=5,  # Never sample less than 5 clients for evaluation
            min_available_clients=2,  # Wait until all 10 clients are available
            momentum = 0.9
            )
        
        client_func = functools.partial(dp_node_creator,fw_config = fw_config)

        # Start simulation
        fl.simulation.start_simulation(
            client_fn=client_func,
            num_clients=fw_config.num_nodes,
            config=fl.server.ServerConfig(num_rounds=fw_config.num_rounds),
            strategy=strategy,
            client_resources=client_resources,
        )

        print("Finished simulation")

    else:

        node = fw_config.node_class()
        node.trainset = fw_config.trainsets[0]
        node.testset = fw_config.testset
        node.sup_train()
        return node

# %%
