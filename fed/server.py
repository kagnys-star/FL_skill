import flwr as fl
import os
import numpy as np
import torch
from strategy import *
from cl_model import BaseClients , FN_Clients
from spirl.components.params import get_args

def fun_save_path(args):
    # Get the environment variable for the experiment directory
    exp_dir = os.environ['EXP_DIR']
    
    # Extract the part of the path after 'configs/'
    path = args.path.split('configs/', 1)[1]  # Extract 'skill_prior_learning/half_cheetah/FL_hierarchial_cl'
    
    # Remove 'FL_hierarchial_cl' (or any last part) from the path
    path_components = path.split('/')[:-1]  # Extract everything except the last part (FL_hierarchial_cl)
    modified_path = '/'.join(path_components)  # Rejoin the components back
    
    # Extract the relevant parts from the prefix (fedadgrad and iid)
    prefix_parts = args.prefix.split('-')
    method = prefix_parts[1]  # 'fedadgrad'
    # 'iid_server' needs to be split to extract just 'iid'
    data_distribution = prefix_parts[2].split('_')[0]  # Extract 'iid' from 'iid_server'
    
    # Create the final save path
    save_path = os.path.join(exp_dir, modified_path, method, data_distribution, 'weights')
    
    return save_path


if __name__ == "__main__":
    """Load model for
    1. server-side parameter initialization
    2. server-side parameter evaluation
    """
    args = get_args()
    init_model = BaseClients(args=args)
    num_clients = 2
    num_rounds = 3
    # Parse command line argument `partition`
    #parser = argparse.ArgumentParser(description="Flower")

    model_parameters = [val.cpu().numpy() for _, val in init_model.model.state_dict().items()]
    save_dir = fun_save_path(args=args)
    #del init_model

    # Create strategy
    strategy = FedDYN(
        dyn_alpha = 0.1,
        n_clients = num_clients,
        save_dir = save_dir,
        #num_rounds = num_rounds,
        min_fit_clients=num_clients,
        min_evaluate_clients=num_clients,
        min_available_clients=num_clients,
        #evaluate_fn=get_evaluate_fn(model, args.toy),
        #on_fit_config_fn=fit_config,
        #on_evaluate_config_fn=evaluate_config,
        initial_parameters=fl.common.ndarrays_to_parameters(model_parameters),
    )

    # Start Flower server for four rounds of federated learning
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
    )
