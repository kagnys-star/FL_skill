import matplotlib; matplotlib.use('Agg')
from shutil import copy
from spirl.components.params import get_args
from client_model import *
import flwr as fl

WANDB_PROJECT_NAME = 'fl-skill'
WANDB_ENTITY_NAME = 'yskang'

if __name__ == '__main__':
    args=get_args()
    exp_mode = args.exp_mode
    if exp_mode == 'fedavg':
        client = BaseClients(args = args).to_client()
    elif exp_mode == 'fedasam':
        client = ASAMClients(args = args).to_client()
    elif exp_mode == 'fedprox':
        client = FP_Clients(args = args).to_client()
    elif exp_mode == 'fednova':
        client = FN_Clients(args = args).to_client()
    elif exp_mode == 'fedsol':
        client = FS_Clients(args = args).to_client()
    elif exp_mode == 'feddyn':
        client = FD_Clients(args = args).to_client()
    elif exp_mode == 'test':
        client = Test_Clients(args = args).to_client()
    elif exp_mode == 'fedopt':
        client = Mulopt_Clients(args = args).to_client()
    elif exp_mode == 'fedgan':
        client = VAEGAN_Clients(args = args).to_client()
    elif exp_mode == 'feddez':
        client = Dualenc_Clients(args = args).to_client()
    else:
        raise ValueError("federated learning '{}' not supported!".format(exp_mode))
    fl.client.start_client(server_address="127.0.0.1:8080", client=client)
