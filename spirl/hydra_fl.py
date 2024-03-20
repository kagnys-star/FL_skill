import flwr as fl
import hydra
import pprint
import datetime
from omegaconf import DictConfig, OmegaConf
from collections import OrderedDict
import os
import matplotlib; matplotlib.use('Agg')
import torch
import numpy as np
from spirl.fed.federated_trainer import ModelTrainer, set_seeds
from spirl.utils.general_utils import AttrDict
from spirl.utils.wandb import WandBLogger
from spirl.fed.federated_sever import SaveModelStrategy

from spirl.fed.federated_client import SPIRLClient

# 저장 위치 자동 생성
def datetime_str():
    return datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

def make_path(exp_dir):
    # extract the subfolder structure from config path
    prefix = datetime_str()
    base = os.path.join(exp_dir, prefix)
    os.makedirs(name = base, exist_ok= True)
    return base

# Log 통합
def setup_logging(conf):
    print('Writing to the experiment directory: {}'.format(os.environ['EXP_DIR']))
    path = make_path(os.environ['EXP_DIR'])
    exp_name = datetime_str() + conf.fed.type
    conf.exp_path = path
    #writer = WandBLogger(exp_name, project_name= conf.wandb.project, entity= conf.wandb.entity,
    #                            path=path, conf=conf, exclude=['model_rewards', 'data_dataset_spec_rewards'])
    return None #writer

#서버에서 각각의 step을 내리는 것과 받는 과정을 추가

@hydra.main(version_base=None, config_path="hydra_conf", config_name="defaults")
def run_experiment(cfg : DictConfig) -> None:
    set_seeds(cfg.random_seed)
    LOGGER = setup_logging(cfg)
    
    #pprint.pprint(OmegaConf.to_yaml(cfg))
    config = AttrDict(cfg.trainer)
    config.prefix = "server",
    init_model = ModelTrainer(cid=0, config = config, logger = LOGGER)
    exit()
    num_client = cfg.fed.clients
    

    def client_fn(cid) -> SPIRLClient:
        config = AttrDict(cfg.trainer)
        config.prefix = "client_{}".format(cid),
        return SPIRLClient(cid = cid , config = config, logger =  LOGGER)

    def evaluate(server_round, parameters, config):
        init_model.global_step = server_round
        params_dict = zip(init_model.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        l = []
        for d in state_dict :
            if "num_batches_tracked" in d :
                l.append(d)
        for d in l :
            del(state_dict[d])
        # parameters update[]
        init_model.global_step += 1
        init_model.model.load_state_dict(state_dict,strict=True)
        loss = init_model.val()
        return loss, {"accuracy": 1}

    """Create model, Create env, define Flower client, start Flower client."""
    weights_path = os.path.join(cfg.exp_path, "server")
    os.makedirs(weights_path, exist_ok=True)
    strategy = SaveModelStrategy(
        weights_path = weights_path,
        min_fit_clients = num_client,
        min_evaluate_clients = num_client,
        min_available_clients = num_client,
        evaluate_fn=evaluate,
        #initial_parameters=fl.common.ndarrays_to_parameters(init_params),
        )
    
    fl.simulation.start_simulation(
    client_fn=client_fn,
    num_clients = num_client,
    config=fl.server.ServerConfig(num_rounds=cfg.fed.rounds), 
    strategy=strategy,
    )

if __name__ == "__main__":
    run_experiment()