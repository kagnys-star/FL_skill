import flwr as fl
from collections import OrderedDict
import os
from typing import Dict, List, Optional, Tuple, Union
import matplotlib; matplotlib.use('Agg')
import torch
from shutil import copy
import numpy as np
import datetime
import Custome_train2
from spirl.components.params import get_args
from flwr.server.strategy import FedAvg , FedProx
from spirl.utils.general_utils import AttrDict
import random

WANDB_PROJECT_NAME = 'fl-skill'
WANDB_ENTITY_NAME = 'yskang'
REWARD = 1
TMP = 100
NUM_CLIENTS = 7
NUM_ROUNDS = int(500)
STEPS = [0,0,0,0,0,0,0]
LENGTH = []

def set_seeds(seed=0, cuda_deterministic=True):
    """Sets all seeds and disables non-determinism in cuDNN backend."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    if torch.cuda.is_available() and cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

def datetime_str():
    return datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

def make_path(exp_dir):
    # extract the subfolder structure from config path
    prefix = datetime_str()
    base = os.path.join(exp_dir, prefix)
    os.makedirs(name = base, exist_ok= True)
    return base

class SPIRLClient(fl.client.NumPyClient):
    def __init__(self, cid, config,init_data_dir,grap_round):
        self.cid = int(cid)
        self.grap_round =grap_round
        self.model = Custome_train2.ModelTrainer(args=config, cid=self.cid, data_dir = init_data_dir , grap_round =grap_round)
        self.init_path = os.path.join("./experiments/skill_prior_learning/kitchen/hetetro_2/5",str(self.cid))
        os.makedirs(self.init_path, exist_ok=True)

    def get_parameters(self,config):
        return [val.cpu().numpy() for _, val in self.model.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.model.state_dict().keys(), parameters)
        #state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        state_dict = OrderedDict()
        for k, v in params_dict:
            state_dict[k] = torch.Tensor(v)
        l = []
        for d in state_dict :
            if "num_batches_tracked" in d :
                l.append(d)
        for d in l :
            del( state_dict[d] )
        # parameters update
        self.model.model.load_state_dict(state_dict,strict=False)

    def fit(self, parameters, config):
        print("=============[fitting start]================") # 각 round를 구분하기위한 출력
        self.set_parameters(parameters)
        params_dict = zip(self.model.model.state_dict().keys(), parameters)
        state_dict = OrderedDict()
        for k, v in params_dict:
            state_dict[k] = torch.Tensor(v)
        self.model.train(state_dict)
        np.savez(os.path.join(self.init_path,f"round-{self.grap_round}-weights.npz"), self.get_parameters(config))
        return self.get_parameters(config), TMP , {'round' : 1}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        #loss = self.model.val()
        return float(0), TMP , {"Reward": float(REWARD)}

class SaveModelStrategy(FedProx):
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: List[Union[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes], BaseException]],
    ) :
        global STEPS
        for i,n in enumerate(STEPS):
            STEPS[i] = n + 1

        # Call aggregate_fit from base class (FedAvg) to aggregate parameters and metrics
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)

        if (aggregated_parameters is not None) and (server_round % 5 == 0):
            # Convert `Parameters` to `List[np.ndarray]`
            aggregated_ndarrays: List[np.ndarray] = fl.common.parameters_to_ndarrays(aggregated_parameters)

            # Save aggregated_ndarrays
            print(f"Saving round {server_round} aggregated_ndarrays...")
            os.makedirs("/home/kangys/workspace/FL_skill/experiments/skill_prior_learning/kitchen/hetetro_2/5/weights", exist_ok=True)
            np.savez(f"/home/kangys/workspace/FL_skill/experiments/skill_prior_learning/kitchen/hetetro_2/5/weights/round-{server_round}-weights.npz", *aggregated_ndarrays)

        return aggregated_parameters, aggregated_metrics


if __name__ == "__main__":
    set_seeds(seed=0)
    init_data_dir = "/home/kangys/workspace/FL_skill/data/data11/"
    config = AttrDict(
        path = "/home/kangys/workspace/FL_skill/spirl/configs/skill_prior_learning/kitchen/FL_hierarchial_cl",
        prefix = "hetetro_2-9_server",
        new_dir = False,
        dont_save = False,
        resume = "",
        train = True,
        test_prediction = True,
        skip_first_val = False,
        val_sweep = False,
        gpu = 0,
        strict_weight_loading = True,
        deterministic = False,
        log_interval = 500,
        per_epoch_img_logs = 1,
        val_data_size = 160,
        val_interval = 5,
        detect_anomaly = False,
        feed_random_data = False,
        train_loop_pdb = False,
        debug = False,
        save2mp4 = False
    )
    
    init_model = Custome_train2.ModelTrainer(args=config,cid=0,data_dir=init_data_dir, grap_round = 0)
    
    def client_fn(cid) -> SPIRLClient:
        config = AttrDict(
            path = "/home/kangys/workspace/FL_skill/spirl/configs/skill_prior_learning/kitchen/FL_hierarchial_cl",
            prefix = "hetetro_2-9_client_{}".format(cid),
            new_dir = False,
            dont_save = False,
            resume = "",
            train = True,
            test_prediction = True,
            skip_first_val = False,
            val_sweep = False,
            gpu = 0,
            strict_weight_loading = True,
            deterministic = False,
            log_interval = 100,
            per_epoch_img_logs = 1,
            val_data_size = 128,
            val_interval = 5,
            detect_anomaly = False,
            feed_random_data = False,
            train_loop_pdb = False,
            debug = False,
            save2mp4 = False
        )
        grap_round =STEPS[int(cid)]
        return SPIRLClient(cid = cid,config = config, init_data_dir = init_data_dir, grap_round = grap_round)

    def evaluate(
    server_round: int, parameters: fl.common.NDArrays, config: Dict[str, fl.common.Scalar]
) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
        params_dict = zip(init_model.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        l = []
        for d in state_dict :
            if "num_batches_tracked" in d :
                l.append(d)
        for d in l :
            del( state_dict[d] )
        # parameters update
        init_model.grap_round += 1
        init_model.model.load_state_dict(state_dict,strict=True)
        loss = init_model.val()
        return loss, {"accuracy": 1}
    #my_client_resources = {'num_cpus': 2, 'num_gpus': 0.1}
    """Create model, Create env, define Flower client, start Flower client."""
    strategy = SaveModelStrategy(
        min_fit_clients=NUM_CLIENTS,
        min_evaluate_clients=NUM_CLIENTS,
        min_available_clients=NUM_CLIENTS,
        proximal_mu = 0.01,
        #evaluate_fn=evaluate,
        #initial_parameters=fl.common.ndarrays_to_parameters(init_params),
        )
    
    fl.simulation.start_simulation(
    client_fn=client_fn,
    #client_resources = my_client_resources,
    num_clients=NUM_CLIENTS,
    config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS), 
    strategy=strategy,
)
