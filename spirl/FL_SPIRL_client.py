from collections import OrderedDict
import os
from typing import Dict, List, Optional, Tuple, Union
import flwr as fl
import matplotlib; matplotlib.use('Agg')
import torch
from shutil import copy
import numpy as np
import Custome_train , Custome_train2
from spirl.components.params import get_args
from spirl.utils.general_utils import AttrDict
import random
from spirl.utils.wandb import WandBLogger

random_seed = 8888
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)
random.seed(random_seed)

WANDB_PROJECT_NAME = 'FL'
WANDB_ENTITY_NAME = 'yskang'
REWARD = 1
TMP = 100
NUM_CLIENTS = 7
NUM_ROUNDS = int(5e2)
STEPS = [0,0,0,0,0,0,0]

def make_path(exp_dir, conf_path, prefix, make_new_dir):
    path = conf_path.split('configs/', 1)[1]
    base_path = os.path.join(exp_dir, path)
    return os.path.join(base_path, prefix)


init_data_dir = "./data/kitchen/"
config = AttrDict(
    path = "spirl/configs/skill_prior_learning/kitchen/hierarchical_cl",
    prefix = "non-iid-server6",
    new_dir = False,
    dont_save = False,
    resume = "",
    train = True,
    test_prediction = True,
    skip_first_val = False,
    val_sweep = False,
    gpu = -1,
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

init_model = Custome_train2.ModelTrainer(args=config,cid=0, data_dir = init_data_dir)

dataset_class = init_model.conf.data.dataset_spec.dataset_class
train_dataset = []
loggers = []
log_dir = []
test_dataset = []

def save_checkpoint(state, folder, filename='checkpoint.pth'):
    os.makedirs(folder, exist_ok=True)
    torch.save(state, os.path.join(folder, filename))
    print(f"Saved checkpoint to {os.path.join(folder, filename)}!")


for i in range(7):
    prefix = "FL_client_"+str(i)
    exp_path = make_path(os.environ['EXP_DIR'], config.path, prefix , config.new_dir)
    log_dir.append(os.path.join(exp_path, 'events'))
    os.makedirs(exp_path)
    exp_name = f"{'_'.join(config.path.split('/')[-3:])}_{prefix}"
    loggers.append(WandBLogger(exp_name, WANDB_PROJECT_NAME, entity=WANDB_ENTITY_NAME,
                                     path=exp_path, conf=config, exclude=['model_rewards', 'data_dataset_spec_rewards']))
    data_dir =os.path.join(init_data_dir, 'FL_{}'.format(i))
    train_dataset.append(dataset_class(data_dir, init_model.conf.data, resolution=init_model.model.resolution,
                        phase="train", shuffle=True, dataset_size=-1). \
        get_data_loader(init_model._hp.batch_size, init_model._hp.epoch_cycles_train))
    test_dataset.append(dataset_class(data_dir, init_model.conf.data, resolution=init_model.model.resolution,
                        phase="val", shuffle=False, dataset_size=160). \
        get_data_loader(init_model._hp.batch_size, init_model._hp.epoch_cycles_train))




class SPIRLClient_prior(fl.client.NumPyClient):
    def __init__(self, cid, args,dataset,log,test,global_step,log_dire):
        self.cid = int(cid)
        self.global_step = global_step
        self.init_path = os.path.join("./experiments/skill_prior_learning/kitchen/",str(self.cid))
        os.makedirs(self.init_path, exist_ok=True)
        self.model = Custome_train.ModelTrainer(args=args,cid=cid,dataset= dataset, log = log,test = test, log_dire = log_dire)

    def get_parameters(self,config):
        return [val.cpu().numpy() for _, val in self.model.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.model.state_dict().keys(), parameters)
        state_dict = OrderedDict()
        for k, v in params_dict:
            if "p.0" in k:
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
        self.model.train()
        save_checkpoint({
    'epoch': 99,
    'global_step':0,
    'state_dict': self.model.model.state_dict(),
    'optimizer': self.model.optimizer.state_dict(),
},  os.path.join(self.init_path, 'weights'), f"weights_ep{self.global_step}.pth")
        return self.get_parameters(config), TMP , {'round' : 1}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss = self.model.val()
        return float(loss), TMP , {"Reward": float(REWARD)}


class SPIRLClient(fl.client.NumPyClient):
    def __init__(self, cid, args,dataset,log,test):
        self.cid = int(cid)
        self.init_path = os.path.join("./experiments/skill_prior_learning/kitchen/",str(self.cid))
        os.makedirs(self.init_path, exist_ok=True)
        self.model = Custome_train.ModelTrainer(args=args,cid=cid,dataset= dataset, log = log,test = test)


    def get_parameters(self,config):
        return [val.cpu().numpy() for _, val in self.model.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        l = []
        for d in state_dict :
            if "num_batches_tracked" in d :
                l.append(d)
        for d in l :
            del( state_dict[d] )
        # parameters update
        self.model.model.load_state_dict(state_dict,strict=True)

    def fit(self, parameters, config):
        #print("=============[fitting start]================") # 각 round를 구분하기위한 출력
        self.set_parameters(parameters)
        self.model.train()
        return self.get_parameters(config), TMP , {'round' : 1}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss = self.model.val()
        return float(loss), TMP , {"Reward": float(REWARD)}


class SaveModelStrategy(fl.server.strategy.FedAvg):
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: List[Union[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes], BaseException]],
    ) :

        # Call aggregate_fit from base class (FedAvg) to aggregate parameters and metrics
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)

        if (aggregated_parameters is not None) and (server_round % 10 == 0):
            # Convert `Parameters` to `List[np.ndarray]`
            aggregated_ndarrays: List[np.ndarray] = fl.common.parameters_to_ndarrays(aggregated_parameters)

            # Save aggregated_ndarrays
            print(f"Saving round {server_round} aggregated_ndarrays...")
            os.makedirs("/home/workspace/skill/experiments/skill_prior_learning/kitchen/real_iid/weights", exist_ok=True)
            np.savez(f"/home/workspace/skill/experiments/skill_prior_learning/kitchen/real_iid/weights/round-{server_round}-weights.npz", *aggregated_ndarrays)

        return aggregated_parameters, aggregated_metrics


def client_fn(cid) -> SPIRLClient:
    config = AttrDict(
        path = "spirl/configs/skill_prior_learning/kitchen/FL_hierarchial_cl",
        prefix = "1_client_{}".format(cid),
        new_dir = False,
        dont_save = False,
        resume = "",
        train = True,
        test_prediction = True,
        skip_first_val = False,
        val_sweep = False,
        gpu = -1,
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
    init_data_dir = "./data/kitchen/"
    dataset = train_dataset[int(cid)]
    log = loggers[int(cid)]
    test = test_dataset[int(cid)]
    global_step =  STEPS[int(cid)]
    log_dire =  log_dir[int(cid)]
    STEPS[int(cid)] += 1
    return SPIRLClient_prior(cid = cid, args = config,dataset= dataset, log = log,test = test,global_step = global_step ,log_dire = log_dire)



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
    init_model.global_step += 1
    init_model.model.load_state_dict(state_dict,strict=True)
    loss = init_model.val()
    return loss, {"accuracy": 1}



"""Create model, Create env, define Flower client, start Flower client."""
strategy = SaveModelStrategy(
min_fit_clients=NUM_CLIENTS,
min_evaluate_clients=NUM_CLIENTS,
min_available_clients=NUM_CLIENTS,
evaluate_fn=evaluate,
#initial_parameters=fl.common.ndarrays_to_parameters(init_params),
)
fl.simulation.start_simulation(
client_fn=client_fn,
num_clients=NUM_CLIENTS,
config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),  # Just three rounds
strategy=strategy,
)