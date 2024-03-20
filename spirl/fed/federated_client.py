from collections import OrderedDict
import flwr as fl
import torch
import os
import numpy as np
from spirl.fed.federated_trainer import ModelTrainer
from logging import INFO, DEBUG
from flwr.common.logger import log


class SPIRLClient(fl.client.NumPyClient):
    def __init__(self, cid, config,logger):
        self.cid = int(cid)
        self.setup_device()
        self.save_path = os.path.join(config.params.exp_path, str(self.cid))
        os.makedirs(self.save_path, exist_ok=True)
        self.trianer = ModelTrainer(cid = self.cid , config= config  , logger = logger)

    def get_parameters(self,config):
        return [val.cpu().numpy() for _, val in self.trianer.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.trianer.model.state_dict().keys(), parameters)
        state_dict = OrderedDict()
        for k, v in params_dict:
            state_dict[k] = torch.Tensor(v)
        l = []
        for d in state_dict :
            if "num_batches_tracked" in d :
                l.append(d)
        for d in l :
            del( state_dict[d] )
        self.trianer.model.load_state_dict(state_dict,strict=False)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        #self.trianer.global_step = config["steps"]
        params_dict = zip(self.trianer.model.state_dict().keys(), parameters)
        state_dict = OrderedDict()
        for k, v in params_dict:
            state_dict[k] = torch.Tensor(v)
        steps = self.trianer.train(state_dict)
        #sever_round = config["server_round"]
        np.savez(os.path.join(self.save_path ,f"round-{1}-weights.npz"), self.get_parameters(config))
        return self.get_parameters(config), len(self.trianer.train_loader) , {'steps' : steps}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        #loss, accuracy = self.trianer.val()
        return float(0), 100 , {"Reward": float(1)}
    
    def setup_device(self):
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device('cuda') if self.use_cuda else torch.device('cpu')
        os.environ["CUDA_VISIBLE_DEVICES"] = str(0)