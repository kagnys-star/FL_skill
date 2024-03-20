import flwr as fl
from typing import List , Tuple , Union
from flwr.common import Parameters, FitIns
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
import numpy as np
import os
from logging import INFO, DEBUG
from flwr.common.logger import log

class SaveModelStrategy(fl.server.strategy.FedAvg):
    def __init__(self, weights_path,*args,**kwargs) -> None:
        super().__init__(*args,**kwargs)
        self.client_steps = {}
        self.weights_path = weights_path

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: List[Union[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes], BaseException]],
    ) :
        # Call aggregate_fit from base class (FedAvg) to aggregate parameters and metrics
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)
        for (client_id, fit_res) in results:
            self.client_steps[client_id] = fit_res.config["steps"]
            
        if (aggregated_parameters is not None) and (server_round % 5 == 0):
            # Convert `Parameters` to `List[np.ndarray]`
            aggregated_ndarrays: List[np.ndarray] = fl.common.parameters_to_ndarrays(aggregated_parameters)
            # Save aggregated_ndarrays, make clear,
            print(f"Saving round {server_round} aggregated_ndarrays...")
            np.savez(os.path.join(self.weights_path, f"round-{server_round}-weights.npz"), *aggregated_ndarrays)
        return aggregated_parameters, aggregated_metrics
"""
    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        results = []
        config = {}
        for client_id in  client_manager.all().keys():
            steps = self.client_steps.get(client_id, 0)
            config[client_id] = {"steps": steps , "server_round" : server_round}
            fit_ins = FitIns(parameters, config)
            results.append((client_id, fit_ins))

        # Return client/config pairs
        return results
"""