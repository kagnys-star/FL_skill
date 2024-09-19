import flwr as fl
import os
import copy
import torch
from torch.optim import SGD
from collections import OrderedDict
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from flwr.server.strategy import FedAvg , FedProx , FedAdam , FedAdagrad, FedYogi
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy.aggregate import aggregate
from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    MetricsAggregationFn,
    NDArrays,
    NDArray,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)


class SM_FedYogi(FedYogi):
    def __init__(self, save_dir, **kwargs):
        self.save_dir = save_dir
        os.makedirs(self.save_dir , exist_ok=True)
        super().__init__(**kwargs)


    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, FitRes]],
        failures: List[Union[Tuple[fl.server.client_proxy.ClientProxy, FitRes], BaseException]],
    ) :
        # Call aggregate_fit from base class (FedAvg) to aggregate parameters and metrics
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)

        if (aggregated_parameters is not None) and (server_round % 5 == 0):
            # Convert `Parameters` to `List[np.ndarray]`
            aggregated_ndarrays: List[np.ndarray] = parameters_to_ndarrays(aggregated_parameters)
            # Save aggregated_ndarrays
            np.savez(os.path.join(self.save_dir,f"round-{server_round}-weights.npz"), *aggregated_ndarrays)
        return aggregated_parameters, aggregated_metrics


class SM_FedAdagrad(FedAdagrad):
    def __init__(self, save_dir, **kwargs):
        self.save_dir = save_dir
        os.makedirs(self.save_dir , exist_ok=True)
        super().__init__(**kwargs)


    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, FitRes]],
        failures: List[Union[Tuple[fl.server.client_proxy.ClientProxy, FitRes], BaseException]],
    ) :
        # Call aggregate_fit from base class (FedAvg) to aggregate parameters and metrics
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)

        if (aggregated_parameters is not None) and (server_round % 5 == 0):
            # Convert `Parameters` to `List[np.ndarray]`
            aggregated_ndarrays: List[np.ndarray] = parameters_to_ndarrays(aggregated_parameters)
            # Save aggregated_ndarrays
            np.savez(os.path.join(self.save_dir,f"round-{server_round}-weights.npz"), *aggregated_ndarrays)
        return aggregated_parameters, aggregated_metrics


class SM_FedProx(FedProx):
    def __init__(self, save_dir, **kwargs):
        self.save_dir = save_dir
        os.makedirs(self.save_dir , exist_ok=True)
        super().__init__(**kwargs)


    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, FitRes]],
        failures: List[Union[Tuple[fl.server.client_proxy.ClientProxy, FitRes], BaseException]],
    ) :
        # Call aggregate_fit from base class (FedAvg) to aggregate parameters and metrics
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)

        if (aggregated_parameters is not None) and (server_round % 5 == 0):
            # Convert `Parameters` to `List[np.ndarray]`
            aggregated_ndarrays: List[np.ndarray] = parameters_to_ndarrays(aggregated_parameters)
            # Save aggregated_ndarrays
            np.savez(os.path.join(self.save_dir,f"round-{server_round}-weights.npz"), *aggregated_ndarrays)
        return aggregated_parameters, aggregated_metrics


class SM_FedAdam(FedAdam):
    def __init__(self, save_dir, **kwargs):
        self.save_dir = save_dir
        os.makedirs(self.save_dir , exist_ok=True)
        super().__init__(**kwargs)


    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, FitRes]],
        failures: List[Union[Tuple[fl.server.client_proxy.ClientProxy, FitRes], BaseException]],
    ) :
        # Call aggregate_fit from base class (FedAvg) to aggregate parameters and metrics
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)

        if (aggregated_parameters is not None) and (server_round % 5 == 0):
            # Convert `Parameters` to `List[np.ndarray]`
            aggregated_ndarrays: List[np.ndarray] = parameters_to_ndarrays(aggregated_parameters)
            # Save aggregated_ndarrays
            np.savez(os.path.join(self.save_dir,f"round-{server_round}-weights.npz"), *aggregated_ndarrays)
        return aggregated_parameters, aggregated_metrics


class SM_FedAVG(FedAvg):
    def __init__(self, save_dir, **kwargs):
        self.save_dir = save_dir
        os.makedirs(self.save_dir , exist_ok=True)
        super().__init__(**kwargs)


    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, FitRes]],
        failures: List[Union[Tuple[fl.server.client_proxy.ClientProxy, FitRes], BaseException]],
    ) :
        # Call aggregate_fit from base class (FedAvg) to aggregate parameters and metrics
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)

        if (aggregated_parameters is not None) and (server_round % 5 == 0):
            # Convert `Parameters` to `List[np.ndarray]`
            aggregated_ndarrays: List[np.ndarray] = parameters_to_ndarrays(aggregated_parameters)
            # Save aggregated_ndarrays
            np.savez(os.path.join(self.save_dir,f"round-{server_round}-weights.npz"), *aggregated_ndarrays)
        return aggregated_parameters, aggregated_metrics


class FEDASAM(FedAvg):
    def __init__(self, save_dir, lr, num_rounds, swa_lr=1e-4, cycle_length=10, swa_start=0.75, **kwargs):
        super().__init__(**kwargs)
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        self.swa_start = int(swa_start * num_rounds)  # Ensure swa_start is an integer
        self.cycle_length = cycle_length
        self.swa_model = None
        self.lr = lr
        self.swa_lr = swa_lr
        self.swa_n = 0


    def schedule_cycling_lr(self, round):
        t = 1 / self.cycle_length * (round % self.cycle_length + 1)
        lr = (1 - t) * self.lr + t * self.swa_lr
        config = {
            "learning_rate": lr,
        }
        return config


    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: List[Union[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes], BaseException]],
    ):
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)
        
        # Initialize SWA model at the starting round
        if server_round == self.swa_start:
            self.swa_model = parameters_to_ndarrays(aggregated_parameters)
        elif server_round > self.swa_start and (server_round - self.swa_start) % self.cycle_length == 0:
            alpha = 1.0 / (self.swa_n + 1)
            for param1, param2 in zip(self.swa_model, parameters_to_ndarrays(aggregated_parameters)):
                param1 *= (1.0 - alpha)
                param1 += param2 * alpha
            self.swa_n += 1
        
        # Save the SWA parameters if needed
        if (aggregated_parameters is not None) and (server_round % 5 == 0):
            # Convert `Parameters` to `List[np.ndarray]`
            aggregated_ndarrays: List[np.ndarray] = parameters_to_ndarrays(aggregated_parameters)
            # Save aggregated_ndarrays
            np.savez(os.path.join(self.save_dir,f"round-{server_round}-weights.npz"), *aggregated_ndarrays)
            if server_round == self.swa_start + self.cycle_length * (self.swa_n + 1):
                # Convert swa_mode to model parameters and save them
                swa_params = ndarrays_to_parameters(self.swa_model)
                # Example: Save the SWA model parameters (implement saving logic as needed)
                torch.save(swa_params, os.path.join(self.save_dir, f"swa_model_round_{server_round}.pth"))
        return aggregated_parameters, aggregated_metrics


    def configure_fit(self, server_round, parameters, client_manager):
        """Configure the next round of training."""
        config = {}
        config = self.schedule_cycling_lr(server_round)
        fit_ins = FitIns(parameters, config)

        # Sample clients
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )
        # Return client/config pairs
        return [(client, fit_ins) for client in clients]


class FEDASAM_opt(FedAvg):
    def __init__(self, model, cycle_length=5, lr = 0.01, swa_lr=1e-4, num_rounds=100, swa_start=0.75, **kwargs):
        super().__init__()
        self.client_model = copy.deepcopy(model)
        self.device = self.client_model.device
        self.model = copy.deepcopy(model.state_dict())
        self.server_opt  = SGD(params=self.model.parameters(), lr=1, momentum=0)
        self.cycle_length = cycle_length
        self.num_rounds = num_rounds
        self.swa_start = swa_start * num_rounds
        self.lr = lr
        self.swa_lr = swa_lr
        self.nmodels = 0


    def schedule_cycling_lr(self,round):
        t = 1 / self.cycle_length * (round % self.cycle_length + 1)
        lr = (1 - t) * self.lr + t * self.swa_lr
        config = {"learning_rate": lr,}
        return config


    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, FitRes]],
        failures: List[Union[Tuple[fl.server.client_proxy.ClientProxy, FitRes], BaseException]],
    ):
        if server_round == self.swa_start:
            self.swa_model = copy.deepcopy(self.client_model)
        self.server_opt.zero_grad()
        self.client_model.load_state_dict(self.model)
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)
        self._update_global_model_gradient(aggregated_parameters)
        self.model = copy.deepcopy(self.client_model.state_dict())
        self.total_grad = self._get_model_total_grad()
        if  server_round > self.swa_start and (server_round - self.swa_start) % self.cycle_length == 0:
            self.update_swa_model()
        return aggregated_parameters, aggregated_metrics


    def configure_fit(
        self, server_round, parameters, client_manager):
        """Configure the next round of training."""
        config = {}
        config = self.schedule_cycling_lr(server_round)
        fit_ins = FitIns(parameters, config)

        # Sample clients
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )
        # Return client/config pairs
        return [(client, fit_ins) for client in clients]
    
    def update_swa_model(self):
        alpha = 1.0 / (self.swa_n + 1)
        for param1, param2 in zip(self.swa_model.parameters(), self.client_model.parameters()):
            param1.data *= (1.0 - alpha)
            param1.data += param2.data * alpha
        self.swa_n += 1
    
    def _update_global_model_gradient(self, pseudo_gradient):
        """Args:
            pseudo_gradient: global pseudo gradient, i.e. weighted average of the trained clients' deltas.

        Updates the global model gradient as -1.0 * pseudo_gradient
        """
        params_dict = zip(self.model.state_dict().keys(), pseudo_gradient)
        state_dict = OrderedDict()
        for k, v in params_dict:
            state_dict[k] = torch.Tensor(v)
        for n, p in self.client_model.named_parameters():
            p.grad = -1.0 * pseudo_gradient[n]
        self.server_opt.step()
        bn_layers = OrderedDict(
            {k: v for k, v in pseudo_gradient.items() if "running" in k or "num_batches_tracked" in k})
        self.client_model.load_state_dict(bn_layers, strict=False)


class FedNova(FedAvg):
    """FedNova."""
    def __init__(self, lr, gmf , *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Maintain a momentum buffer for the weight updates across rounds of training
        self.global_momentum_buffer: List[NDArray] = []
        if self.initial_parameters is not None:
            self.global_parameters: List[NDArray] = parameters_to_ndarrays(
                self.initial_parameters
            )

        self.lr = lr

        # momentum parameter for the server/strategy side momentum buffer
        self.gmf = gmf

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ):
        """Aggregate the results from the clients."""
        if not results:
            return None, {}

        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        # Compute tau_effective from summation of local client tau: Eqn-6: Section 4.1
        local_tau = [res.metrics["tau"] for _, res in results]
        tau_eff = np.sum(local_tau)

        aggregate_parameters = []

        for _client, res in results:
            params = parameters_to_ndarrays(res.parameters)
            # compute the scale by which to weight each client's gradient
            # res.metrics["local_norm"] contains total number of local update steps
            # for each client
            # res.metrics["weight"] contains the ratio of client dataset size
            # Below corresponds to Eqn-6: Section 4.1
            scale = tau_eff / float(res.metrics["local_norm"])
            scale *= float(res.metrics["weight"])

            aggregate_parameters.append((params, scale))

        # Aggregate all client parameters with a weighted average using the scale
        # calculated above
        agg_cum_gradient = aggregate(aggregate_parameters)

        # In case of Server or Hybrid Momentum, we decay the aggregated gradients
        # with a momentum factor
        self.update_server_params(agg_cum_gradient)

        return ndarrays_to_parameters(self.global_parameters), {}

    def update_server_params(self, cum_grad: NDArrays):
        """Update the global server parameters by aggregating client gradients."""
        for i, layer_cum_grad in enumerate(cum_grad):
            if self.gmf != 0:
                # check if it's the first round of aggregation, if so, initialize the
                # global momentum buffer

                if len(self.global_momentum_buffer) < len(cum_grad):
                    buf = layer_cum_grad / self.lr
                    self.global_momentum_buffer.append(buf)

                else:
                    # momentum updates using the global accumulated weights buffer
                    # for each layer of network
                    self.global_momentum_buffer[i] *= self.gmf
                    self.global_momentum_buffer[i] += layer_cum_grad / self.lr

                self.global_parameters[i] -= self.global_momentum_buffer[i] * self.lr

            else:
                # weight updated eqn: x_new = x_old - gradient
                # the layer_cum_grad already has all the learning rate multiple
                self.global_parameters[i] -= layer_cum_grad


class Fed(FedAvg):
    def __init__(self,**kwargs) -> None:
        super().__init__(**kwargs)