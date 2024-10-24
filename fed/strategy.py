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
from flwr.server.strategy.aggregate import aggregate , weighted_loss_avg
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


class FEDASAM(fl.server.strategy.Strategy):
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
        self.initial_evaluation_done = False


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
    def __init__(self, lr , gmf , *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lr = lr
        # Maintain a momentum buffer for the weight updates across rounds of training
        self.global_momentum_buffer: List[NDArray] = []
        if self.initial_parameters is not None:
            self.global_parameters: List[NDArray] = parameters_to_ndarrays(
                self.initial_parameters
            )
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

        # 전체 클라이언트의 데이터 크기를 합산
        total_data_size = np.sum([fit_res.num_examples for _, fit_res in results])

        # tau_eff 계산: 각 클라이언트의 tau 값에 데이터 비율을 곱한 후 합산
        local_tau = [res.metrics["tau"] * (res.num_examples / total_data_size) for _, res in results]
        tau_eff = np.sum(local_tau)  # 데이터 비율을 고려한 tau_eff 계산

        aggregate_parameters = []

        for _, res in results:
            params = parameters_to_ndarrays(res.parameters)
            client_data_size = res.num_examples  # 각 클라이언트의 데이터 크기

            # 데이터 비율 계산 (전체 데이터 대비 클라이언트 데이터 비율)
            data_ratio = client_data_size / total_data_size
            
            # tau_eff와 데이터 비율을 기반으로 가중치 조정
            scale = tau_eff * res.metrics["tau"] * data_ratio  # 데이터 비율을 적용한 스케일링
            aggregate_parameters.append((params, scale))

        # 클라이언트의 파라미터를 가중치 평균으로 합산
        agg_cum_gradient = aggregate(aggregate_parameters)

        # 서버 파라미터 업데이트
        self.update_server_params(agg_cum_gradient)
        if (self.global_parameters is not None) and (server_round % 5 == 0):
            # Save aggregated_ndarrays
            np.savez(os.path.join(self.save_dir,f"round-{server_round}-weights.npz"), *self.global_parameters)
            
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

                self.global_parameters[i] = self.global_parameters[i].astype(np.float64)
                self.global_parameters[i] -= self.global_momentum_buffer[i] * self.lr

            else:
                # weight updated eqn: x_new = x_old - gradient
                # the layer_cum_grad already has all the learning rate multiple
                self.global_parameters[i] = self.global_parameters[i].astype(np.float64)
                self.global_parameters[i] -= layer_cum_grad


class Scaffold(FedAvg):

    def __init__(self, save_dir, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        #Scaffold specific variables
        self.weight_shapes = list(map(lambda arr: arr.shape, parameters_to_ndarrays(self.initial_parameters)))
        self.covariates = [np.zeros(shape) for shape in self.weight_shapes]
        #store client_covariates on server side because clients are generated on demand. Instead of each client holding its state
        # as the Scaffold paper describes, we store the client_covariates on server side. This of course can be changed.
        self.clients_covariates: Dict[str, NDArrays] = {} # cid -> client_covariates
        self.covariates_zero = [np.zeros(shape) for shape in self.weight_shapes] # initial client covariates

    def initialize_parameters(
        self, client_manager
    ) -> Optional[Parameters]:
        """Initialize global model parameters."""
        return self._pack_weights_and_covariates(parameters_to_ndarrays(self.initial_parameters), self.covariates)

    def _unpack_parameters(self, parameters: Parameters) -> Tuple[NDArrays, NDArrays]:
        """Extract weights and covariates from parameters"""
        weights_and_covariates = parameters_to_ndarrays(parameters)
        self._check_shapes(weights_and_covariates)
        weights = weights_and_covariates[:len(self.weight_shapes)]
        covariates = weights_and_covariates[len(self.weight_shapes):]
        return weights, covariates

    def _pack_weights_and_covariates(
        self, 
        weights: NDArrays, 
        server_covariates: Optional[NDArrays] = None,
      ) -> Parameters:
        """Convert weights and covariates to parameters"""
        weights_and_covariates = (
            weights 
          + (server_covariates if server_covariates is not None else [])
        )
        self._check_shapes(weights_and_covariates)
        return ndarrays_to_parameters(weights_and_covariates)

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using weighted average."""

        if not results:
            return None, {}

        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        weights_results = []
        for client_proxy, fit_res in results:
          weights, delta_c = self._unpack_parameters(fit_res.parameters)
          self.clients_covariates[client_proxy.cid] = delta_c
          weights_results.append((weights, fit_res.num_examples))

        # equation (5)(i) - Scaffold paper
        # no need to use previous weights because the server learning rate is 1 
        # and so mathematically the update depends only on the updated weights from clients
        weights_aggregated = aggregate(weights_results)
        # equation (5)(ii) - Scaffold paper
        # similar trick as previously - no need to use previous server covariates
        client_covar = list(np.sum(list(self.clients_covariates.values()), axis=0) / len(self.clients_covariates))
        new_server_covariates = [
            x + y for x, y in zip(self.covariates, client_covar)
        ]
        
        parameters_aggregated = self._pack_weights_and_covariates(weights_aggregated, new_server_covariates)
        self.covariates = new_server_covariates

        if (weights_aggregated is not None) and (server_round % 5 == 0):
            # Convert `Parameters` to `List[np.ndarray]`
            # Save aggregated_ndarrays
            np.savez(os.path.join(self.save_dir,f"round-{server_round}-weights.npz"), *weights_aggregated)

        return parameters_aggregated, {}

    def _check_shapes(self, weights_and_covariates: NDArrays) -> None:
        """Given a list of numpy arrays checks whether they have a repeating pattern of given shapes"""
        assert len(weights_and_covariates) % len(self.weight_shapes) == 0
        for i in range(len(weights_and_covariates)):
            expected_shape = self.weight_shapes[i % len(self.weight_shapes)]
            assert weights_and_covariates[i].shape == expected_shape


class FedSOL(SM_FedAVG):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class Feddyn(SM_FedAVG):
    def __init__(self, dyn_alpha, n_clients, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dyn_alpha = dyn_alpha
        self.h_t = [np.array(param, dtype=np.float64) for param in parameters_to_ndarrays(self.initial_parameters)]
        self.global_parameters = [np.array(param, dtype=np.float64) for param in parameters_to_ndarrays(self.initial_parameters)]
        self.n_clients = n_clients

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using weighted average."""

        if not results:
            return None, {}

        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}
        
        # Step 1: Update h_t using client updates
        total_size = 0
        for _, fit_res in results:
            total_size += fit_res.num_examples
            for i , layers in enumerate(parameters_to_ndarrays(fit_res.parameters)):
                self.h_t[i] -= (self.dyn_alpha / self.n_clients) * (layers - self.global_parameters[i])
        
        # Step 2: Use only h_t to update the global weights
        new_global_weights = copy.deepcopy(self.h_t)
        for _, fit_res in results:
            for i , layers in enumerate(parameters_to_ndarrays(fit_res.parameters)):
                new_global_weights[i] = -(1.0/ self.dyn_alpha) * new_global_weights[i]
                new_global_weights[i] += (fit_res.num_examples / total_size) * (layers)
        self.global_parameters = new_global_weights

        if (new_global_weights is not None) and (server_round % 5 == 0):
            # Convert `Parameters` to `List[np.ndarray]`
            # Save aggregated_ndarrays
            np.savez(os.path.join(self.save_dir,f"round-{server_round}-weights.npz"), *new_global_weights)

        return ndarrays_to_parameters(self.global_parameters) , {}