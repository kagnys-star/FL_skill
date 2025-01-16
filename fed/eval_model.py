import os
import math
from scipy.spatial import distance
from sklearn.neighbors import NearestNeighbors
from scipy.linalg import sqrtm
from strategy import *
from spirl.components.params import get_args
import matplotlib; matplotlib.use('Agg')
import torch
import os
import time
from shutil import copy
import datetime
import imp
from tensorboardX import SummaryWriter
import numpy as np
import random
from torch import autograd
from spirl.modules.losses import L2Loss, CSLoss
from spirl.utils.general_utils import RecursiveAverageMeter, map_dict , AttrDict, ParamDict
from spirl.components.checkpointer import get_config_path
from spirl.utils.pytorch_utils import  ten2ar
from spirl.components.trainer_base import BaseTrainer
from spirl.utils.wandb import WandBLogger
from collections import OrderedDict
from spirl.modules.variational_inference import MultivariateGaussian
import pandas as pd
WANDB_PROJECT_NAME = 'fl-skill'
WANDB_ENTITY_NAME = 'yskang'


class ServerModel(BaseTrainer):
    def __init__(self, args):
        self.args = args
        self.setup_device()
        # set up params
        self.conf = conf = self.get_config()
        self._hp = self._default_hparams()
        self._hp.overwrite(conf.general)  # override defaults with config file
        self._hp.exp_path = make_path(conf.exp_dir, args.path, args.prefix, args.new_dir)
        self.log_dir = log_dir = os.path.join(self._hp.exp_path, 'events')
        print('using log dir: ', log_dir)
        self.conf = self.postprocess_conf(conf)
        if args.deterministic: set_seeds()
        # set up logging + training monitoring
        self.writer = self.setup_logging(conf, self.log_dir)

        # buld dataset, model. logger, etc.
        train_params = AttrDict(logger_class=self._hp.logger,
                                model_class=self._hp.model,
                                n_repeat=10,
                                dataset_size=-1)
        self.logger, self.model, self.train_loader = self.build_phase(train_params, 'train')

        self.evaluator = self._hp.evaluator(self._hp, self.log_dir, self._hp.top_of_n_eval,
                                            self._hp.top_comp_metric, tb_logger=self.logger)

    def _default_hparams(self):
        default_dict = ParamDict({
            'model': None,
            'logger': None,
            'evaluator': None,
            'data_dir': None,  # directory where dataset is in
            'batch_size': 32,
            'exp_path': None,  # Path to the folder with experiments
            'num_epochs': 200,
            'epoch_cycles_train': 1,
            'optimizer': 'radam',    # supported: 'adam', 'radam', 'rmsprop', 'sgd'
            'minimizer ' : None,
            'top_of_n_eval': 1,     # number of samples used at eval time
            'top_comp_metric': None,    # metric that is used for comparison at eval time (e.g. 'mse')
            'logging_target': 'wandb',
        })
        return default_dict

    def val(self , rounds):
        print('server Testing')
        save_data = {}
        start = time.time()
        losses_meter = RecursiveAverageMeter()
        iw_meter = RecursiveAverageMeter()
        z_samples  = []
        log_q_zx_samples = []
        z_hat_samples  = []
        log_q_zx_hat_samples = []
        mu_samples = []
        real_samples = []
        fake_samples = []
        self.model.eval()
        self.evaluator.reset()
        with autograd.no_grad():
            for sample_batched in self.train_loader:
                inputs = AttrDict(map_dict(lambda x: x.to(self.device), sample_batched))
                # run evaluator with val-mode model
                with self.model.val_mode():
                    self.evaluator.eval(inputs, self.model)
                # run non-val-mode model (inference) to check overfitting
                output = self.model(inputs)
                losses = self.model.loss(output, inputs)
                output.q_reconstruction = self.model.decode(output.z_p,
                                            cond_inputs=self.model._learned_prior_input(inputs),
                                            steps=self.model._hp.n_rollout_steps,
                                            inputs=inputs)
                losses.prior_mse = L2Loss(1.0)(inputs.actions,(output.q_reconstruction))
                losses.act_mse = L2Loss(1.0)(output.reconstruction,(output.q_reconstruction))
                losses.z_mse = L2Loss(1.0)(output.z,(output.z_p))
                losses.act_cos = CSLoss(1.0)(output.reconstruction,(output.q_reconstruction))
                losses.z_cos = CSLoss(1.0)(output.z,(output.z_p))
                iw_meter.update(AttrDict(
                        LL = self.importants_weights(output, inputs),
                        FID = self.calulate_fid(output, inputs)
                                        )
                                )
                losses_meter.update(losses)
                z_prior = MultivariateGaussian(torch.zeros_like(output.q.mu), torch.zeros_like(output.q.sigma))
                z_samples.append(ten2ar(z_prior.log_prob(output.z)))
                log_q_zx_samples.append(ten2ar(output.q.log_prob(output.z)))
                z_hat_samples.append(ten2ar(z_prior.log_prob(output.z_p)))
                log_q_zx_hat_samples.append(ten2ar(output.q_hat.log_prob(output.z_p)))
                mu_samples.append(ten2ar(output.q.mu))
                real_samples.append(ten2ar(output.z))
                fake_samples.append(ten2ar(output.z_p))
                del losses

            if self.evaluator is not None:
                self.evaluator.dump_results(rounds)
            log_q_z = np.concatenate(np.array(z_samples), axis=0)
            log_q_z_hat = np.concatenate(np.array(z_hat_samples), axis=0)
            #q(z|x)
            log_q_zx = np.concatenate(np.array(log_q_zx_samples), axis=0)
            log_q_zx_hat = np.concatenate(np.array(log_q_zx_hat_samples), axis=0)
            #p(z)
            mi_estimate = (log_q_zx - log_q_z)
            mi_hat_estimate = (log_q_z_hat - log_q_zx_hat)
            mu_s = np.concatenate(np.array(mu_samples), axis=0)
            z_var = np.var(mu_s, axis=0, ddof=1)
            threshold = 0.01
            activated_units = (z_var > threshold).astype(float)
            real_x = np.concatenate(np.array(real_samples), axis=0)
            #avg_dist, median_dist = self.calculate_average_distance(real_x)
            #epsilon = avg_dist * 0.3
            precision, recall = self.compute_precision_recall_torch(real_x, np.concatenate(np.array(fake_samples), axis=0), epsilon=1.3)
            k_precision, k_recall = self.compute_prd(real_x, np.concatenate(np.array(fake_samples), axis=0), k=10)
            self.logger.log_scalar(np.mean(activated_units), "AU", rounds, "val")
            #self.logger.log_scalar(np.mean(log_q_z), "log_q_z", rounds, "val")
            #self.logger.log_scalar(np.mean(log_q_zx), "log_q_zx", rounds, "val")
            self.logger.log_scalar(np.mean(mi_estimate), "mi_estimate", rounds, "val")
            self.logger.log_scalar_dict(iw_meter.avg ,'val', rounds )
            self.model.log_outputs(output, inputs, losses_meter.avg, rounds,
                                        log_images=False, phase='val', **self._logging_kwargs)
            print(('\nTest set: Average loss: {:.4f} in {:.2f}s\n'
                    .format(losses_meter.avg.total.value.item(), time.time() - start)))
            save_data["activated_units"] = np.mean(activated_units)
            save_data["mi"] = np.mean(mi_estimate)
            save_data["prior_mi"] = np.mean(mi_hat_estimate)
            save_data["e_precision"] = precision
            save_data["e_recall"] = recall
            save_data["k_precision"] = k_precision
            save_data["k_recall"] = k_recall
            #save_data["epsilon"] = epsilon
            for key , values in losses_meter.avg.items():
                save_data[key] = values["value"].item()
            for key , values in iw_meter.avg.items():
                save_data[key] =values
        del output , inputs
        return save_data


    def setup_device(self):
        self.use_cuda = torch.cuda.is_available() and not self.args.debug
        self.device = torch.device('cuda') if self.use_cuda else torch.device('cpu')
        if self.args.gpu != -1:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.args.gpu)

    def get_config(self):
        conf = AttrDict()

        # paths
        conf.exp_dir = self.get_exp_dir()
        conf.conf_path = get_config_path(self.args.path)

        # general and model configs
        print('loading from the config file {}'.format(conf.conf_path))
        conf_module = imp.load_source('conf', conf.conf_path)
        conf.general = conf_module.configuration
        conf.model = conf_module.model_config

        # data config
        try:
            data_conf = conf_module.data_config
        except AttributeError:
            data_conf_file = imp.load_source('dataset_spec', os.path.join(AttrDict(conf).data_dir, 'dataset_spec.py'))
            data_conf = AttrDict()
            data_conf.dataset_spec = AttrDict(data_conf_file.dataset_spec)
            data_conf.dataset_spec.split = AttrDict(data_conf.dataset_spec.split)
        conf.data = data_conf

        # model loading config
        conf.ckpt_path = conf.model.checkpt_path if 'checkpt_path' in conf.model else None

        return conf

    def postprocess_conf(self, conf):
        conf.model['batch_size'] = self._hp.batch_size if not torch.cuda.is_available() \
            else int(self._hp.batch_size / torch.cuda.device_count())
        conf.model.update(conf.data.dataset_spec)
        conf.model['device'] = conf.data['device'] = self.device.type
        return conf

    def setup_logging(self, conf, log_dir):
        if not self.args.dont_save:
            print('Writing to the experiment directory: {}'.format(self._hp.exp_path))
            if not os.path.exists(self._hp.exp_path):
                os.makedirs(self._hp.exp_path)
            if self._hp.logging_target == 'wandb':
                exp_name = f"{'_'.join(self.args.path.split('/')[-3:])}_{self.args.prefix}" if self.args.prefix \
                    else os.path.basename(self.args.path)
                writer = WandBLogger(exp_name, WANDB_PROJECT_NAME, entity=WANDB_ENTITY_NAME,
                                     path=self._hp.exp_path, conf=conf, exclude=['model_rewards', 'data_dataset_spec_rewards'])
            else:
                writer = SummaryWriter(log_dir)
        else:
            writer = None

        # set up additional logging args
        self._logging_kwargs = AttrDict(
        )
        return writer


    def build_phase(self, params, phase):
        if not self.args.dont_save:
            if self._hp.logging_target == 'wandb':
                logger = self.writer
            else:
                logger = params.logger_class(self.log_dir, summary_writer=self.writer)
        else:
            logger = None
        model = params.model_class(self.conf.model, logger)
        if torch.cuda.device_count() > 1:
            raise ValueError("Detected {} devices. Currently only single-GPU training is supported!".format(torch.cuda.device_count()),
                             "Set CUDA_VISIBLE_DEVICES=<desired_gpu_id>.")
        model = model.to(self.device)
        model.device = self.device
        loader = self.get_dataset(self.args, model, self.conf.data, phase, params.n_repeat, params.dataset_size)
        return logger, model, loader

    def get_dataset(self, args, model, data_conf, phase, n_repeat, dataset_size=-1):
        dataset_class = data_conf.dataset_spec.dataset_class
        loader = dataset_class(self.args.data_dir, data_conf, resolution=model.resolution,
                               phase=phase, shuffle=phase == "train", dataset_size=dataset_size). \
            get_data_loader(self._hp.batch_size, n_repeat)
        return loader

    def get_exp_dir(self):
        return os.environ['EXP_DIR']

    def importants_weights(self, output, inputs, k=50):
        log_weights = []
        z_prior = MultivariateGaussian(torch.zeros_like(output.q.mu),torch.zeros_like(output.q.sigma))
        for _ in range(k):
            z_sample = output.q.sample()  # (batch_size, latent_dim)
            x_reconstructed = self.model.decode(z_sample,
                                            cond_inputs=self.model._learned_prior_input(inputs),
                                            steps=self.model._hp.n_rollout_steps,
                                            inputs=inputs)
            mse = torch.sum((inputs.actions - x_reconstructed)**2, dim=-1)
            log_px_given_z = z_prior.log_prob(mse) # 추가텀 확장
            log_pz = z_prior.log_prob(z_sample)
            log_qz_given_x = output.q.log_prob(z_sample)
            log_weight = log_px_given_z + log_pz - log_qz_given_x
            log_weights.append(log_weight)
        log_weights = torch.stack(log_weights, dim=0)
        log_likelihood = torch.logsumexp(log_weights, dim=0) - torch.log(torch.tensor(k, device=z_sample.device))
        return  log_likelihood.mean().item()
    
    def calulate_fid(self, output, inputs):
        mu1 = torch.mean(output.q.mu, dim=0).cpu().numpy()
        mu2 = torch.mean( output.q_hat.mu, dim=0).cpu().numpy()
        sigma1 =  torch.diag_embed( output.q_hat.sigma).mean(dim=0).cpu().numpy()
        sigma2 =  torch.diag_embed( output.q.sigma).mean(dim=0).cpu().numpy()
        diff = mu1 - mu2
        diff_squared = np.sum(diff**2)

        # Compute the square root of the product of covariance matrices
        covmean = sqrtm(sigma1 @ sigma2)
        if np.iscomplexobj(covmean):  # Handle numerical issues
            covmean = covmean.real

        # Compute the FID
        trace_covmean = np.trace(covmean)
        fid = diff_squared + np.trace(sigma1) + np.trace(sigma2) - 2 * trace_covmean
        return fid
    
    def calculate_average_distance(self,data):
        data2 = torch.Tensor(data).to(self.device)
        distances = torch.cdist(data2, data2)  # Pairwise 거리 계산
        upper_triangle = distances[torch.triu(torch.ones_like(distances), diagonal=1) > 0]  # 대각선 위 삼각형만
        avg_distance = upper_triangle.mean().item()  # 평균 거리
        median_distance = upper_triangle.median().item()  # 중앙값 거리
        return avg_distance, median_distance

    def compute_precision_recall_torch(self,real_data, gen_data, epsilon=0.1):
        """
        PyTorch를 사용하여 Precision과 Recall 계산
        Args:
            real_data (torch.Tensor): 실제 데이터 (N_real x D)
            gen_data (torch.Tensor): 생성된 데이터 (N_gen x D)
            k (int): K 값 (k-NN)
            device (str): GPU/CPU 선택
        Returns:
            precision (float): Precision 값
            recall (float): Recall 값
        """
        # 유클리드 거리 계산 함수
        def pairwise_distances(x, y):
            """
            두 배열 x, y 간의 유클리드 거리 계산
            Args:
                x (numpy.ndarray): 첫 번째 데이터셋 (N1 x D)
                y (numpy.ndarray): 두 번째 데이터셋 (N2 x D)
            Returns:
                numpy.ndarray: x와 y 간의 pairwise 거리 행렬 (N1 x N2)
            """
            return np.sqrt(np.sum((x[:, np.newaxis, :] - y[np.newaxis, :, :]) ** 2, axis=-1))

        # 실제 데이터 -> 생성 데이터 거리 계산
        real_to_gen_dist = pairwise_distances(real_data, gen_data)
        # 생성 데이터 -> 실제 데이터 거리 계산
        gen_to_real_dist = pairwise_distances(gen_data, real_data)

        # Precision: 생성된 데이터가 실제 manifold에 포함된 비율
        precision = np.mean(np.min(real_to_gen_dist, axis=1) <= epsilon)

        # Recall: 실제 데이터가 생성된 manifold에 포함된 비율
        recall = np.mean(np.min(gen_to_real_dist, axis=1) <= epsilon)

        return precision, recall
        '''
        real_data = torch.Tensor(real_data).to(self.device)
        gen_data = torch.Tensor(gen_data).to(self.device)
        real_to_gen_dist = torch.cdist(real_data, gen_data)  # 실제 -> 생성 거리
        gen_to_real_dist = torch.cdist(gen_data, real_data)  # 생성 -> 실제 거리

        # Precision: 생성된 데이터가 실제 manifold에 포함된 비율
        precision = (real_to_gen_dist.min(dim=1).values <= epsilon).float().mean().item()

        # Recall: 실제 데이터가 생성된 manifold에 포함된 비율
        recall = (gen_to_real_dist.min(dim=1).values <= epsilon).float().mean().item()

        return precision, recall
        '''
    def compute_prd(self, real_samples, generated_samples, k=10 , mean = True):
        """
        Compute Precision and Recall for Distributions (PRD).

        Args:
            real_samples (np.ndarray): Samples from the real data distribution (N_real x D).
            generated_samples (np.ndarray): Samples from the generated distribution (N_gen x D).
            k (int): Number of nearest neighbors to use.

        Returns:
            precision (float): Proportion of generated samples close to real data.
            recall (float): Proportion of real samples close to generated data.
        """
        # Nearest Neighbors for Real Samples
        nn_real = NearestNeighbors(n_neighbors=k).fit(real_samples)
        distances_real, _ = nn_real.kneighbors(generated_samples)

        # Precision: Proportion of generated samples within the k-nearest real samples
        precision = np.mean(np.min(distances_real, axis=1) <= np.mean(distances_real))


        # Nearest Neighbors for Generated Samples
        nn_gen = NearestNeighbors(n_neighbors=k).fit(generated_samples)
        distances_gen, _ = nn_gen.kneighbors(real_samples)

        # Recall: Proportion of real samples within the k-nearest generated samples
        recall = np.mean(np.min(distances_gen, axis=1) <= np.mean(distances_gen))
        return precision, recall


def save_checkpoint(state, folder, filename='checkpoint.pth'):
    os.makedirs(folder, exist_ok=True)
    torch.save(state, os.path.join(folder, filename))
    print(f"Saved checkpoint to {os.path.join(folder, filename)}!")

def datetime_str():
    return datetime.datetime.now().strftime("_%Y-%m-%d_%H-%M-%S")


def make_path(exp_dir, conf_path, prefix, make_new_dir):
    # extract the subfolder structure from config path
    path = conf_path.split('configs/', 1)[1]
    if make_new_dir:
        prefix += datetime_str()
    base_path = os.path.join(exp_dir, path)
    return os.path.join(base_path, prefix) if prefix else base_path


def set_seeds(seed=0, cuda_deterministic=True):
    """Sets all seeds and disables non-determinism in cuDNN backend."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    if torch.cuda.is_available() and cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

def save_config(conf_path, exp_conf_path):
    copy(conf_path, exp_conf_path)


if __name__ == "__main__":
    """Load model for
    1. server-side parameter initialization
    2. server-side parameter evaluation
    """
    args = get_args()
    data_dir = args.data_dir[:-1] + "0"
    args.data_dir = data_dir
    file_path = os.path.join(os.environ['EXP_DIR'],"vae_info","mulstage_elipson.csv")
    parent_dir = os.path.dirname(file_path)
    # 파일을 저장할 디렉토리가 존재하지 않으면 생성
    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir)
    init_model = ServerModel(args=args)
    experiments  = ["zero"] # ["fedavg" , "fedprox" , "fedasam" , "fednova" , "fedsol" , "feddyn"] # 
    for exp in experiments :
        
        #init_path  = f"/home/kangys/workspace/FL_skill/experiments/skill_prior_learning/mulstage/{exp}/iid/weights/round-300-weights.npz"
        init_path  = f"/home/kangys/workspace/FL_skill/experiments/skill_prior_learning/testmodel/prior_sim2/hetero2/weights/round-300-weights.npz"
        np_dict =  np.load(init_path,allow_pickle=True)
        key_value = init_model.model.state_dict().keys()
        params_dict = zip(key_value, np_dict)
        state_dict = OrderedDict()
        for k, v in params_dict:
            state_dict[k] = torch.Tensor(np_dict[v]).to(init_model.device)
        np_dict.close()
        l = []
        for d in state_dict :
            if "num_batches_tracked" in d :
                l.append(d)
        for d in l :
            del(state_dict[d])
        
        #checkpoint = torch.load("/home/kangys/workspace/FL_skill/experiments/skill_prior_learning/cheetah/weights_ep99.pth", map_location=init_model.device)
        #state_dict = checkpoint['state_dict']
        init_model.model.load_state_dict(state_dict,strict = True)
        data = init_model.val(100)
        data["mode"] = exp
        print(data)
        '''
        # 파일이 없으면 CSV 파일 생성
        if not os.path.exists(file_path):
            df = pd.DataFrame.from_dict(data=[data])
            df = df.sort_index(axis=1)
            #df = pd.DataFrame(columns=['scores' , 'rounds' ])  # 원하는 컬럼명으로 DataFrame 생성
            #df = pd.concat([df, pd.DataFrame({'scores': [value] , 'rounds' : [rounds]})], ignore_index=True)
            df.to_csv(file_path, index=False)
        else:
            # 파일이 있으면 파일 열기
            
            df = pd.read_csv(file_path)
            # 새로운 값을 추가
            df = pd.concat([df, pd.DataFrame.from_dict(data=[data])], ignore_index=True)
            df.to_csv(file_path, index=False)
        '''