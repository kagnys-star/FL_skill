from collections import OrderedDict
from spirl.models.closed_loop_spirl_mdl import ClSPiRLMdl
from spirl.rl.components.replay_buffer import UniformReplayBuffer
from spirl.components.evaluator import TopOfNSequenceEvaluator
from spirl.rl.policies.cl_model_policies import ClModelPolicy
from spirl.rl.components.critic import MLPCritic
from spirl.rl.agents.ac_agent import SACAgent
from spirl.modules.variational_inference import MultivariateGaussian
from spirl.utils.pytorch_utils import no_batchnorm_update
import os
import datetime
import matplotlib; matplotlib.use('Agg')
import torch
import h5py
import numpy as np
import pandas as pd
from spirl.utils.general_utils import AttrDict , map_dict  
from spirl.utils.pytorch_utils import map2torch
from torch import autograd
from spirl.utils.wandb import WandBLogger

WANDB_PROJECT_NAME = 'fl-skill'
WANDB_ENTITY_NAME = 'yskang'

#init_path= "/home/workspace/skill/experiments/skill_prior_learning/kitchen/hierarchical_cl/normal-learning4/weights/weights_ep999.pth"
#init_path = "/home/workspace/skill/experiments/skill_prior_learning/kitchen/non-iid/weights/round-1000-weights.npz"

def datetime_str():
    return datetime.datetime.now().strftime("_%Y-%m-%d_%H-%M-%S")

def pytorch_model_load(config,init_path):
    basemodel =  ClSPiRLMdl(params = config, logger= None)
    basemodel.load_state_dict(torch.load(init_path)['state_dict'], strict=False)
    return basemodel

def gl_numpy_model_load(config,init_path):
    basemodel =  ClSPiRLMdl(params = config, logger= None)
    np_dict = np.load(init_path,allow_pickle=True)
    key_value = basemodel.state_dict().keys()
    params_dict = zip(key_value,np_dict)
    state_dict = OrderedDict()
    for k, v in params_dict:
        state_dict[k] = torch.Tensor(np_dict[v]).to(basemodel.device)
    np_dict.close()
    l = []
    for d in state_dict :
        if "num_batches_tracked" in d :
            l.append(d)
    for d in l :
        del(state_dict[d])
    basemodel.load_state_dict(state_dict,strict = True)
    return basemodel

def ll_numpy_model_load(config,init_path):
    basemodel =  ClSPiRLMdl(params = config, logger= None)
    np_dict = np.load(init_path,allow_pickle=True)
    key_value = basemodel.state_dict().keys()
    params_dict = zip(key_value, np_dict["arr_0"])
    state_dict = OrderedDict()
    for k, v in params_dict:
        state_dict[k] = torch.Tensor(v).to(basemodel.device)
    np_dict.close()
    l = []
    for d in state_dict :
        if "num_batches_tracked" in d :
            l.append(d)
    for d in l :
        del(state_dict[d])
    basemodel.load_state_dict(state_dict,strict = True)
    return basemodel

def ll_numpy_model_load_change(config,prior_path,init_path):
    basemodel =  ClSPiRLMdl(params = config, logger= None)
    basemodel.load_state_dict(torch.load(init_path)['state_dict'])
    chagnge_model =  ClSPiRLMdl(params = config, logger= None)
    np_dict = np.load(init_path,allow_pickle=True)
    pr_dict = np.load(prior_path)
    key_value = basemodel.state_dict().keys()
    params_dict = zip(key_value, np_dict["arr_0"], pr_dict)
    state_dict = OrderedDict()
    for k, v ,p in params_dict:
        if "p.0" in k :
            state_dict[k] = torch.Tensor(pr_dict[p]).to(basemodel.device)
        else:
            state_dict[k] = torch.Tensor(v).to(basemodel.device)
    np_dict.close()
    l = []
    for d in state_dict :
        if "num_batches_tracked" in d :
            l.append(d)
        if "classifier" in d :
            l.append(d)
    for d in l :
        del(state_dict[d])
    chagnge_model.load_state_dict(state_dict,strict = True)
    return chagnge_model

def gl_numpy_model_load_change(config,init_path):
    basemodel =  ClSPiRLMdl(params = config, logger= None)
    chagnge_model =  ClSPiRLMdl(params = config, logger= None)
    np_dict = np.load(init_path,allow_pickle=True)
    key_value = basemodel.state_dict().keys()
    params_dict = zip(key_value, np_dict)
    state_dict = OrderedDict()
    for k, v in params_dict:
        state_dict[k] = torch.Tensor(np_dict[v]).to(basemodel.device)
    np_dict.close()
    l = []
    for d in state_dict :
        if "num_batches_tracked" in d :
            l.append(d)
        #if "classifier" in d :
        #    l.append(d)
    for d in l :
        del(state_dict[d])
    chagnge_model.load_state_dict(state_dict,strict = True)
    return chagnge_model

def gl_torch_model_load_change(config,init_path):
    basemodel =  ClSPiRLMdl(params = config, logger= None)
    chagnge_model =  ClSPiRLMdl(params = config, logger= None)
    basemodel.load_state_dict(torch.load(init_path)['state_dict'])
    state_dict = basemodel.state_dict()
    l = []
    for d in state_dict :
        if "classifier" in d :
            l.append(d)
    for f in l :
        del(state_dict[f])
    chagnge_model.load_state_dict(state_dict,strict = True)
    return chagnge_model

def model_val(model,config):
    exp_name = config.prefix + datetime_str()
    logdir = os.path.join("/home/kangys/workspace/FL_skill/spirl/experiments",exp_name)
    os.makedirs(logdir, exist_ok=True)
    logger = WandBLogger(exp_name, WANDB_PROJECT_NAME, entity=WANDB_ENTITY_NAME, path = logdir, conf= config) 
    dataset_class = config.dataset_spec.dataset_class
    phase = 'val'
    total_score = []
    # config 잘 만들어서 넣어라
    for i in range(7):
        score  = []
        data_dir =os.path.join(config.data_dir, 'FL_{}'.format(i))
        val_loader = dataset_class(data_dir, config, resolution=config.dataset_spec.res,
                            phase=phase, shuffle=phase, dataset_size=512). \
            get_data_loader(config.batch_size, 1)
        #evaluator 만들기
        evaluator = TopOfNSequenceEvaluator(config, logdir, 100,
                                            'mse', tb_logger=logger)
        print('Running Testing')
        with autograd.no_grad():
            for sample_batched in val_loader:
                inputs = AttrDict(map_dict(lambda x: x.to(model.device), sample_batched))
                with model.val_mode():
                    evaluator.eval(inputs, model)
                    score.append(evaluator.dump_results(i))
        total_score.append(score)
    return total_score

def get_z_space(model,config):
    model.eval()
    dataset_class = config.dataset_spec.dataset_class
    labels = []
    hole_data =[] 
    for i in range(7):
        output_list = []
        data_dir =os.path.join(config.data_dir, 'FL_{}'.format(i))
        val_loader = dataset_class(data_dir, config, resolution=config.dataset_spec.res,
                            phase="train", shuffle= "train", dataset_size = 512). \
            get_data_loader(config.batch_size, 1)
        with autograd.no_grad():
            for sample_batched in val_loader:
                    inputs = AttrDict(map_dict(lambda x: x.to(config.device), sample_batched))
                    output = model(inputs)
                    output_list.extend(output.z.cpu().tolist())
        hole_data.append(output_list)
        labels.append(np.repeat(i, len(output_list)))
    out_np = np.concatenate(hole_data)
    labels = np.concatenate(labels)
    name = config.prefix + '.npz'
    np.savez( os.path.join("/home/kangys/workspace/FL_skill/spirl/experiments",name), out_np , labels)

def find_skill(num_samples, model):
    samples = np.random.normal(0, 1, (num_samples, 10))
    model.model.eval()
    for i in range(7):
        success = 0
        data_dir =os.path.join("/home/kangys/workspace/FL_skill/data/kitchen", 'FL_{}'.format(i))
        filenames = None
        for root, dirs, files in os.walk(data_dir):
            for file in files:
                if file.endswith(".hdf5"):
                    filenames = h5py.File((os.path.join(root, file)), 'r')
                    #print("data_load : {}".format((os.path.join(root, file))))
        seq_end_idxs = np.where(filenames['terminals'])[0]
        start = 0
        for end_idx in seq_end_idxs:
            state=torch.Tensor(filenames['observations'][start]).unsqueeze(0)
            action=np.array(filenames['actions'][start])
            with autograd.no_grad():
                for z in samples:
                    z = torch.Tensor(z).unsqueeze(0)
                    decode_input = torch.cat((z, state), dim= -1)
                    output = model.decoder(decode_input.to(model.device))
                    is_skill = np.isclose(action, output.cpu().numpy(),rtol=0.06,)
                    if is_skill.all():
                        success = 1 + success
            start = end_idx+1
        print(success)

def save_checkpoint(basemodel,folder):
    state = {
            'epoch': 99,
            'global_step':0,
            'state_dict': basemodel.state_dict(),
            'optimizer': {},
            }
    os.makedirs(folder, exist_ok=True)
    torch.save(state, folder+"/weights_ep99.pth")
    print("Saved checkpoint!")

#def weight_divergence(eval_models):
    '''
    keys = []
    iid_values = []
    non_values = []
    semi_iid_values = []
    for key in non_iid_state_dict.keys():
        if "weight" in key:
            keys.append(key)
            non_values.append(weight_diverse(base_dict[key],non_iid_state_dict[key]))
            iid_values.append(weight_diverse(base_dict[key],iid_state_dict[key]))
            semi_iid_values.append(weight_diverse(base_dict[key],semi_iid_state_dict[key]))
    student_card = pd.DataFrame({'keys':keys,
                             'iid_values':iid_values,
                             'non_values': non_values,
                             "semi_iid_values" : semi_iid_values})
                             
    student_card.to_csv("/home/workspace/skill/experiments/real_result.csv")
    '''

def prior_sequence_action(obs,model):
    obs = map2torch(obs, device = model.device)
    z = model.compute_learned_prior(obs[None], first_only=True).detach().sample()
    inputs = torch.cat((obs[None],z),dim=-1)
    act = model.decoder(inputs)
    output_dist = MultivariateGaussian(mu=act, log_sigma=log_sigma[None].repeat(act.shape[0], 1))
    action = output_dist.rsample()
    return action[0]

if __name__ == "__main__":
    from spirl.configs.default_data_configs.mulstage import data_spec
    from spirl.rl.envs.mulstage import mulstage
    ## SPIRL MODEL CONFIGS
    model_config =AttrDict(
        state_dim=data_spec.state_dim,
        action_dim=data_spec.n_actions,
        n_rollout_steps=10,
        kl_div_weight=5e-4,
        nz_enc=128,
        nz_mid=128,
        n_processing_layers=5,
        cond_decode=True,
        device = torch.device('cuda').type,
        batch_size = 512
    )

    data_config = AttrDict()
    data_config.dataset_spec = data_spec
    data_config.dataset_spec.subseq_len = 11
    data_config.batch_size = 512
    data_config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu").type
    data_config.data_dir = "/home/kangys/workspace/FL_skill/data/mulstage/0"
    data_config.prefix = "task_id-iid"

    init_path="/home/kangys/workspace/FL_skill/experiments/skill_prior_learning/mulstage/fedsol/hetero/weights/round-300-weights.npz"
    basemodel =  gl_numpy_model_load(config=model_config, init_path=init_path)
    basemodel.to(torch.device("cuda"))
    #['reach-v2','door-open-v2','drawer-open-v2','button-press-v2']
    env_config = AttrDict(
    task_id=0,
    seed = 20241122,
    )
    log_sigma = torch.tensor(-50 * np.ones(data_spec.n_actions, dtype=np.float32),
                                       device=torch.device("cuda"), requires_grad=True)
    env = mulstage(env_config)
    with env.val_mode():
        basemodel.eval()
        with autograd.no_grad():
            success = 0
            for _ in range(10):
                done = False
                obs = env.reset()
                count = 0
                while count < 800 and not done:
                    count += 1
                    a = prior_sequence_action(obs, basemodel)
                    obs, reward, done, info = env.step(a)
                success += sum(info['is_success'])
                
        
    print(success/10)
    '''
    ll_policy_params = AttrDict(
        policy_model=ClSPiRLMdl,
        policy_model_params=model_config,
        policy_model_checkpoint=os.path.join("/home/kangys/workspace/FL_skill/experiments/skill_prior_learning/mt4/fedavg/iid/repeat/300"),
    )
    hl_critic_params = AttrDict(
    action_dim=model_config.action_dim,
    input_dim=model_config.state_dim,
    output_dim=1,
    n_layers=5,  # number of policy network laye
    nz_mid=256,
    action_input=True,
)
    ll_policy_params.update(model_config)
    # create LL SAC agent (by default we will only use it for rolling out decoded skills, not finetuning skill decoder)
    ll_agent_config = AttrDict(
        policy=ClModelPolicy,
        policy_params=ll_policy_params,
        critic=MLPCritic,                   # LL critic is not used since we are not finetuning LL
        critic_params=hl_critic_params,
        replay=UniformReplayBuffer,
        replay_params=AttrDict(),
)
    ll_agent = SACAgent(ll_agent_config)

    #basemodel.to(basemodel.device)
    #get_z_space(model=basemodel, config=data_config)
    #model_val(model=basemodel, config=data_config)
    #find_skill(num_samples = 10000, model=basemodel)
    #basemodel =  gl_numpy_model_load_change(config=model_config, init_path="/home/kangys/workspace/FL_skill/experiments/skill_prior_learning/half_cheetah/fedprox/hetro3/weights/round-500-weights.npz")
    #save_checkpoint(basemodel,folder="/home/kangys/workspace/FL_skill/experiments/skill_prior_learning/half_cheetah/fedprox/hetro3/weights")
    '''