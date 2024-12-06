import numpy as np
import torch
import pickle

with open('/home/kangys/workspace/FL_skill/experiments/skill_prior_learning/half_cheetah/sam/cheetah-fedasam-hetero3_client_1-6/nan_debug_info.pkl', 'rb') as f:
	data = pickle.load(f)

#dict_keys(['inputs', 'outputs', 'losses'])
#output : ['z', 'z_q', 'z_p', 'reconstruction']
checkpoint = torch.load('/home/kangys/workspace/FL_skill/experiments/skill_prior_learning/half_cheetah/sam/cheetah-fedasam-hetero3_client_1-6/nan_debug_ckpt.pth')
state_dict = checkpoint['state_dict']
q_dict = {}
for key , value in state_dict.items():
    if "p.0" in key:
        q_dict[key] = value
#print(q_dict.keys())
for name, param in q_dict.items():
    print(f'{name} gradient NaN check: ', param.grad)
#print(data["inputs"].keys())
#print(data["outputs"]["z_p"])