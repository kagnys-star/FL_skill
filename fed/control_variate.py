import pickle
import numpy as np
from pprint import pprint

with open('c_i_client_0-1.pickle', 'rb') as fr:
    user_loaded1 = pickle.load(fr)

with open('c_i_client_1-1.pickle', 'rb') as fr:
    user_loaded2 = pickle.load(fr)

with open('server_control_variate-1.pickle', 'rb') as fr:
    server = pickle.load(fr)


for i , k in enumerate( user_loaded1.keys() ):
    cv_mean = np.mean(user_loaded1[k] - server[i])
    cv_std = np.std(user_loaded1[k] - server[i])
    print(f"parameter : {k} -> {cv_mean}, variance : {cv_std}")

'''
for i , k in enumerate( user_loaded1.keys() ):
    mean_var = np.mean(user_loaded1[k] - server[k])
    print(f"parameter : {k} -> {mean_var}")
'''