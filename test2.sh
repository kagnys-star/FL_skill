#['microwave', 'kettle', 'slidecabinet', 'hingecabinet', 'bottomburner', 'lightswitch', 'topburner']
export EXP_DIR=./experiments
export DATA_DIR=./data
#python3 spirl/train.py --path=spirl/configs/skill_prior_learning/kitchen/hierarchical_cl --val_data_size=160
#CUDA_VISIBLE_DEVICES=0 python3 spirl/rl/trains.py --path=spirl/configs/hrl/half_cheetah/spirl_cl --seed=0 --prefix=cheetah --resume=/home/kangys/workspace/FL_skill/experiments/hrl/half_cheetah/spirl_cl/hetero-2_cheetah-3-1/weights/weights_ep2.pth --mode=val
CUDA_VISIBLE_DEVICES=0 mpirun -np 9 python3 spirl/rl/trains.py --path=spirl/configs/hrl/mt10/spirl_cl --seed=0 --prefix=wtf8
#CUDA_VISIBLE_DEVICES=0 mpirun -np 9 python3 spirl/rl/trains.py --path=spirl/configs/hrl/mt10/spirl_cl --seed=0 --prefix=button-press-test_1
#CUDA_VISIBLE_DEVICES=0 python3 spirl/train.py --path=spirl/configs/skill_prior_learning/half_cheetah/hierarchical_cl --prefix=what_is
#CUDA_VISIBLE_DEVICES=0 python3 spirl/rl/train.py --path=spirl/configs/hrl/metaworld/MT_10 --seed=0 --prefix=meta_10
#CUDA_VISIBLE_DEVICES=0 python3 spirl/hydra_fl.py
#CUDA_VISIBLE_DEVICES=0 python3 spirl/FL_SPIRL_client2.py
#CUDA_VISIBLE_DEVICES=0 python3 spirl/train.py --path=spirl/configs/skill_prior_learning/metaworld
#CUDA_VISIBLE_DEVICES=0 python3 spirl/train.py --path=spirl/configs/skill_prior_learning/mt1/hierarchical_cl --prefix=mixing_data_test-4