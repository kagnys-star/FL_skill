#['microwave', 'kettle', 'slidecabinet', 'hingecabinet', 'bottomburner', 'lightswitch', 'topburner']
export EXP_DIR=./experiments
export DATA_DIR=./data
#CUDA_VISIBLE_DEVICES=0 python3 spirl/rl/trains.py --path=spirl/configs/hrl/half_cheetah/spirl_cl --seed=0 --prefix=cheetah --resume=/home/kangys/workspace/FL_skill/experiments/hrl/half_cheetah/spirl_cl/hetero-2_cheetah-3-1/weights/weights_ep2.pth --mode=val
#CUDA_VISIBLE_DEVICES=0 python3 spirl/rl/trains.py --path=spirl/configs/hrl/half_cheetah/spirl_cl --seed=0 --prefix=prox_hetero2_cheetah-3
#CUDA_VISIBLE_DEVICES=0 python3 spirl/train.py --path=spirl/configs/skill_prior_learning/half_cheetah/hierarchical_cl --prefix=asam
#CUDA_VISIBLE_DEVICES=0 python3 spirl/rl/train.py --path=spirl/configs/hrl/metaworld/MT_10 --seed=0 --prefix=meta_10
#CUDA_VISIBLE_DEVICES=0 python3 spirl/hydra_fl.py
CUDA_VISIBLE_DEVICES=0 python3 spirl/FL_SPIRL_client2.py
#CUDA_VISIBLE_DEVICES=0 python3 spirl/train.py --path=spirl/configs/skill_prior_learning/metaworld