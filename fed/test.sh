#['microwave', 'kettle', 'slidecabinet', 'hingecabinet', 'bottomburner', 'lightswitch', 'topburner']
export EXP_DIR=/home/kangys/workspace/FL_skill/experiments
export DATA_DIR=/home/kangys/workspace/FL_skill/data/cheetah/0

CUDA_VISIBLE_DEVICES=0 python3 client.py --path=/home/kangys/workspace/FL_skill/spirl/configs/skill_prior_learning/half_cheetah/fedmmd \
        --exp_mode=test \
        --data_dir=${DATA_DIR} --prefix=half_cheetah-fedmmd-hetero2_server-23