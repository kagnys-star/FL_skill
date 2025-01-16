#['microwave', 'kettle', 'slidecabinet', 'hingecabinet', 'bottomburner', 'lightswitch', 'topburner']
export EXP_DIR=/home/kangys/workspace/FL_skill/experiments
export DATA_DIR=/home/kangys/workspace/FL_skill/data/half_cheetah/0

CUDA_VISIBLE_DEVICES=0 python3 client.py --path=/home/kangys/workspace/FL_skill/spirl/configs/skill_prior_learning/half_cheetah/feddez \
        --data_dir=${DATA_DIR} --prefix=half_cheetah-feddez-hetero2_server-23 --exp_mode=feddez
        #--exp_mode=fedopt \