#['microwave', 'kettle', 'slidecabinet', 'hingecabinet', 'bottomburner', 'lightswitch', 'topburner']
export EXP_DIR=/home/yskang/workspace/FL_skill/experiments
export DATA_DIR=/home/yskang/workspace/FL_skill/data/cheetah/1

CUDA_VISIBLE_DEVICES=0 python3 client.py --path=/home/yskang/workspace/FL_skill/spirl/configs/skill_prior_learning/half_cheetah/feddyn \
        --data_dir=${DATA_DIR}/FL_0 --prefix=0