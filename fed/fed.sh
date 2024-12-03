export EXP_DIR=/home/kangys/workspace/FL_skill/experiments
export DATA_DIR=/home/kangys/workspace/FL_skill/data/cheetah/3




#!/bin/bash
set -e

# start server, wait before launching clients
CUDA_VISIBLE_DEVICES=0 python server.py --path=/home/kangys/workspace/FL_skill/spirl/configs/skill_prior_learning/half_cheetah/feddyn \
                                        --data_dir=/home/kangys/workspace/FL_skill/data/cheetah/0\
                                        --prefix=cheetah-feddyn-hetero2_server-0 &
#: <<'END'
sleep 3


# start clients
for i in `seq 0 3`; do
    echo "Starting client $i"
    CUDA_VISIBLE_DEVICES=0 python3 client.py --path=/home/kangys/workspace/FL_skill/spirl/configs/skill_prior_learning/half_cheetah/feddyn \
        --prefix=cheetah-feddyn-hetero2_client_${i}-0\
        --data_dir=${DATA_DIR}/FL_${i} &
    sleep 10
done
#END

# enable CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM

# wait for all background processes to complete
wait