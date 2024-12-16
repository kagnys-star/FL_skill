export EXP_DIR=/home/kangys/workspace/FL_skill/experiments
export DATA_DIR=/home/kangys/workspace/FL_skill/data/6_mix_task/4

#!/bin/bash
set -e

# start server, wait before launching clients
CUDA_VISIBLE_DEVICES=0 python server.py --path=/home/kangys/workspace/FL_skill/spirl/configs/skill_prior_learning/drawer_open/feddyn \
                                        --data_dir=${DATA_DIR} \
                                        --exp_mode=feddyn \
                                        --prefix=6_mix_task-feddyn-one_server-3 &
#: <<'END'
sleep 3


# start clients
for i in `seq 0 3`; do
    echo "Starting client $i"
    CUDA_VISIBLE_DEVICES=0 python3 client.py --path=/home/kangys/workspace/FL_skill/spirl/configs/skill_prior_learning/drawer_open/feddyn \
        --prefix=6_mix_task-feddyn-one_client_${i}-3 \
        --exp_mode=feddyn \
        --data_dir=${DATA_DIR}/FL_${i} &
    sleep 10
done
#END


# enable CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM

# wait for all background processes to complete
wait