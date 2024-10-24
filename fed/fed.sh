export EXP_DIR=/home/yskang/workspace/FL_skill/experiments
export DATA_DIR=/home/yskang/workspace/FL_skill/data/cheetah/1

#!/bin/bash
set -e

# start server, wait before launching clients
CUDA_VISIBLE_DEVICES=0 python server.py --path=/home/yskang/workspace/FL_skill/spirl/configs/skill_prior_learning/half_cheetah/feddyn \
                                        --data_dir=${DATA_DIR} \
                                        --prefix=cheetah-feddyn-iid_server_1 &
#: <<'END'
sleep 5

# start clients
for i in `seq 0 1`; do
    echo "Starting client $i"
    CUDA_VISIBLE_DEVICES=0 python3 client.py --path=/home/yskang/workspace/FL_skill/spirl/configs/skill_prior_learning/half_cheetah/feddyn \
        --prefix=cheetah-feddyn-iid_client_${i}_1 \
        --data_dir=${DATA_DIR}/FL_${i} &
    sleep 10
done
#END

# enable CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM

# wait for all background processes to complete
wait