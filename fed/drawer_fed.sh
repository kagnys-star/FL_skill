export EXP_DIR=/home/kangys/workspace/FL_skill/experiments
export DATA_DIR=/home/kangys/workspace/FL_skill/data/drawer_open/2

#!/bin/bash
set -e

# start server, wait before launching clients
CUDA_VISIBLE_DEVICES=0 python server.py --path=/home/kangys/workspace/FL_skill/spirl/configs/skill_prior_learning/drawer_open/fedasam \
                                        --data_dir=/home/kangys/workspace/FL_skill/data/drawer_open/0 \
                                        --exp_mode=fedasam \
                                        --prefix=drawer_open-fedasam-iid_server-12&
#: <<'END'
sleep 3


# start clients
for i in `seq 0 3`; do
    echo "Starting client $i"
    CUDA_VISIBLE_DEVICES=0 python3 client.py --path=/home/kangys/workspace/FL_skill/spirl/configs/skill_prior_learning/drawer_open/fedasam \
        --prefix=drawer_open-fedasam-iid_client_${i}-12 \
        --exp_mode=fedasam \
        --data_dir=${DATA_DIR}/FL_${i} &
    sleep 10
done
#END

# enable CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM

# wait for all background processes to complete
wait