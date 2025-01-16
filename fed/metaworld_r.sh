#!/bin/bash
set -e

#exp_list=("fedavg" "fedasam" "fedprox" "fednova" "fedsol" "feddyn")
#alpha_list=("iid" "hetero3" "hetero2")
exp_list=("fednova" "fedsol" "feddyn" "fedprox")
alpha_list=("iid" )

for mode in "${exp_list[@]}"; do
    #for alpha in `seq 0 0`; do
    export EXP_DIR=/home/kangys/workspace/FL_skill/experiments
    export DATA_DIR=/home/kangys/workspace/FL_skill/data/6_mix_task/1
    
    # start server, wait before launching clients
    echo "$mode"
    CUDA_VISIBLE_DEVICES=0 python server.py --path=/home/kangys/workspace/FL_skill/spirl/configs/skill_prior_learning/drawer_open/${mode} \
                                            --data_dir=${DATA_DIR} \
                                            --exp_mode=${mode} \
                                            --prefix=6_mix_task-${mode}-${alpha_list[0]}_server_15 &
    server_pid=$!  # 서버 프로세스 ID 저장
    sleep 5

    # start clients
    for i in `seq 0 3`; do
        echo "Starting client $i"
        CUDA_VISIBLE_DEVICES=0 python3 client.py --path=/home/kangys/workspace/FL_skill/spirl/configs/skill_prior_learning/drawer_open/${mode} \
                                                --prefix=6_mix_task-${mode}-${alpha_list[0]}_client_${i}_15 \
                                                --exp_mode=${mode} \
                                                --data_dir=${DATA_DIR}/FL_${i} &
        client_pids+=($!)  # 각 클라이언트 프로세스 ID 저장
        sleep 10
    done

    # 서버와 모든 클라이언트가 종료될 때까지 대기
    wait $server_pid
    for pid in "${client_pids[@]}"; do
        wait $pid
    done
    
    # 다음 반복을 위해 client_pids 배열 초기화
    unset server_pid
    unset client_pids
    #done
done

# enable CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM

# wait for all background processes to complete
wait
