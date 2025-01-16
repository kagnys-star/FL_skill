export EXP_DIR=./experiments
export DATA_DIR=./data

#!/bin/bash
set -e

exp_list=("fedavg")
alpha_list=("one")

# enable CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM

for mode in "${exp_list[@]}"; do
    for alpha in "${alpha_list[@]}"; do
        # 경로가 없는 경우 생성
        save_path="${EXP_DIR}/skill_prior_learning/6_mix_task/${mode}/${alpha}/repeat/300"
        object_name="6_mix_task_${mode}_${alpha}"
        # Python 스크립트 실행
        for j in $(seq 0 23); do
        #여기서 바꾸는거
            env_name="${object_name}_${j}_300"
            CUDA_VISIBLE_DEVICES=0 mpirun -np 9 python3 spirl/rl/train_save.py \
                    --csv="${object_name}.csv" \
                    --path=/home/kangys/workspace/FL_skill/spirl/configs/hrl/mt10/spirl_cl --seed=0 --prefix=$env_name \
                    --config_override=agent.hl_agent_params.policy_params.prior_model_checkpoint=${save_path},\
agent.ll_agent_params.policy_params.policy_model_checkpoint=${save_path}            
            wait
        done
    done
done


# wait for all background processes to complete
wait