export EXP_DIR=./experiments
export DATA_DIR=./data


#mpirun -np 9
# 경로가 없는 경우 생성
save_path="$EXP_DIR/skill_prior_learning/mulstage/fedavg/hetero/repeat/300"
object_name="mulstage_fedavg_hetero"
# Python 스크립트 실행
env_name="${object_name}_test4_300"
CUDA_VISIBLE_DEVICES=0 mpirun -np 9 python3 spirl/rl/train_save.py --csv="${object_name}.csv" \
        --path=/home/kangys/workspace/FL_skill/spirl/configs/hrl/mulstage/spirl_cl --seed=0 --prefix=$env_name\
        --config_override=agent.hl_agent_params.policy_params.prior_model_checkpoint=${save_path},\
agent.ll_agent_params.policy_params.policy_model_checkpoint=${save_path},\
env.task_id=0
# enable CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTER

# wait for all background processes to complete
wait