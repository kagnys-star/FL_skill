export EXP_DIR=./experiments
export DATA_DIR=./data


#mpirun -np 9
# 경로가 없는 경우 생성
save_path="$EXP_DIR/skill_prior_learning/mulstage/fedsol/hetero/repeat/300"
object_name="mulstage_fedsol_hetero"
# Python 스크립트 실행
for j in $(seq 0 2); do
env_name="${object_name}_35_${j}_300"
CUDA_VISIBLE_DEVICES=0 mpirun -np 9 python3 spirl/rl/train_save.py \
        --csv="${object_name}.csv" \
        --path=/home/kangys/workspace/FL_skill/spirl/configs/hrl/mulstage/spirl_cl --seed=20241210 --prefix=$env_name\
        --config_override=agent.hl_agent_params.policy_params.prior_model_checkpoint=${save_path},\
agent.ll_agent_params.policy_params.policy_model_checkpoint=${save_path},\
env.task_id=${j}
done
# enable CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTER

# wait for all background processes to complete
wait