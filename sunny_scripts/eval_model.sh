!/bin/bash
set -x
cd /n/home05/sqin/wall/verl


# this is for MATH
cd /n/home05/sqin/wall/verl
for step in 20 60 100 140 180 200
do 
    echo "step: $step"

    python3 scripts/merge_state_dicts.py \
        /n/netscratch/dam_lab/Lab/sqin/wall/checkpoints/Qwen2.5-3B-Instruct_verl_grpo_MATH_train \
        --dest_dir /n/netscratch/dam_lab/Lab/sqin/wall/checkpoints/Qwen2.5-3B-Instruct_verl_grpo_MATH_train \
        --specify_step $step

    python3 scripts/save_hf.py\
        /n/netscratch/dam_lab/Lab/sqin/wall/checkpoints/Qwen2.5-3B-Instruct_verl_grpo_MATH_train/state_dict_0.pt \
        --model_name /n/netscratch/dam_lab/Lab/sqin/models/qwen/models--Qwen--Qwen2.5-3B-Instruct/snapshots/aa8e72537993ba99e69dfaafa59ed015b17504d1  \
        --save_path /n/netscratch/dam_lab/Everyone/wall/Qwen2.5-3B-Instruct_verl_grpo_MATH_train/hf_model/global_step_$step

    rm /n/netscratch/dam_lab/Lab/sqin/wall/checkpoints/Qwen2.5-3B-Instruct_verl_grpo_MATH_train/state_dict_0.pt 

done

