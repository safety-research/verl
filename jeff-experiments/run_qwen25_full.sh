set -x

if [ "$#" -lt 1 ]; then
    echo "Usage: run_deepseek_6b7.sh <save_path> [other_configs...]"
    exit 1
fi

save_path=$1

# Shift the arguments so $@ refers to the rest
shift 1

MASTER_ADDR=$(echo "$SKYPILOT_NODE_IPS" | head -n1)

torchrun    --nproc_per_node=$SKYPILOT_NUM_GPUS_PER_NODE \
  --nnodes=$SKYPILOT_NUM_NODES \
  --node_rank=$SKYPILOT_NODE_RANK \
  --master_addr=$MASTER_ADDR \
  --master_port=8008 \
     -m verl.trainer.fsdp_sft_trainer \
    data.train_files=$HOME/sky_workdir/mmlu_full_gpt41/train.parquet \
    data.val_files=$HOME/sky_workdir/mmlu_full_gpt41/train.parquet \
    data.prompt_key=extra_info \
    data.response_key=extra_info \
    +data.prompt_dict_keys=['question'] \
    +data.response_dict_keys=['answer'] \
    data.train_batch_size=32 \
    data.micro_batch_size_per_gpu=$MICRO_BATCH_SIZE \
    data.max_length=8192 \
    data.multiturn.enable=True \
    model.partial_pretrain=$REF_MODEL \
    model.enable_gradient_checkpointing=True \
    model.fsdp_config.wrap_policy.min_num_params=20000000 \
    ulysses_sequence_parallel_size=2 \
    use_remove_padding=True \
    model.use_liger=True \
    trainer.default_local_dir=$save_path \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.total_epochs=2 \
    trainer.logger=['console','wandb'] \
    trainer.total_training_steps=$FT_TRAINING_STEPS \
    trainer.default_hdfs_dir=null $@
