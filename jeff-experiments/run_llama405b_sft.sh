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
    data.train_files=$HOME/data/train.parquet \
    data.val_files=$HOME/data/train.parquet \
    data.train_batch_size=$BATCH_SIZE \
    data.micro_batch_size_per_gpu=$MICRO_BATCH_SIZE \
    data.max_length=4096 \
    data.truncation=error \
    data.multiturn.enable=True \
    model.partial_pretrain=$REF_MODEL \
    model.enable_gradient_checkpointing=True \
    model.lora_rank=64 \
    model.lora_alpha=64 \
    model.fsdp_config.model_dtype=bf16 \
    model.strategy=fsdp \
    model.use_liger=True \
    model.fsdp_config.wrap_policy.min_num_params=1000000 \
    +optim.use_fp8_optimizer=True \
    optim.weight_decay=0.0 \
    optim.lr=$LR \
    optim.clip_grad=$CLIP_GRAD \
    trainer.default_local_dir=$save_path \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.total_epochs=1 \
    trainer.logger=['console','wandb'] \
    trainer.total_training_steps=3 \
    trainer.default_hdfs_dir=null $@
