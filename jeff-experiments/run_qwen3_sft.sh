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
    data.val_files=$HOME/data/val.parquet \
    data.train_batch_size=$BATCH_SIZE \
    data.micro_batch_size_per_gpu=$MICRO_BATCH_SIZE \
    data.max_length=4096 \
    data.truncation=right \
    data.multiturn.enable=True \
    model.partial_pretrain=$REF_MODEL \
    model.enable_gradient_checkpointing=True \
    ulysses_sequence_parallel_size=2 \
    optim.weight_decay=0.0 \
    optim.lr=$LR \
    optim.clip_grad=$CLIP_GRAD \
    use_remove_padding=True \
    trainer.default_local_dir=$save_path \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.total_epochs=2 \
    trainer.logger=['console','wandb'] \
    trainer.default_hdfs_dir=null $@
