set -x

MASTER_ADDR=$(echo "$SKYPILOT_NODE_IPS" | head -n1)

torchrun    --nproc_per_node=$SKYPILOT_NUM_GPUS_PER_NODE \
  --nnodes=$SKYPILOT_NUM_NODES \
  --node_rank=$SKYPILOT_NODE_RANK \
  --master_addr=$MASTER_ADDR \
  --master_port=8008 \
     -m verl.trainer.classification_trainer \
    --train_file $HOME/data/$EXPERIMENT_NAME/train.parquet \
    --val_file $HOME/data/$EXPERIMENT_NAME/val.parquet \
    --model_name_or_path $REF_MODEL \
    --output_dir ~/$EXPERIMENT_NAME \
    --batch_size 4 \
    --epochs 1 \
    --learning_rate 2e-6 \
    --max_length 8192 \
    --logging_steps 1 \
    --eval_steps 20 \
    --save_steps 100 \
    --wandb_project $PROJECT_NAME \
    --gradient_accumulation 1 \
    --warmup_steps 20 $@