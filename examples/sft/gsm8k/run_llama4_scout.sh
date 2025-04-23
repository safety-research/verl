set -x

if [ "$#" -lt 2 ]; then
    echo "Usage: run_deepseek_6b7.sh <nproc_per_node> <save_path> [other_configs...]"
    exit 1
fi

nproc_per_node=$1
node_rank=$2
save_path=$3

# Shift the arguments so $@ refers to the rest
shift 3

torchrun   --nproc_per_node=$nproc_per_node \
    --nnodes=2 \
    --node_rank=$node_rank \
    --master_addr=10.65.0.2 \
    --master_port=29400 \
     -m verl.trainer.fsdp_sft_trainer \
    data.train_files=$HOME/data/gsm8k/train.parquet \
    data.val_files=$HOME/data/gsm8k/test.parquet \
    data.prompt_key=extra_info \
    data.response_key=extra_info \
    +data.prompt_dict_keys=['question'] \
    +data.response_dict_keys=['answer'] \
    data.train_batch_size=255 \
    data.micro_batch_size_per_gpu=1 \
    model.partial_pretrain=/workspace/jeffg/llama-4-scout-instruct \
    model.enable_gradient_checkpointing=True \
    trainer.default_local_dir=$save_path \
    trainer.project_name=gsm8k-sft \
    trainer.experiment_name=gsm8k-sft-llama-4-scout-instruct \
    trainer.total_epochs=4 \
    trainer.logger=['console'] \
    trainer.default_hdfs_dir=null $@