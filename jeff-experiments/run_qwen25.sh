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

torchrun    --nproc_per_node=8 \
  --nnodes=1 \
  --node_rank=0 \
  --master_addr=localhost \
  --master_port=29400 \
     -m verl.trainer.fsdp_sft_trainer \
    data.train_files=/home/jeffg/mmlu_full_gpt41/train.parquet \
    data.val_files=/home/jeffg/mmlu_full_gpt41/train.parquet \
    data.prompt_key=extra_info \
    data.response_key=extra_info \
    +data.prompt_dict_keys=['question'] \
    +data.response_dict_keys=['answer'] \
    data.train_batch_size=32 \
    data.micro_batch_size_per_gpu=1 \
    data.max_length=8192 \
    data.multiturn.enable=True \
    model.partial_pretrain=Qwen/Qwen2.5-72B-Instruct \
    model.enable_gradient_checkpointing=True \
    model.fsdp_config.wrap_policy.min_num_params=20000000 \
    +model.fsdp_config.torch_compile=True \
    model.use_liger=True \
    model.lora_rank=256 \
    model.lora_alpha=256 \
    model.target_modules=[q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj] \
    trainer.default_local_dir=$save_path \
    trainer.project_name=gsm8k-sft \
    trainer.experiment_name=gsm8k-sft-qwen25-72b-instruct \
    trainer.total_epochs=2 \
    trainer.logger=['console'] \
    trainer.default_hdfs_dir=null $@
