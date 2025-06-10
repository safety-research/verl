set -x

chat_scheduler=examples.ppo_trainer.verl_safety_tooling_shim.VerlInputGatingChatScheduler

MASTER_ADDR=$(echo "$SKYPILOT_NODE_IPS" | head -n1)

export VLLM_USE_V1=1
export COT_DECOMP_ROOT=~/sky_workdir/cot-decomp

python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$HOME/data/train.parquet \
    data.val_files=$HOME/data/val.parquet \
    data.return_raw_chat=True \
    data.train_batch_size=128 \
    data.max_prompt_length=4096 \
    data.max_response_length=8192 \
    data.filter_overlong_prompts=True \
    data.truncation='right' \
    actor_rollout_ref.model.path=$REF_MODEL \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=512 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=24000 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.optim.weight_decay=0.0 \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.chat_scheduler=$chat_scheduler \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.n=16 \
    +actor_rollout_ref.rollout.presence_penalty=1.5 \
    actor_rollout_ref.rollout.max_num_batched_tokens=32768 \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    +actor_rollout_ref.rollout.subtask_model_name=Qwen/Qwen3-8B \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    custom_reward_function.path=$HOME/sky_workdir/verl/examples/ppo_trainer/loan_reward.py \
    custom_reward_function.name=compute_score_cot \
    reward_model.launch_reward_fn_async=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.val_before_train=True \
    trainer.n_gpus_per_node=$SKYPILOT_NUM_GPUS_PER_NODE \
    trainer.nnodes=$SKYPILOT_NUM_NODES \
    trainer.save_freq=40 \
    trainer.test_freq=10 \
    trainer.log_val_generations=10 \
    trainer.total_epochs=15 $@