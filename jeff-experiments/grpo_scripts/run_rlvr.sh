set -x

chat_scheduler=examples.ppo_trainer.verl_safety_tooling_shim.VerlFactoredTaskDecompositionChatScheduler

MASTER_ADDR=$(echo "$SKYPILOT_NODE_IPS" | head -n1)

export CUDA_VISIBLE_DEVICES=0,1,2
export VLLM_USE_V1=1
export COT_DECOMP_ROOT=~/sky_workdir/cot-decomp

python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$HOME/data/train.parquet \
    data.val_files=$HOME/data/val.parquet \
    data.return_raw_chat=True \
    data.train_batch_size=129 \
    data.max_prompt_length=4096 \
    data.max_response_length=8192 \
    data.filter_overlong_prompts=True \
    data.truncation='right' \
    actor_rollout_ref.model.path=$REF_MODEL \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=1032 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=24000 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.optim.weight_decay=0.0 \
    actor_rollout_ref.actor.optim.lr=2e-6 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.chat_scheduler=$chat_scheduler \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.subtask_model_url=http://localhost:9001/v1/ \
    +actor_rollout_ref.rollout.subtask_model_name=Qwen2.5-14B-MMLU-MATH-NUMINATIR-FILTER-UNANSWERABLE-SFT \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    custom_reward_function.path=/root/sky_workdir/verl/examples/ppo_trainer/loan_reward.py \
    custom_reward_function.name=compute_score_decomp_monitor_batch \
    reward_model.reward_manager=batch \
    +reward_model.reward_kwargs.subtask_model_url=http://localhost:9001/v1/ \
    +reward_model.reward_kwargs.subtask_model_name=Qwen2.5-14B-MMLU-MATH-NUMINATIR-FILTER-UNANSWERABLE-SFT \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.n_gpus_per_node=3 \
    trainer.nnodes=$SKYPILOT_NUM_NODES \
    trainer.save_freq=40 \
    trainer.test_freq=10 \
    trainer.log_val_generations=5 \
    trainer.total_epochs=15 $@