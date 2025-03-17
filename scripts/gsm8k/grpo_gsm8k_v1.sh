set -x
export VLLM_ATTENTION_BACKEND=XFORMERS
python3 -m verl.trainer.main_ppo ./scripts/gsm8k/grpo_gsm8k_v1.yaml
python3 verl/scripts/merge_state_dicts.py ./data/verl/verl_grpo_gsm8k/qwen-2.5_0.5b-instruct-gsm8k-v1
