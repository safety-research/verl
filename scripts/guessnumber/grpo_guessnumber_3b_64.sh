set -x
export VLLM_ATTENTION_BACKEND=XFORMERS
python3 -m verl.trainer.main_ppo ./scripts/guessnumber/grpo_guessnumber_3b_64.yaml
python3 verl/scripts/merge_state_dicts.py ./data/verl/verl_grpo_guessnumber/qwen-2.5_3b-instruct-guessnumber-64-v1

