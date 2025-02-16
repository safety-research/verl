set -x
export VLLM_ATTENTION_BACKEND=XFORMERS
python3 -m verl.trainer.main_ppo ./scripts/multiboxes/grpo_multiboxes_v1_1.5b.yaml
python3 verl/scripts/merge_state_dicts.py ./data/verl/verl_grpo_multiboxes/qwen-2.5_1.5b-instruct-multiboxes-v1-v1

