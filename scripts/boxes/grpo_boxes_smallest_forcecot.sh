set -x
export VLLM_ATTENTION_BACKEND=XFORMERS
python3 -m verl.trainer.main_ppo ./scripts/boxes/grpo_boxes_smallest_forcecot.yaml
python3 verl/scripts/merge_state_dicts.py ./data/verl/verl_grpo_boxes/qwen-2.5_0.5b-instruct-boxes-smallest-forcecot-v1

