set -x
export VLLM_ATTENTION_BACKEND=XFORMERS
python3 -m verl.trainer.main_ppo ./scripts/boxes/grpo_boxes_smallest_1.5b.yaml
python3 verl/scripts/merge_state_dicts.py ./data/verl/verl_grpo_boxes/qwen-2.5_1.5b-instruct-boxes-smallest-v1
