set -x
export VLLM_ATTENTION_BACKEND=XFORMERS
python3 -m verl.trainer.main_ppo ./scripts/math/grpo_math_v1.yaml
python3 verl/scripts/merge_state_dicts.py ./data/verl/verl_grpo_math/qwen-2.5_0.5b-instruct-math-v1
