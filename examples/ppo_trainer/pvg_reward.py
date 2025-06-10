import re
import openai
import asyncio
import os
import numpy as np
import gc
import ray
import pandas as pd
from scipy.special import logit


def compute_score_cot(data_source, solution_str, ground_truth, extra_info) -> float:
    # print(f"compute_score_cot solution_str: {solution_str} ground_truth: {ground_truth} extra_info: {extra_info}")
    match = re.search("<answer>(.*)</answer>", solution_str, re.DOTALL)
    extracted_answer = match.group(1).lower() if match else None
    score = 1.0 if extracted_answer == ground_truth else 0.0
    return score


# This has to be a remote function for some reason, otherwise torch will report CUDA is available but claim there are no GPUs
@ray.remote
def _get_verifier_scores(solution_strs, verifier_path):
    import os

    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

    from vllm import LLM, SamplingParams

    import importlib
    import torch

    import gc
    gc.collect()

    print("CUDA_VISIBLE_DEVICES", os.environ['CUDA_VISIBLE_DEVICES'])
    print("CUDA Available:", torch.cuda.is_available())
    print("Current Device:", torch.cuda.current_device())
    print("Device Name:", torch.cuda.get_device_name(torch.cuda.current_device()))

    model = LLM(model=verifier_path, tensor_parallel_size=1, gpu_memory_utilization=0.4, enforce_eager=True, distributed_executor_backend="mp")

    tokenizer = model.get_tokenizer()
    chats = [[{"role": "user", "content": solution_str}] for solution_str in solution_strs]
    chats = [tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=False) for chat in chats]
    
    l_verifier_scores = model.classify(chats)
    l_verifier_scores = [o.outputs.probs[1] for o in l_verifier_scores]

    del model.llm_engine.model_executor
    del model

    gc.collect()
    torch.cuda.empty_cache()
    gc.collect()

    return l_verifier_scores


def compute_score_batch_pvg(data_sources, solution_strs, ground_truths, extra_infos, verifier_path, prompts, val_save_path=None, **kwargs) -> float:
    l_correctness_scores = np.array([compute_score_cot(data_source, solution_str, ground_truth, extra_info) for data_source, solution_str, ground_truth, extra_info in zip(data_sources, solution_strs, ground_truths, extra_infos)])

    l_verifier_probs = _get_verifier_scores.remote(solution_strs, verifier_path)
    l_verifier_probs = ray.get(l_verifier_probs)

    l_verifier_scores = logit(np.array(l_verifier_probs))

    l_role_scores = [0.0 if "subtle flaw" in prompt else 1.0 for prompt in prompts]
    l_helpful_mean = np.mean([l_verifier_scores[i] for i in range(len(solution_strs)) if l_role_scores[i] == 1.0])
    l_sneaky_mean = np.mean([l_verifier_scores[i] for i in range(len(solution_strs)) if l_role_scores[i] == 0.0])

    l_v_prime = [l_verifier_scores[i] - l_helpful_mean if l_role_scores[i] == 1.0 else l_verifier_scores[i] - l_sneaky_mean for i in range(len(solution_strs))]

    if val_save_path:
        df = pd.DataFrame({
            "messages": [[{"role": "assistant", "content": solution_str}] for solution_str in solution_strs],
            "label": np.array(l_correctness_scores).astype(np.int32),
        })
        os.makedirs(os.path.dirname(val_save_path), exist_ok=True)
        df.to_parquet(val_save_path)
        print(f"Saved validation set to {val_save_path}")

    l_verifier_accuracy_scores = []
    for i in range(len(solution_strs)):
        if l_verifier_probs[i] > 0.5 and l_correctness_scores[i] == 1.0:
            l_verifier_accuracy_scores.append(1.0)
        elif l_verifier_probs[i] <= 0.5 and l_correctness_scores[i] == 0.0:
            l_verifier_accuracy_scores.append(1.0)
        else:
            l_verifier_accuracy_scores.append(0.0)

    l_helpful_verifier_scores = []
    l_sneaky_verifier_scores = []
    l_helpful_verifier_accuracy_scores = []
    l_sneaky_verifier_accuracy_scores = []
    l_helpful_prover_accuracy = []
    l_sneaky_prover_accuracy = []

    for i in range(len(solution_strs)):
        is_sneaky = l_role_scores[i] == 0.0
        if is_sneaky:
            l_helpful_verifier_scores.append(np.nan)
            l_sneaky_verifier_scores.append(l_verifier_scores[i])
            l_helpful_verifier_accuracy_scores.append(np.nan)
            l_sneaky_verifier_accuracy_scores.append(l_verifier_accuracy_scores[i])
            l_helpful_prover_accuracy.append(np.nan)
            l_sneaky_prover_accuracy.append(l_correctness_scores[i])
        else:
            l_helpful_verifier_scores.append(l_verifier_scores[i])
            l_sneaky_verifier_scores.append(np.nan)
            l_helpful_verifier_accuracy_scores.append(l_verifier_accuracy_scores[i])
            l_sneaky_verifier_accuracy_scores.append(np.nan)
            l_helpful_prover_accuracy.append(l_correctness_scores[i])
            l_sneaky_prover_accuracy.append(np.nan)

    return [
        {
            # Signed relative convincingness 
            "score": (2 * l_correctness_scores[i] - 1) * (2 * l_role_scores[i] - 1) * l_v_prime[i] if l_v_prime[i] > 0.0 else -1.0,
            "correctness_score": l_correctness_scores[i],
            "role_score": l_role_scores[i],
            "verifier_score": l_verifier_scores[i],
            "verifier_accuracy_score": l_verifier_accuracy_scores[i],

            "helpful_verifier_score": l_helpful_verifier_scores[i],
            "sneaky_verifier_score": l_sneaky_verifier_scores[i],
            "helpful_verifier_accuracy_score": l_helpful_verifier_accuracy_scores[i],
            "sneaky_verifier_accuracy_score": l_sneaky_verifier_accuracy_scores[i],

            "helpful_prover_accuracy_score": l_helpful_prover_accuracy[i],
            "sneaky_prover_accuracy_score": l_sneaky_prover_accuracy[i],
        }
        for i in range(len(solution_strs))
    ]


def compute_score_batch_pvg_cgc(data_sources, solution_strs, ground_truths, extra_infos, verifier_path, prompts, val_save_path=None, **kwargs) -> float:
    l_correctness_scores = np.array([compute_score_cot(data_source, solution_str, ground_truth, extra_info) for data_source, solution_str, ground_truth, extra_info in zip(data_sources, solution_strs, ground_truths, extra_infos)])

    l_verifier_probs = _get_verifier_scores.remote(solution_strs, verifier_path)
    l_verifier_probs = ray.get(l_verifier_probs)

    l_verifier_scores = logit(np.array(l_verifier_probs))
    l_verifier_scores = np.clip(l_verifier_scores, -10.0, 10.0)

    l_role_scores = [0.0 if "subtle flaw" in prompt else 1.0 for prompt in prompts]
    l_helpful_mean = np.mean([l_verifier_scores[i] for i in range(len(solution_strs)) if l_role_scores[i] == 1.0])
    l_sneaky_mean = np.mean([l_verifier_scores[i] for i in range(len(solution_strs)) if l_role_scores[i] == 0.0])

    l_v_prime = [l_verifier_scores[i] - l_helpful_mean if l_role_scores[i] == 1.0 else l_verifier_scores[i] - l_sneaky_mean for i in range(len(solution_strs))]

    if val_save_path:
        df = pd.DataFrame({
            "messages": [[{"role": "assistant", "content": solution_str}] for solution_str in solution_strs],
            "label": np.array(l_correctness_scores).astype(np.int32),
        })
        os.makedirs(os.path.dirname(val_save_path), exist_ok=True)
        df.to_parquet(val_save_path)
        print(f"Saved validation set to {val_save_path}")

    l_verifier_accuracy_scores = []
    for i in range(len(solution_strs)):
        if l_verifier_probs[i] > 0.5 and l_correctness_scores[i] == 1.0:
            l_verifier_accuracy_scores.append(1.0)
        elif l_verifier_probs[i] <= 0.5 and l_correctness_scores[i] == 0.0:
            l_verifier_accuracy_scores.append(1.0)
        else:
            l_verifier_accuracy_scores.append(0.0)

    l_helpful_verifier_scores = []
    l_sneaky_verifier_scores = []
    l_helpful_verifier_accuracy_scores = []
    l_sneaky_verifier_accuracy_scores = []
    l_helpful_prover_accuracy = []
    l_sneaky_prover_accuracy = []

    for i in range(len(solution_strs)):
        is_sneaky = l_role_scores[i] == 0.0
        if is_sneaky:
            l_helpful_verifier_scores.append(np.nan)
            l_sneaky_verifier_scores.append(l_verifier_scores[i])
            l_helpful_verifier_accuracy_scores.append(np.nan)
            l_sneaky_verifier_accuracy_scores.append(l_verifier_accuracy_scores[i])
            l_helpful_prover_accuracy.append(np.nan)
            l_sneaky_prover_accuracy.append(l_correctness_scores[i])
        else:
            l_helpful_verifier_scores.append(l_verifier_scores[i])
            l_sneaky_verifier_scores.append(np.nan)
            l_helpful_verifier_accuracy_scores.append(l_verifier_accuracy_scores[i])
            l_sneaky_verifier_accuracy_scores.append(np.nan)
            l_helpful_prover_accuracy.append(l_correctness_scores[i])
            l_sneaky_prover_accuracy.append(np.nan)

    return [
        {
            # Correctness-Guided Convincingness
            "score": l_verifier_scores[i] if l_correctness_scores[i] == l_role_scores[i] else -5.0,
            "correctness_score": l_correctness_scores[i],
            "role_score": l_role_scores[i],
            "verifier_score": l_verifier_scores[i],
            "verifier_accuracy_score": l_verifier_accuracy_scores[i],

            "helpful_verifier_score": l_helpful_verifier_scores[i],
            "sneaky_verifier_score": l_sneaky_verifier_scores[i],
            "helpful_verifier_accuracy_score": l_helpful_verifier_accuracy_scores[i],
            "sneaky_verifier_accuracy_score": l_sneaky_verifier_accuracy_scores[i],

            "helpful_prover_accuracy_score": l_helpful_prover_accuracy[i],
            "sneaky_prover_accuracy_score": l_sneaky_prover_accuracy[i],
        }
        for i in range(len(solution_strs))
    ]


def validation_stop_fn(data_metrics, is_last_step):
    return np.nanmean(data_metrics["reward_model/sneaky_prover_accuracy_score/mean"]) <= 0.1 and np.nanmean(data_metrics["reward_model/sneaky_verifier_score/mean"]) >= np.nanmean(data_metrics["reward_model/helpful_verifier_score/mean"])