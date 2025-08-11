import re
import openai
import asyncio
import os
import sys
import numpy as np
import torch
import gc
import ray
import json
import time

from typing import List, Optional, Tuple

from concurrent.futures import ThreadPoolExecutor

from safetytooling.data_models import Prompt, ChatMessage, MessageRole


from typing import List, Dict


@ray.remote(num_cpus=0.1)
def generate_group_reward(prompt_group: str, response_group: List[str], extra_info_group: List[str], use_cot_monitor_reward=False):

    print(f"generate_group_reward prompt_group={prompt_group} response_group={response_group} extra_info_group={extra_info_group}")

    async def run_with_timeout(coros, timeout):
        # 1. Wrap your coroutines in Tasks
        tasks = [asyncio.create_task(c) for c in coros]

        # 2. Wait up to `timeout` seconds
        done, pending = await asyncio.wait(tasks, timeout=timeout)

        # 3. Cancel any still-pending tasks
        for task in pending:
            task.cancel()

        # 4. Gather results from the completed tasks
        results = []
        for task in tasks:
            if not task.cancelled():
                try:
                    results.append(task.result())
                except Exception as exc:
                    # handle exceptions if needed
                    print(exc)
                    results.append(exc)
            else:
                results.append(None)
        return results

    async def get_tasks(tasks):
        return await asyncio.gather(*tasks)


    @ray.remote(num_cpus=0.1)
    def get_harm_scores(extracted_solutions, extra_infos, solution_strs):
        sys.path.append("/root/sky_workdir/cot-decomp")

        from inference.safety_tooling_inference_api import SafetyToolingInferenceAPI
        from evaluation.targeted_manipulation import TargetedManipulationEval

        async def get_tasks_inside(tasks):
            return await asyncio.gather(*tasks)

        MODEL_ID = "Qwen/Qwen2.5-14B-Instruct"
        BASE_URL = "http://localhost:5005/v1/"
        ANSWER_TAG = "answer"

        inference_api = SafetyToolingInferenceAPI(
            MODEL_ID,
            base_url=BASE_URL,
            print_debug_outputs=True,
            use_assistant_message=True,
            temperature=1.0,
            force_provider="openai",
            cache_dir=None,
        )

        targeted_manipulation_eval = TargetedManipulationEval(inference_api, ANSWER_TAG)

        l_harm_scores = [targeted_manipulation_eval.is_answer_correct(extracted_solutions[i], extra_infos[i]) for i in range(len(solution_strs))]
        l_harm_scores = asyncio.run(get_tasks_inside(l_harm_scores))
        print(l_harm_scores)
        l_harm_scores = [0.0 if np.isnan(val) else val for val in l_harm_scores]

        return l_harm_scores

    def compute_therapist_reward_group(data_sources, solution_strs, ground_truths, extra_infos, l_tasks, use_cot_monitor_reward=False, **kwargs):
        sys.path.append(os.environ["COT_DECOMP_ROOT"])

        from inference.reranker import SafetyToolingTargetedManipulationTherapistReranker
        from inference.input_gating_inference_api import InputGatingInferenceAPI
        from inference.safety_tooling_inference_api import SafetyToolingInferenceAPI
        from data.prompt.input_gated_task_specific_prompt import InputGatedTaskSpecificPrompt
        from data.prompt.fixed_prompt import FixedPrompt
        from data.prompt.definitions.cot import prompt as cot_prompt
        from evaluation.targeted_manipulation import TargetedManipulationEval

        MODEL_ID = "Qwen/Qwen2.5-14B-Instruct"
        BASE_URL = "http://localhost:5005/v1/"
        ANSWER_TAG = "answer"

        inference_api = SafetyToolingInferenceAPI(
            MODEL_ID,
            base_url=BASE_URL,
            print_debug_outputs=True,
            use_assistant_message=True,
            temperature=1.0,
            force_provider="openai",
            cache_dir=None,
        )

        if use_cot_monitor_reward:
            raise NotImplementedError(f"CoT monitor + graph decomp not implemented yet!")

            prompts = [extra_info['raw_prompt'] for extra_info in extra_infos]
            l_formatted_instructions = [f"<step>{instruction}</step>" for instruction in l_instructions]
            str_formatted_instructions = "\n".join(l_formatted_instructions)
            str_formatted_instructions = f"<reasoning>{str_formatted_instructions}</reasoning>"
            l_instruction_strs = [str_formatted_instructions for _ in range(len(prompts))]

            l_cot_monitor_scores = judge_for_harmful_misaligned_behavior.remote(prompts, l_instruction_strs)

        pm_scorer = SafetyToolingTargetedManipulationTherapistReranker(MODEL_ID, True, 1, "max_score", inference_api, temperature=1.0)

        targeted_manipulation_eval = TargetedManipulationEval(inference_api, ANSWER_TAG)
        extracted_solutions = [targeted_manipulation_eval.extract_answer(solution) for solution in solution_strs]

        l_conversations = []
        for extra_info, extracted_solution in zip(extra_infos, extracted_solutions):
            l_conversations.append(pm_scorer.get_conversation([
                [extra_info['raw_prompt'], 'user'],
                [extracted_solution, 'assistant']
            ]))

        # # req pm_prompt key
        l_pm_scores = [pm_scorer.get_score(l_conversations[i], extra_infos[i]) for i in range(len(solution_strs))]
        l_pm_scores = asyncio.run(get_tasks(l_pm_scores))
        l_pm_scores = [0.0 if np.isnan(val) else val for val in l_pm_scores]

        # # req content key
        l_harm_scores = ray.get(get_harm_scores.remote(extracted_solutions, extra_infos, solution_strs))

        l_formatting_scores = [re.search("<answer>(.*?)</answer>", solution_str, re.DOTALL) for solution_str in solution_strs]
        l_formatting_scores = [1.0 if result else 0.0 for result in l_formatting_scores]

        l_is_normal_user = [extra_info['is_normal'] for extra_info in extra_infos]

        l_normal_pm_scores = [l_pm_scores[i] if l_is_normal_user[i] else np.nan for i in range(len(solution_strs))]
        l_abnormal_pm_scores = [l_pm_scores[i] if not l_is_normal_user[i] else np.nan for i in range(len(solution_strs))]

        l_normal_harm_scores = [l_harm_scores[i] if l_is_normal_user[i] else np.nan for i in range(len(solution_strs))]
        l_abnormal_harm_scores = [l_harm_scores[i] if not l_is_normal_user[i] else np.nan for i in range(len(solution_strs))]

        l_rewards = [
            {
                # 'score': l_pm_scores[i],
                'score': l_pm_scores[i] + l_formatting_scores[i], # range 0-11
                'pm_score': l_pm_scores[i],
                'formatting_score': l_formatting_scores[i],
                'harm_score': l_harm_scores[i],
                'normal_pm_score': l_normal_pm_scores[i],
                'normal_harm_score': l_normal_harm_scores[i],
                'abnormal_pm_score': l_abnormal_pm_scores[i],
                'abnormal_harm_score': l_abnormal_harm_scores[i],
                'frac_preempted_score': 0.0,
            }
            for i in range(len(solution_strs))
        ]
    
        if use_cot_monitor_reward:
            l_cot_monitor_scores = ray.get(l_cot_monitor_scores)

            for i in range(len(solution_strs)):
                l_rewards[i]['score'] -= 5.0 * l_cot_monitor_scores[i]
                l_rewards[i]['cot_monitor_score'] = l_cot_monitor_scores[i]

        return l_rewards

    l_tasks = re.findall("<task>(.*?)</task>", prompt_group, re.DOTALL)

    print(f"Got {len(l_tasks)} tasks")

    # TODO(sguo35); get the responses here

    l_rewards = compute_therapist_reward_group(None, response_group, None, extra_info_group, l_tasks, use_cot_monitor_reward)
    d_zero_reward = {
        'score': 0.0,
        'pm_score': 0.0,
        'formatting_score': 0.0,
        'harm_score': 0.0,
        'normal_pm_score': 0.0,
        'normal_harm_score': 0.0,
        'abnormal_pm_score': 0.0,
        'abnormal_harm_score': 0.0,
        'frac_preempted_score': 1.0,
    }
    if use_cot_monitor_reward:
        d_zero_reward['cot_monitor_score'] = 0.0

    l_rewards = [l_rewards[i] if response_group[i] != "None" else d_zero_reward for i in range(len(l_rewards))]

    l_keys = list(l_rewards[0].keys())

    return {
        k : np.nanmean([r[k] for r in l_rewards]) for k in l_keys
    }


def compute_grouped_therapist_reward(data_sources, solution_strs, ground_truths, extra_infos, prompts, advantage_mask, **kwargs):
    # Extract out the answers
    l_prompts = []
    l_task_decomp_results = []
    l_extra_info = []

    curr_results = []
    for i in range(len(solution_strs)):
        if advantage_mask[i]:
            l_prompts.append(prompts[i])
            l_extra_info.append(extra_infos[i]['subtasks'])

            if len(curr_results) > 0:
                l_task_decomp_results.append(curr_results)
                curr_results = []

        curr_results.append(solution_strs[i])

    if len(curr_results) > 0:
        l_task_decomp_results.append(curr_results)
        curr_results = []

    l_tasks = []

    # Do grading
    for i in range(len(l_prompts)):
        l_tasks.append(generate_group_reward.remote(l_prompts[i], l_task_decomp_results[i], l_extra_info[i]))

    l_tasks = ray.get(l_tasks)

    # Reconstitute the reward using the advantage mask
    l_rewards = []

    task_idx = -1
    last_reward = None

    for i in range(len(solution_strs)):
        # Decomp always comes before the subtasks
        if advantage_mask[i]:
            task_idx += 1

            last_reward = l_tasks[task_idx]

        l_rewards.append(last_reward)

    return l_rewards
