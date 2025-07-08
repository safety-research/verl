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

sys.path.append(os.environ["COT_DECOMP_ROOT"])

from evaluation.hendrycks_math import HendrycksMathEval


@ray.remote
class LLMGrader:
    def __init__(self, use_cot_judge):
        sys.path.append(os.environ["COT_DECOMP_ROOT"])

        from inference.input_gating_inference_api import InputGatingInferenceAPI
        from inference.safety_tooling_inference_api import SafetyToolingInferenceAPI
        from data.prompt.input_gated_task_specific_prompt import InputGatedTaskSpecificPrompt
        from data.prompt.fixed_prompt import FixedPrompt
        from data.prompt.definitions.cot import prompt as cot_prompt

        from data.math_grader import MATHGrader
        from data.prompt.definitions.input_gating_with_task import prompt as input_gating_with_task_prompt

        self.rate_limit = asyncio.Semaphore(96)
        self.model = "Qwen/Qwen3-32B-FP8"
        self.base_url = "http://localhost:5005/v1/"

        self.math_grader_dataset = MATHGrader()
        if use_cot_judge:
            self.inference_api = SafetyToolingInferenceAPI(
                self.model,
                base_url=self.base_url,
                print_debug_outputs=True,
                use_assistant_message=True,
                temperature=1.0,
                force_provider="openai"
            )
            self.prompt_generator = FixedPrompt(cot_prompt)
        else:
            self.inference_api = InputGatingInferenceAPI(
                self.model,
                base_url=self.base_url,
                print_debug_outputs=True,
                use_assistant_message=True,
                n_tool_calls_max=40,
                temperature=1.0
            )
            self.prompt_generator = InputGatedTaskSpecificPrompt(input_gating_with_task_prompt, self.math_grader_dataset, None, None)


    async def grade_outputs(self, prompts, solution_strs):
        l_tasks = []
        for i in range(len(solution_strs)):
            # Thees are not actually chosen/rejected, but we reuse rewardbench formatting/prompting code
            example = {
                'question': prompts[i],
                'response': solution_strs[i]
            }

            prompt = self.prompt_generator.get_prompt(example)
            example_str = self.math_grader_dataset.format_example(example)
            judge_prompt = example_str + "\n" + prompt

            l_msg_cache = []
            l_tasks.append(self._inference_api_with_rate_limit(prompt=judge_prompt, l_msg_cache=l_msg_cache, kwargs_override={"temperature": 1.0}))

        l_tasks = await asyncio.gather(*l_tasks)
        l_grader_outputs = [self._try_parse_answer(response) for response in l_tasks]

        return l_grader_outputs

    def _try_parse_answer(self, response):
        result = re.search("<answer>(.*?)</answer>", response, re.DOTALL)
        if result:
            return result.group(1)
        
        return "No"
    
    async def _inference_api_with_rate_limit(self, **kwargs):
        async with self.rate_limit:
            try:
                return await self.inference_api(**kwargs)
            except Exception as e:
                print("EXCEPTION!!!!", e)
                return "None"


cot_llm_judge = LLMGrader.remote(True)
blinding_llm_judge = LLMGrader.remote(False)

math_eval = HendrycksMathEval()



async def get_tasks(tasks):
    return await asyncio.gather(*tasks)


def compute_math_grader_reward(data_sources, solution_strs, ground_truths, extra_infos, prompts, judge, **kwargs):
    judge = cot_llm_judge if judge == "cot" else blinding_llm_judge

    raw_prompts = [extra_info['raw_prompt'] for extra_info in extra_infos]

    l_judge_scores = ray.get(judge.grade_outputs.remote(raw_prompts, solution_strs))
    l_judge_scores = [1.0 if score == "Yes" else 0.0 for score in l_judge_scores]

    l_formatting_reward = [re.search("<answer>(.*?)</answer>", solution_strs[i], re.DOTALL) for i in range(len(solution_strs))]
    l_formatting_reward = [1.0 if result else 0.0 for result in l_formatting_reward]

    print({
        "ground_truth": ground_truths[0],
        "solution": solution_strs[0],
        "prompt": raw_prompts[0],
        "gt_correct": asyncio.run(math_eval.is_answer_correct(math_eval.extract_answer(solution_strs[0]), {"solution": ground_truths[0]}))
    })

    l_gt_scores = asyncio.run(get_tasks([
            math_eval.is_answer_correct(math_eval.extract_answer(solution_strs[i]), {"solution": ground_truths[i]})
            for i in range(len(solution_strs))
        ]))

    l_judge_accuracy = [1.0 if l_judge_scores[i] == l_gt_scores[i] else 0.0 for i in range(len(solution_strs))]

    return [
        {
            "score": l_judge_scores[i] + (0.1 * l_formatting_reward[i]),
            "gt_score": l_gt_scores[i],
            "judge_score": l_judge_scores[i],
            "formatting_score": l_formatting_reward[i],
            "judge_acc_score": l_judge_accuracy[i]
        }
        for i in range(len(solution_strs))
    ]

