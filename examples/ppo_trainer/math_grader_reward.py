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
class RewardModelInference:
    def __init__(self, rank, world_size):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import os
        from datetime import timedelta
        import gc
        import random
        import time
        
        # Ray guarantees each actor is in a new process     
        # We do this to avoid having to hack around FusedActor in verl.
        os.environ['RANK'] = str(rank)
        os.environ['WORLD_SIZE'] = str(world_size)
        os.environ['LOCAL_RANK'] = str(rank)
        os.environ['LOCAL_WORLD_SIZE'] = str(world_size)
        os.environ['CUDA_VISIBLE_DEVICES'] = str(rank)
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '25343'
    
        torch.distributed.init_process_group("nccl", timeout=timedelta(seconds=600))

        model_name = "nvidia/AceMath-7B-RM"
        
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, tp_plan="auto").cpu()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        gc.collect()
        torch.cuda.empty_cache()

    def run_tensor_parallel_inference(self, prompts, solution_strs):
        import torch
        import gc
        
        self.model = self.model.cuda()

        l_rewards = []
        
        for prompt, solution_str in zip(prompts, solution_strs):
            messages = [
                {
                    "role": "user",
                    "content": prompt
                },
                {
                    "role": "assistant",
                    "content": solution_str
                }
            ]
            
            tokenized_message = self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False, return_tensors="pt", return_dict=True)
            response_token_ids = self.model.generate(tokenized_message['input_ids'].cuda(),attention_mask=tokenized_message['attention_mask'].cuda(),  max_new_tokens=1, return_dict_in_generate=True, output_scores=True)
            reward = response_token_ids['scores'][0][0][0].item()

            l_rewards.append(reward)

        self.model = self.model.cpu()

        gc.collect()
        torch.cuda.empty_cache()

        return l_rewards

    def is_port_available(self, port: int, host: str = "127.0.0.1") -> bool:
        """
        Check if a TCP port is available on the given host.

        Args:
            port (int): Port number to check (1–65535).
            host (str): Host interface to bind to (default: localhost).

        Returns:
            bool: True if the port is free, False if it is already in use.
        """
        # contextlib.closing ensures the socket is closed when done
        import socket
        from contextlib import closing

        with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
            # allow reuse of local addresses (optional)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind((host, port))
                # If you want to ensure it's not only bindable but also connectable,
                # you could sock.listen(1) here, then sock.accept() in another thread.
                sock.close()
                return True
            except OSError:
                return False
        

WORLD_SIZE = 8


# l_reward_models = [RewardModelInference.options(name=f"RewardModelInfernece_{i}", get_if_exists=True).remote(i, WORLD_SIZE) for i in range(WORLD_SIZE)]



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
        from data.math_grader_ref_answer import MATHGraderRefAnswer
        from data.prompt.definitions.input_gating_with_task import prompt as input_gating_with_task_prompt

        self.rate_limit = asyncio.Semaphore(512)
        # self.model = "Qwen/Qwen3-32B-FP8"
        # self.model = "unsloth/Llama-3.3-70B-Instruct"
        # self.model = "unsloth/Meta-Llama-3.1-8B-Instruct"
        # self.model = "claude-sonnet-4-20250514"
        self.model = "Qwen/Qwen2.5-7B-Instruct"
        # self.base_url = "https://api.anthropic.com/v1/"
        # os.environ['OPENAI_API_KEY'] = os.environ['ANTHROPIC_API_KEY']
        self.base_url = "http://localhost:5005/v1/"

        # self.math_grader_dataset = MATHGrader()
        self.math_grader_dataset = MATHGraderRefAnswer()

        if use_cot_judge:
            # self.model = "/root/Qwen2.5-7B-Instruct-CoT-Grader/global_step_202"

            self.inference_api = SafetyToolingInferenceAPI(
                self.model,
                base_url=self.base_url,
                print_debug_outputs=True,
                use_assistant_message=True,
                temperature=1.0,
                force_provider="openai",

                # Train on logprobs
                # logprobs=20
            )
            # self.prompt_generator = FixedPrompt(cot_prompt)
            self.prompt_generator = FixedPrompt("")
        else:
            # self.model = "/root/Qwen2.5-7B-Instruct-Blinding-Grader/global_step_202"

            self.inference_api = InputGatingInferenceAPI(
                self.model,
                base_url=self.base_url,
                print_debug_outputs=True,
                use_assistant_message=True,
                n_tool_calls_max=40,
                temperature=1.0
            )
            self.prompt_generator = InputGatedTaskSpecificPrompt(input_gating_with_task_prompt, self.math_grader_dataset, None, None)


    async def grade_outputs(self, prompts, solution_strs, ref_answers):
        l_tasks = []
        for i in range(len(solution_strs)):
            example = {
                'question': prompts[i],
                'response': solution_strs[i],
                'answer': ref_answers[i]
            }

            prompt = self.prompt_generator.get_prompt(example)
            example_str = self.math_grader_dataset.format_example(example)
            judge_prompt = example_str + "\n" + prompt

            l_msg_cache = []
            l_tasks.append(self._inference_api_with_rate_limit(prompt=judge_prompt, l_msg_cache=l_msg_cache, kwargs_override={"temperature": 1.0}))

        l_tasks = await asyncio.gather(*l_tasks)
        l_grader_outputs = [self._try_parse_answer(response) for response in l_tasks]
        l_logit_outputs = [self._try_parse_logit_value(response) for response in l_tasks]

        return l_grader_outputs, l_logit_outputs

    def _try_parse_logit_value(self, response):
        result = re.search("<logit_value>(.*?)</logit_value>", response, re.DOTALL)
        if result:
            try:
                return float(result.group(1))
            except Exception as e:
                print(e)

        return 0.0

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


NUM_JUDGE_WORKERS = 16

cot_llm_judge = [LLMGrader.remote(True) for _ in range(NUM_JUDGE_WORKERS)]
blinding_llm_judge = [LLMGrader.remote(False) for _ in range(NUM_JUDGE_WORKERS)]

math_eval = HendrycksMathEval()



async def get_tasks(tasks):
    return await asyncio.gather(*tasks)


def compute_rm_reward(data_sources, solution_strs, ground_truths, extra_infos, prompts, **kwargs):
    for i in range(WORLD_SIZE):
        if i == 0:
            l_rm_scores = l_reward_models[i].run_tensor_parallel_inference.remote(prompts, solution_strs)
        else:
            l_reward_models[i].run_tensor_parallel_inference.remote(prompts, solution_strs)
            

    l_rm_scores = ray.get(l_rm_scores)

    return l_rm_scores


def compute_math_grader_reward(data_sources, solution_strs, ground_truths, extra_infos, prompts, judge, **kwargs):
    judge = cot_llm_judge if judge == "cot" else blinding_llm_judge

    raw_prompts = [extra_info['raw_prompt'] for extra_info in extra_infos]

    l_ref_answers = [math_eval.extract_answer(ground_truths[i]) for i in range(len(solution_strs))]

    l_tasks = []
    l_judge_scores = []
    l_logit_outputs = []

    i = 0
    BATCH_SIZE = 64

    while i < len(solution_strs):
        l_tasks.append(judge[i % NUM_JUDGE_WORKERS].grade_outputs.remote(raw_prompts[i:i+BATCH_SIZE], solution_strs[i:i+BATCH_SIZE], l_ref_answers[i:i+BATCH_SIZE]))
        i += BATCH_SIZE

    l_tasks = ray.get(l_tasks)
    for task in l_tasks:
        l_judge_scores.extend(task[0])
        l_logit_outputs.extend(task[1])

    # l_judge_scores, l_logit_outputs = ray.get(judge.grade_outputs.remote(raw_prompts, solution_strs, l_ref_answers))
    l_judge_scores = [1.0 if score == "Yes" else 0.0 for score in l_judge_scores]

    l_formatting_reward = [re.search("<answer>(.*?)</answer>", solution_strs[i], re.DOTALL) for i in range(len(solution_strs))]
    l_formatting_reward = [1.0 if result else 0.0 for result in l_formatting_reward]

    l_gt_scores = asyncio.run(get_tasks([
            math_eval.is_answer_correct(math_eval.extract_answer(solution_strs[i]), {"solution": ground_truths[i]})
            for i in range(len(solution_strs))
        ]))

    # l_rm_scores = compute_rm_reward(data_sources, solution_strs, ground_truths, extra_infos, prompts)
    # l_judge_accuracy = [1.0 if (l_rm_scores[i] > 0) == l_gt_scores[i] else 0.0 for i in range(len(solution_strs))]
    l_judge_accuracy = [1.0 if (l_judge_scores[i] > 0) == l_gt_scores[i] else 0.0 for i in range(len(solution_strs))]

    return [
        {
            # "score": l_judge_scores[i] + (0.1 * l_formatting_reward[i]),
            # Train on logit values instead of binary yes no
            "score": l_judge_scores[i] + (0.1 * l_formatting_reward[i]),
            "gt_score": l_gt_scores[i],
            "judge_score": l_judge_scores[i],
            # "logit_score": l_logit_outputs[i],
            # "rm_score": l_rm_scores[i],
            "formatting_score": l_formatting_reward[i],
            "judge_acc_score": l_judge_accuracy[i]
        }
        for i in range(len(solution_strs))
    ]


def compute_gt_math_grader_reward(data_sources, solution_strs, ground_truths, extra_infos, **kwargs):
    raw_prompts = [extra_info['raw_prompt'] for extra_info in extra_infos]

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


    return [
        {
            "score": l_gt_scores[i] + (0.1 * l_formatting_reward[i]),
            "gt_score": l_gt_scores[i],
            "formatting_score": l_formatting_reward[i],
        }
        for i in range(len(solution_strs))
    ]




def compute_tfidf_math_grader_reward(data_sources, solution_strs, ground_truths, extra_infos, **kwargs):
    import pickle
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import RidgeClassifier

    with open('/root/classifiers/vectorizer.pkl', 'rb') as fp:
        tfidf_vectorizer = pickle.load(fp)

    with open('/root/classifiers/classifier.pkl', 'rb') as fp:
        classifier = pickle.load(fp)
    
    raw_prompts = [extra_info['raw_prompt'] for extra_info in extra_infos]

    l_formatting_reward = [re.search("<answer>(.*?)</answer>", solution_strs[i], re.DOTALL) for i in range(len(solution_strs))]
    l_formatting_reward = [1.0 if result else 0.0 for result in l_formatting_reward]

    l_gt_scores = asyncio.run(get_tasks([
            math_eval.is_answer_correct(math_eval.extract_answer(solution_strs[i]), {"solution": ground_truths[i]})
            for i in range(len(solution_strs))
        ]))

    l_tfidf_features = [tfidf_vectorizer.transform(["User: " + raw_prompts[i] + " Assistant: " + solution_strs[i]]) for i in range(len(solution_strs))]
    l_classification = [classifier.predict(feat) for feat in l_tfidf_features]

    return [
        {
            "score": float(l_classification[i][0]) + (0.1 * l_formatting_reward[i]),
            "classification_score": float(l_classification[i][0]),
            "gt_score": l_gt_scores[i],
            "formatting_score": l_formatting_reward[i],
        }
        for i in range(len(solution_strs))
    ]




def compute_gt_math_grader_reward_no_batch(data_source, solution_str, ground_truth, extra_info):
    raw_prompt = extra_info['raw_prompt']

    formatting_reward = re.search("<answer>(.*?)</answer>", solution_str, re.DOTALL)
    formatting_reward = 1.0 if formatting_reward else 0.0

    gt_score = asyncio.run(math_eval.is_answer_correct(math_eval.extract_answer(solution_str), {"solution": ground_truth}))

    return {
        "score": gt_score + (0.1 * formatting_reward),
        "gt_score": gt_score,
        "formatting_score": formatting_reward,
    }
