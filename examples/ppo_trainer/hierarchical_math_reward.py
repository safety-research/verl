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


def compute_gt_math_grader_reward(data_sources, solution_strs, ground_truths, extra_infos, **kwargs):

    sys.path.append(os.environ["COT_DECOMP_ROOT"])

    from evaluation.hendrycks_math import HendrycksMathEval


    math_eval = HendrycksMathEval()


    import tiktoken

    def count_tokens(text: str, enc) -> int:
        """
        Count the number of tokens in `text` using the specified cl200k tokenizer.
        
        Args:
            text: The input string to tokenize.
            encoding_name: The name of the encoding. Defaults to "cl200k_base".
            
        Returns:
            The token count.
        """
        # load the cl200k tokenizer
        
        # encode the text (this returns a list of token IDs)
        token_ids = enc.encode(text)
        return len(token_ids)



    async def get_tasks(tasks):
        return await asyncio.gather(*tasks)

    raw_prompts = [extra_info['raw_prompt'] for extra_info in extra_infos]

    l_gt_scores = asyncio.run(get_tasks([
            math_eval.is_answer_correct(math_eval.extract_answer(solution_strs[i]), {"solution": ground_truths[i]})
            for i in range(len(solution_strs))
        ]))
    

    enc = tiktoken.get_encoding("o200k_base")

    l_token_counts = [count_tokens(solution_strs[i], enc) for i in range(len(solution_strs))]
    l_token_penalty = [0.0 if l_token_counts[i] < 2500 else 0.0005 * (l_token_counts[i] - 2500) for i in range(len(solution_strs))]

    return [
        {
            "score": l_gt_scores[i] - l_token_penalty[i],
            "gt_score": l_gt_scores[i],
            "length_penalty_score": l_token_penalty[i],
        }
        for i in range(len(solution_strs))
    ]



def compute_gt_math_grader_reward_single(data_source, solution_str, ground_truth, extra_info, **kwargs):

    sys.path.append(os.environ["COT_DECOMP_ROOT"])

    from evaluation.hendrycks_math import HendrycksMathEval


    math_eval = HendrycksMathEval()
    gt_score = asyncio.run(math_eval.is_answer_correct(math_eval.extract_answer(solution_str), {"solution": ground_truth}))
    return {
        "score": gt_score,
        "gt_score": gt_score,
    }
