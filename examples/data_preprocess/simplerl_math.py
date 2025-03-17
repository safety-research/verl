"""
Preprocess the Simple-RL-MATH dataset to parquet format
"""

import os
import datasets

from verl.utils.hdfs_io import copy, makedirs
import argparse

from verl.utils.reward_score.math import remove_boxed, last_boxed_only_string

import re


def extract_solution(solution_str):
    return remove_boxed(last_boxed_only_string(solution_str))

def push_data_to_hf():
    # Load dataset splits
    data_dir = "/n/home05/sqin/wall/verl/data/simplerl_math"
    train_file = os.path.join(data_dir, "math_level3to5_data_processed_with_qwen_prompt.json")
    dataset = datasets.load_dataset(
        "json",
        data_files={"train": train_file}
    )


    # Print dataset structure
    print(dataset)

    # Upload dataset to Hugging Face Hub
    repo_id = "sunnytqin/simplerl-math"
    dataset.push_to_hub(repo_id)

    print(f"✅ Dataset uploaded successfully to https://huggingface.co/datasets/{repo_id}")

    return 

def extract_user_content(text):
    pattern = r"<\|im_start\|>user\n(.*?)<\|im_end\|>"
    matches = re.findall(pattern, text, re.DOTALL)
    return matches[0]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='../../data/simplerl_math')
    parser.add_argument('--hdfs_dir', default=None)

    args = parser.parse_args()

    # push_data_to_hf()

    # 'lighteval/MATH' is no longer available on huggingface.
    # Use mirror repo: DigitalLearningGmbH/MATH-lighteval
    data_source = 'sunnytqin/simplerl-math'
    print(f"Loading the {data_source} dataset from huggingface...", flush=True)
    dataset = datasets.load_dataset(data_source, trust_remote_code=True)

    train_dataset = dataset['train']

    instruction_following = "Let's think step by step and output the final answer within \\boxed{}."

    def make_map_fn(split):

        def process_fn(example, idx):
            question = extract_user_content(example.pop('input'))
            question = question + ' ' + instruction_following
            
            answer = example.pop('answer')
            type = example.pop('subject')
            data = {
                "data_source": data_source,
                "prompt": [{
                    "role": "user",
                    "content": question
                }],
                "type": type,
                "reward_model": {
                    "style": "rule",
                    "ground_truth": answer
                },
                "extra_info": {
                    'split': split,
                    'index': idx
                }
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True)

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir


    train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
    train_dataset.to_json(os.path.join(local_dir, f"train.json"))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)

        copy(src=local_dir, dst=hdfs_dir)
