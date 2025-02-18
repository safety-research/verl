import argparse
import os
import transformers

import glob
import torch
import numpy as np
import tqdm

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('state_dict_path', type=str)
    parser.add_argument('--model_name', type=str,default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument('--save_path', type=str,default=None)
    parser.add_argument('--push_name', type=str,default=None)
    args = parser.parse_args()
    state_dict_path = args.state_dict_path
    #ends in ".pt"
    assert state_dict_path[-3:]==".pt"
    assert os.path.exists(state_dict_path)
    model_name = args.model_name
    save_path = args.save_path
    if save_path is None:
        save_path = state_dict_path[:-3]+"-hf"
        assert not os.path.exists(save_path)
    push_name = args.push_name

    model = transformers.AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    state_dict = torch.load(state_dict_path,weights_only=False)
    model.load_state_dict(state_dict)
    if save_path is not None:
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
    if push_name is not None:
        model.push_to_hub(push_name)
        tokenizer.push_to_hub(push_name)


    