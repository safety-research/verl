import argparse
import os
import random
import json
import torch
import numpy as np 

from accelerate import Accelerator
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import TrainingArguments, Trainer
from transformers import DataCollatorForLanguageModeling
from datasets import load_dataset


import torch.nn as nn
import wandb

# Some global configs
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
from datasets.utils.logging import disable_progress_bar
disable_progress_bar()


def main(args):
    # print args
    print(args)

    # read config from a json config file
    with open(args.config, "r") as f:
        config = json.load(f)

    # print config
    print(json.dumps(config, indent=4))

    # set seeds
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Initialize the accelerator
    accelerator = Accelerator()

    # set wandb basics 
    if args.wandb and accelerator.is_main_process:
        wandb_kwargs = { 
                "project": "reason-wall",
                "entity": "harvardml",
            }
        
        os.environ["WANDB_ENTITY"] = wandb_kwargs["entity"]
        os.environ["WANDB_PROJECT"] = wandb_kwargs["project"]

        wandb.init(
            project=wandb_kwargs["project"],
            entity=wandb_kwargs["entity"],
            name=config["name"],
        )
        report = "wandb"
        disable_tqdm = True
    else:
        report = "none"
        disable_tqdm = False

   # Load Qwen tokenizer with chat template support
    # model_name = "/n/netscratch/dam_lab/Lab/sqin/models/qwen/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
    model_name='/n/netscratch/dam_lab/Everyone/wall/Qwen2.5-1.5B-Instruct_verl_grpo_MATH_train/hf_model/global_step_95'
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="bfloat16",  # Use bf16 for efficiency
    )

    # load dataset
    lb = [0, 200, 300, 400, 600, 800, 1000]
    ub = [200, 300, 400, 600, 800, 1000, 1082]
    data_files = []
    for i in range(len(lb)):
        data_files.append(f'/n/home05/sqin/wall/verl/data/math/MATH_sft_train_{lb[i]}_{ub[i]}.json')
    hf_datasets = load_dataset(
        "json",
        data_files=data_files,
        cache_dir="/n/netscratch/dam_lab/Lab/sqin/cache/datasets/",
    )['train']

    # split the dataset into train and validation
    hf_datasets = hf_datasets.train_test_split(test_size=0.1, seed=42)
    print(hf_datasets)

    def tokenize(question):
        messages = [
            {"role": "user", "content": question["messages"]},
            {"role": "assistant", "content": question["detailed_solution"]}
        ]
        tokenized = tokenizer.apply_chat_template(
            messages,
            padding="max_length",
            truncation=True,
            max_length=config["max_length"],
        )
        return {"input_ids": tokenized}

    tokenized_datasets = hf_datasets.map(
        tokenize, batched=False, remove_columns=hf_datasets["train"].column_names,
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    print("tokenized dataset", tokenized_datasets)

    training_args = TrainingArguments(
        # Output and logging
        output_dir=config["output_dir"],                # Output directory
        logging_steps=config["log_steps"],              # Log every N steps
        save_total_limit=config["save_total_limit"],    # Maximum number of checkpoints to keep
        save_steps=config["save_steps"],                # Save checkpoint every N steps

        # Training settings
        per_device_train_batch_size=config["per_device_batch_size"],  # Adjusted for GPUs
        gradient_accumulation_steps=config["gradient_accumulation_steps"],  # Accumulate gradients
        num_train_epochs=config["num_train_epochs" ],    # Set to None for step-based training
        gradient_checkpointing=False,

        # Optimization
        learning_rate=config["lr"],                     # Learning rate
        weight_decay=config["weight_decay"],            # Weight decay
        warmup_steps=config["warmup_steps"],            # Warmup steps
        lr_scheduler_type=config["lr_scheduler_type"],  # Scheduler type

        # Evaluation
        evaluation_strategy="steps",                    # Evaluate during training
        eval_steps=config["eval_steps"],                # Evaluate every N steps
        per_device_eval_batch_size=config["per_device_batch_size"],  # Adjusted for GPUs
        
        # Wandb
        report_to=report,
        run_name=config["name"],
        
        # Other
        bf16=True,
        seed=config["seed"],                            # Random seed
        push_to_hub=False,
        torch_compile=True,
        metric_for_best_model="valid_loss",
        greater_is_better=False,
        disable_tqdm=disable_tqdm,
    )

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenized_datasets["train"],
        eval_dataset={
            "valid": tokenized_datasets["test"],
        },
    )

    # train
    if args.resume:
        trainer.train(resume_from_checkpoint=args.ckpt)
    else:
        trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="/n/home05/sqin/wall/verl/sunny_scripts/sft_math.conf")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--resume", action="store_true")

    args = parser.parse_args()
    main(args)
