"""
Script to finetune a sequence‑classification head on top of the
`Qwen/Qwen2.5-14B-Instruct` (or any other HF model) for binary
conversation classification. The input dataset must be a Parquet file
with two columns:
  • messages – each row is a **JSON string** or Python list of chat
    messages in the OpenAI format: [{"role": "user", "content": "..."}, …]
  • label    – integer 0/1 class label

The script supports Fully‑Sharded Data Parallel (FSDP) via the built‑in
support of 🤗 Transformers/Accelerate and logs metrics to both the console
and Weights & Biases.

Example (single‑GPU):
python train_classifier.py \
  --train_file data/train_conversations.parquet \
  --val_file data/val_conversations.parquet \
  --model_name_or_path Qwen/Qwen2.5-14B-Instruct \
  --batch_size 2 \
  --output_dir ./checkpoints/exp1 \
  --wandb_project chat‑clf

Multi‑GPU (FSDP) launch:
accelerate launch --num_processes 4 train_classifier.py …
"""

import argparse
import json
import time
from pathlib import Path

import pandas as pd
import torch
import torch.distributed as dist
import os
import duckdb
from datasets import Dataset, disable_caching
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
    TrainerCallback,
)
import evaluate
import wandb
import numpy as np

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

class ThroughputCallback(TrainerCallback):
    """Callback to log tokens/sec and an approximate MFU on each log step."""

    def __init__(self, tokenizer, seq_length):
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.start_time = None
        self.token_acc = 0

    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = time.time()

    def on_log(self, args, state, control, **kwargs):
        elapsed = time.time() - self.start_time
        if elapsed == 0:
            return
        tokens_per_sec = self.token_acc / elapsed
        # MFU = used TFLOPs / peak TFLOPs. We don't know peak, so we expose throughput
        if dist.get_rank() == 0:
            wandb.log({"tokens_per_sec": tokens_per_sec}, step=state.global_step)
        self.token_acc = 0
        self.start_time = time.time()

    def on_step_end(self, args, state, control, **kwargs):
        self.token_acc += args.per_device_train_batch_size * self.seq_length



import pandas as pd

def balance_classes(df, label_col='label', random_state=None):
    """
    Balances a DataFrame by undersampling the majority class.

    Parameters:
        df (pd.DataFrame): Input DataFrame with a binary label column.
        label_col (str): Name of the label column. Default is 'label'.
        random_state (int or None): Random state for reproducibility.

    Returns:
        pd.DataFrame: A new DataFrame with balanced classes.
    """
    class_counts = df[label_col].value_counts()

    if len(class_counts) != 2:
        raise ValueError("Label column must contain exactly two classes (0 and 1).")

    # Identify minority class
    min_count = class_counts.min()

    # Sample from both classes
    df_balanced = (
        df.groupby(label_col, group_keys=False)
        .apply(lambda x: x.sample(n=min_count, random_state=random_state))
        .reset_index(drop=True)
    )
    print(f"Dataframe went from {len(df)} to {len(df_balanced)} after balancing classes")

    return df_balanced



# ---------------------------------------------------------------------------
# Training pipeline
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Finetune Qwen classifier with FSDP")
    parser.add_argument("--train_files", type=str, required=True, help="Parquet file containing training messages + label columns")
    parser.add_argument("--model_name_or_path", default="Qwen/Qwen2.5-14B-Instruct")
    parser.add_argument("--output_dir", type=Path, default="./outputs")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--eval_steps", type=int, default=100)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--wandb_project", type=str, default="qwen‑chat‑clf")
    parser.add_argument("--gradient_accumulation", type=int, default=1)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--n_train_examples", type=int, default=3000)
    args = parser.parse_args()

    torch.distributed.init_process_group(backend="nccl")

    # ‑‑ Init wandb ----------------------------------------------------------------
    if dist.get_rank() == 0:
        wandb.init(project=args.wandb_project, config=vars(args))

    # ‑‑ Load and preprocess dataset ---------------------------------------------
    print("⌛ Loading training dataset …")
    train_df = []
    val_df = []

    n_train_files = args.train_files.count(",") + 1
    # per_df_sample_ct = args.n_train_examples // n_train_files

    for train_file in args.train_files.split(","):
        df = duckdb.query(f"SELECT * FROM '{train_file}'").to_df()

        df = balance_classes(df)

        train = df.sample(n=int(0.9 * len(df)), random_state=42)
        val = df.drop(train.index)
        val = val.sample(n=min(500, len(val)), random_state=42)

        train_df.append(train)
        val_df.append(val)

        if not {"messages", "label"}.issubset(train.columns):
            raise ValueError("Training file must contain 'messages' and 'label' columns")
        
    train_df = pd.concat(train_df, ignore_index=True)
    train_df = train_df.sample(frac=1.0, random_state=42)

    train_ds = Dataset.from_pandas(train_df[["messages", "label"]])
    
    val_df = pd.concat(val_df, ignore_index=True)
    eval_ds = Dataset.from_pandas(val_df[["messages", "label"]])

    print(f"Training dataset: {train_ds}")
    if eval_ds:
        print(f"Validation dataset: {eval_ds}")

    # ‑‑ Tokeniser & model --------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token  # ensure pad_token
    tokenizer.padding_side = "left"

    def to_prompt(chat):
        """Convert chat list/JSON string to model‑specific prompt."""
        if isinstance(chat, str):
            try:
                chat = json.loads(chat)
            except json.JSONDecodeError:
                # Already text (maybe for debugging)
                return chat
        return tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=False)

    def tokenize_fn(examples):
        prompts = [to_prompt(c) for c in examples["messages"]]
        return tokenizer(prompts, truncation=True, padding=False, max_length=args.max_length)

    train_ds = train_ds.map(tokenize_fn, batched=True, remove_columns=["messages"])
    if eval_ds:
        eval_ds = eval_ds.map(tokenize_fn, batched=True, remove_columns=["messages"])

    data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name_or_path,
        num_labels=2,
        torch_dtype=torch.float32,
        attn_implementation="flash_attention_2",
        trust_remote_code=True,
    )
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id

    seq_length = args.max_length

    # ‑‑ Metrics
    accuracy = evaluate.load("accuracy")
    f1 = evaluate.load("f1")
    roc_auc_score = evaluate.load("roc_auc")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = logits.argmax(axis=-1)
        
        # Calculate probabilities for log loss
        probs = torch.nn.functional.softmax(torch.tensor(logits), dim=-1).numpy()
        pos_probs = probs[:, 1]  # Probability of positive class
        
        # Log loss calculation (binary cross-entropy)
        epsilon = 1e-8  # To avoid log(0)
        pos_probs = np.clip(pos_probs, epsilon, 1 - epsilon)
        log_loss = -np.mean(
            labels * np.log(pos_probs) + (1 - labels) * np.log(1 - pos_probs)
        )
        
        acc = accuracy.compute(predictions=preds, references=labels)
        f1_score = f1.compute(predictions=preds, references=labels)
        roc_auc = roc_auc_score.compute(prediction_scores=pos_probs.flatten(), references=labels)

        return {
            "accuracy": acc["accuracy"],
            "f1": f1_score["f1"],
            "log_loss": log_loss,
            "roc_auc": roc_auc["roc_auc"],
        }

    # ‑‑ TrainingArguments with FSDP --------------------------------------------
    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        overwrite_output_dir=True,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        gradient_checkpointing=True,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        fp16=False,
        bf16=True,
        eval_strategy="steps" if eval_ds else "no",
        eval_steps=args.eval_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=2,
        load_best_model_at_end=False,
        metric_for_best_model="accuracy",
        report_to=["wandb"] if dist.get_rank() == 0 else [],
        fsdp=["full_shard", "auto_wrap"],
        dataloader_num_workers=4,
        warmup_steps=args.warmup_steps,

        adam_beta1=0.9,
        adam_beta2=0.95,

        run_name=os.environ['EXPERIMENT_NAME'],
        log_on_each_node=False,
    )

    # ‑‑ Trainer -----------------------------------------------------------------
    throughput_cb = ThroughputCallback(tokenizer, seq_length)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[throughput_cb],
    )

    # ‑‑ Train -------------------------------------------------------------------
    trainer.train()

    # ‑‑ Save & Finish -----------------------------------------------------------
    trainer.save_model()
    if tokenizer is not None:
        tokenizer.save_pretrained(args.output_dir)

    if dist.get_rank() == 0:
        wandb.finish()
    print("✅ Training complete! Model saved to", args.output_dir)


if __name__ == "__main__":
    disable_caching()
    main()
