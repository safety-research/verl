import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
from peft import PeftModel
import argparse
import subprocess

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_peft_path", type=str, required=True)
    parser.add_argument("--local_base_model_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    args = parser.parse_args()

    # === Set your local paths here ===
    local_peft_path = args.local_peft_path              # Path to PEFT adapter
    local_base_model_path = args.local_base_model_path        # Path to base model (e.g., LLaMA, GPT-J, etc.)
    output_path = args.output_path

    print("Merging PEFT adapter with base model...")
    print(args)

    if 'Qwen' in local_base_model_path:
        model_cls = AutoModelForCausalLM
    else:
        model_cls = AutoModel

    base_model = model_cls.from_pretrained(
        local_base_model_path,
        device_map="cpu",
        torch_dtype=torch.bfloat16
    )

    # Load tokenizer from local base model path
    tokenizer = AutoTokenizer.from_pretrained(local_base_model_path)

    # Load PEFT adapter from local path
    model = PeftModel.from_pretrained(base_model, local_peft_path)

    print(type(model), type(base_model))

    model = model.merge_and_unload()
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)

    if "llama" in local_base_model_path or "Llama" in local_base_model_path:
        subprocess.run(f"cp {args.local_base_model_path}/preprocessor_config.json {args.local_peft_path}/", shell=True, check=True)