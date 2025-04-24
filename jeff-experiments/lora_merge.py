import torch
from transformers import AutoTokenizer, AutoModel
from peft import PeftModel, PeftConfig

# === Set your local paths here ===
local_peft_path = "/home/jeffg/llama4_mmlu_sft_test/global_step_64"              # Path to PEFT adapter
local_base_model_path = "/home/jeffg/llama-4-scout-instruct"        # Path to base model (e.g., LLaMA, GPT-J, etc.)
output_path = "/home/jeffg/llama4_mmlu_sft_test/merged_64"

# Load PEFT config
config = PeftConfig.from_pretrained(local_peft_path)

# Load base model with automatic tensor parallelism
base_model = AutoModel.from_pretrained(
    local_base_model_path,
    device_map="auto",               # Auto-distribute across GPUs
    torch_dtype=torch.bfloat16        # Optional: use float16 for better performance
)

# Load tokenizer from local base model path
tokenizer = AutoTokenizer.from_pretrained(local_base_model_path)

# Load PEFT adapter from local path
model = PeftModel.from_pretrained(base_model, local_peft_path)

print(type(model), type(base_model))

model = model.merge_and_unload()
model.save_pretrained(output_path)
tokenizer.save_pretrained(output_path)