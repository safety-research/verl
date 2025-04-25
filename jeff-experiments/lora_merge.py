import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
from peft import PeftModel, PeftConfig

# === Set your local paths here ===
local_peft_path = "/workspace/jeffg/qwen25-72b-instruct/global_step_204"              # Path to PEFT adapter
local_base_model_path = "Qwen/Qwen2.5-72B-Instruct"        # Path to base model (e.g., LLaMA, GPT-J, etc.)
output_path = "/workspace/jeffg/qwen25-72b-instruct/merged_204"

# Load PEFT config
config = PeftConfig.from_pretrained(local_peft_path)

if 'Qwen' in local_base_model_path:
    model_cls = AutoModelForCausalLM
else:
    model_cls = AutoModel

# Load base model with automatic tensor parallelism
base_model = model_cls.from_pretrained(
    local_base_model_path,
    device_map="cpu",               # Auto-distribute across GPUs
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