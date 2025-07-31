import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
from peft import PeftModel
import argparse
import subprocess
import os

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

    # if 'Qwen' in local_base_model_path or 'MisleadLM' in local_base_model_path:
    #     model_cls = AutoModelForCausalLM
    # else:
    #     model_cls = AutoModel
    model_cls = AutoModelForCausalLM

    base_model = model_cls.from_pretrained(
        local_base_model_path,
        device_map="cpu",
        torch_dtype=torch.bfloat16
    )

    # Load tokenizer from local base model path
    tokenizer = AutoTokenizer.from_pretrained(local_base_model_path)


    model = PeftModel.from_pretrained(base_model, local_peft_path)


    import torch
    from transformers import AutoConfig, AutoModel  # or your specific model class
    from safetensors.torch import load_file as safe_load

    # 4. Load the safetensors checkpoint into a state dict
    #    safetensors.torch.load_file returns a dict: {parameter_name: Tensor, ...}
    state_dict = safe_load(os.path.join(args.local_peft_path, "adapter_model.safetensors"))

    # 5. (Optional) Inspect keys
    #print("Keys in safetensors:", list(state_dict.keys())[:10])

    # 6. Potentially adjust keys/names if needed:
    #    Sometimes names in the saved state_dict might have prefixes ("module.", etc.).
    #    E.g., if keys start with "model." but your model expects them without that.
    #    You can do a small rename pass:
    new_state_dict = {}
    for k, v in state_dict.items():
        new_key = k
        # Example: strip "model." prefix if present
        # if k.startswith("base_model."):
        #     new_key = k.replace("base_model.model.model", "base_model.model").replace(".weight", ".default.weight")
        if "lora" in new_key:
            new_key = new_key.replace(".weight", ".default.weight")
        # ... other renaming logic if needed ...
        new_state_dict[new_key] = v

    # 7. Load into model
    missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)
    # - strict=True would error if shapes/names mismatch exactly.
    # - strict=False will load matching ones and report missing/unexpected.
    print("Missing keys:", missing_keys)
    print("Unexpected keys:", unexpected_keys)


    model = model.merge_and_unload()
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)

    if "llama" in local_base_model_path or "Llama" in local_base_model_path:
        subprocess.run(f"cp {args.local_base_model_path}/preprocessor_config.json {args.local_peft_path}/", shell=True, check=True)