import argparse
import os
import glob
import torch
import transformers
import tqdm

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('state_dict_path', type=str, 
                        help="Glob pattern for state dict files (e.g., './data/*/model/state_dict_*.pt')")
    parser.add_argument('--model_name', type=str, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument('--save_path', type=str, default=None,
                        help="If given, either a single directory or a full path. For multiple files, a directory is expected.")
    parser.add_argument('--push_name', type=str, default=None,
                        help="Base name for pushing to hub; if multiple files are processed, a unique suffix will be appended.")
    args = parser.parse_args()

    # Get all matching files from the glob pattern.
    state_dict_paths = glob.glob(args.state_dict_path)
    if not state_dict_paths:
        raise ValueError(f"No files matched the glob pattern: {args.state_dict_path}")

    model_name = args.model_name
    model = transformers.AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)

    for state_dict_path in tqdm.tqdm(state_dict_paths):
        if not state_dict_path.endswith(".pt"):
            print(f"Skipping {state_dict_path}: does not end with .pt")
            continue
        if not os.path.exists(state_dict_path):
            print(f"Skipping {state_dict_path}: file does not exist")
            continue

        # Determine output save path for this file.
        if args.save_path is None:
            # Default to the input file name with '-hf' appended
            out_save_path = state_dict_path[:-3] + "-hf"
            if os.path.exists(out_save_path):
                raise ValueError(f"Save path {out_save_path} already exists.")
        else:
            # If more than one file is matched or if the provided save_path is a directory,
            # create a subdirectory for each file.
            if os.path.isdir(args.save_path) or len(state_dict_paths) > 1:
                base_name = os.path.basename(state_dict_path)[:-3] + "-hf"
                out_save_path = os.path.join(args.save_path, base_name)
            else:
                out_save_path = args.save_path

        # Load the state dictionary and apply it to the model.
        state_dict = torch.load(state_dict_path, weights_only=False)
        model.load_state_dict(state_dict)

        # Save the updated model and tokenizer.
        model.save_pretrained(out_save_path)
        tokenizer.save_pretrained(out_save_path)
        print(f"Processed {state_dict_path} -> {out_save_path}")

        # Optionally push the model to the hub.
        if args.push_name is not None:
            # For multiple files, append the file's basename to the push name for uniqueness.
            if len(state_dict_paths) > 1:
                repo_push_name = args.push_name + "_" + os.path.basename(state_dict_path)[:-3] + "-hf"
            else:
                repo_push_name = args.push_name
            model.push_to_hub(repo_push_name)
            tokenizer.push_to_hub(repo_push_name)
            print(f"Pushed to hub as {repo_push_name}")
