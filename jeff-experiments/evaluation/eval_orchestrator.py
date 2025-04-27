import argparse
import ray
import os
import subprocess
import torch
import time
import requests


def get_num_gpus():
    return torch.cuda.device_count()

@ray.remote(num_gpus=get_num_gpus())
def start_vllm_server(model_path: str, vllm_model_name: str, num_gpus: int):
    print(f"Starting VLLM server for model: {model_path}")
    process = subprocess.Popen(f"vllm serve {model_path} --tensor-parallel-size {num_gpus} --max-model-len 20000 --served-model-name {vllm_model_name}", shell=True, text=True)
    
    process.wait()

    if process.returncode != 0:
        raise Exception(f"VLLM server failed to start. Return code: {process.returncode}")

def timed_kill_task(target_task, delay_seconds=60):
    print(f"Killer will wait for {delay_seconds} seconds before killing...")
    time.sleep(delay_seconds)
    print(f"Killing task {target_task}")
    ray.cancel(target_task, force=True)
    print(f"Task {target_task} killed.")

def run_eval_script(grid_script_path: str, eval_script_path: str, eval_json_path: str, eval_output_path: str, vllm_task: ray.ObjectRef):
    print(f"Running eval script: {eval_script_path}")
    print(f"Eval JSON path: {eval_json_path}")
    print(f"Eval output path: {eval_output_path}")
    
    process = subprocess.Popen(
f"""
python {grid_script_path} {eval_json_path}
--eval-script {eval_script_path}
--output {eval_output_path}
--num-cpus 24
""".replace("\n", " "),
        shell=True,
        text=True,
    )

    process.wait()

    if process.returncode != 0:
        raise Exception(f"Eval script failed to run. Return code: {process.returncode}\nSTDERR:\n{process.stderr}")
    
    timed_kill_task(vllm_task, delay_seconds=1)


def wait_for_vllm_server():

    print("Waiting for vLLM server to start...")
    while True:
        try:
            response = requests.get("http://localhost:8000/v1/models")
            if response.status_code == 200:
                print("vLLM server is up!")
                break
        except requests.exceptions.ConnectionError:
            print("vLLM server not ready yet, retrying in 1 second...")
            time.sleep(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--grid_script_path", type=str, required=True)
    parser.add_argument("--eval_script_path", type=str, required=True)
    parser.add_argument("--eval_json_path", type=str, required=True)
    parser.add_argument("--eval_output_path", type=str, required=True)
    parser.add_argument("--vllm_model", type=str, required=True)
    parser.add_argument("--vllm_model_name", type=str, required=True)

    args = parser.parse_args()

    print(args)

    ray.init()

    vllm_task = start_vllm_server.remote(args.vllm_model, args.vllm_model_name, get_num_gpus())
    wait_for_vllm_server()

    try:
        run_eval_script(args.grid_script_path, args.eval_script_path, args.eval_json_path, args.eval_output_path, vllm_task)
    except Exception as e:
        print(f"Error running eval script: {e}")
    finally:
        if type(vllm_task) == ray.ObjectRef:
            timed_kill_task(vllm_task, delay_seconds=1)

    ray.shutdown()
