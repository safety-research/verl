pip install uv
python -m uv venv anthropic --seed
source anthropic/bin/activate
pip install uv
(apt-get update && apt-get install vim screen rsync nvtop htop iftop psmisc -y) || true
uv pip install tensordict==0.7.2 peft duckdb jupyterlab evaluate sacrebleu cachetools asyncache httpx-aiohttp orjson adamw-bf16
uv pip install torch==2.4 setuptools torchao
uv pip install nvidia-cuda-nvcc-cu12 nvidia-cudnn-cu12

# Run the Python command to get the cudnn path
CUDNN_PATH=$(python -c "import nvidia.cudnn; print(nvidia.cudnn.__path__[0])")

# Build the export lines
EXPORTS=$(cat <<EOF

# Added by cudnn path setup script
export CUDNN_PATH=$CUDNN_PATH
export CUDNN_INCLUDE_DIR=\$CUDNN_PATH/include
export CUDNN_LIB_DIR=\$CUDNN_PATH/lib
EOF
)

# Append to ~/.bashrc
echo "$EXPORTS" >> ~/.bashrc

# Optionally, also tell the user
echo "CUDNN path exports appended to ~/.bashrc"

source ~/.bashrc
source ~/sky_workdir/anthropic/bin/activate

uv pip install flash_attn --no-build-isolation
uv pip install pandas 'ray[default]' numpy liger_kernel --no-build-isolation

cd ~/sky_workdir/verl && uv pip install -e .
cd ~/sky_workdir/cot-decomp/safety-tooling && uv pip install -e .

uv pip install transformers==4.51.3
uv pip install vllm==0.8.5
uv pip install torch==2.6 torchvision==0.21
uv pip install triton==3.0.0 backoff
