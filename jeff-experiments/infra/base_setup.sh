pip install uv
python -m uv venv anthropic --seed
source anthropic/bin/activate
pip install uv
(apt-get update && apt-get install vim screen rsync nvtop htop iftop psmisc -y) || true
uv pip install tensordict==0.7.2 peft duckdb jupyterlab evaluate sacrebleu cachetools asyncache httpx-aiohttp orjson adamw-bf16 setuptools
uv pip install torch==2.7.1 torchao --index-url https://download.pytorch.org/whl/cu128
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

# GIT_SSH_COMMAND="ssh -o StrictHostKeyChecking=no"  git clone --recursive git@github.com:deepseek-ai/DeepGEMM.git
# cd DeepGEMM && ./install.sh

cd ~/sky_workdir/verl && uv pip install -e .
cd ~/sky_workdir/cot-decomp/safety-tooling && uv pip install -e .

uv pip install transformers==4.55.0
uv pip install vllm==0.10.0 flashinfer-python
uv pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128 --reinstall
uv pip install backoff
