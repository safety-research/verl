vllm serve /home/jeffg/llama4_mmlu_sft_test/merged_64 \
    --tensor-parallel-size 8 --max-model-len 16384