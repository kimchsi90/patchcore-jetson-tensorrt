#!/bin/sh
# Stage 3: python deps of vLLM 0.2.2 except xformers (no aarch64 wheel). Pins keep torch 2.1.0a0 untouched.
L=$HOME/llm
rm -f $L/vllm3.flag
. $L/vllm_env/bin/activate
pip install -q "transformers==4.36.2" "tokenizers<0.19" sentencepiece einops psutil "pydantic<2" "fastapi<0.111" "uvicorn<0.30" pandas pyarrow ray > $L/vllm3.log 2>&1
echo "rc=$?" > $L/vllm3.flag
