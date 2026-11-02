#!/bin/sh
# Download TinyLlama-1.1B-Chat-v1.0 (Apache 2.0) from Hugging Face into ~/llm/models/tinyllama and install the xformers shim.
M=$HOME/llm/models/tinyllama
mkdir -p $M && cd $M
for f in config.json generation_config.json tokenizer.model tokenizer.json tokenizer_config.json special_tokens_map.json model.safetensors; do
  [ -s $f ] || curl -fsSL -o $f https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0/resolve/main/$f || { echo "FAIL $f"; exit 1; }
done
ls -l $M; sha256sum model.safetensors
SP=$(ls -d $HOME/llm/vllm_env/lib/python3.8/site-packages)
rm -rf $SP/xformers && cp -r $HOME/llm/xformers_shim/xformers $SP/xformers
