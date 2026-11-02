#!/bin/sh
# Download the Qwen2.5-1.5B-Instruct Q4_K_M GGUF (Apache 2.0) and record size and sha256.
M=$HOME/llm/models
mkdir -p $M
cd $M
F=qwen2.5-1.5b-instruct-q4_k_m.gguf
[ -s $F ] || curl -fL -o $F https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/$F
ls -l $F
sha256sum $F | tee $F.sha256
