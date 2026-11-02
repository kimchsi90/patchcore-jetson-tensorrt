#!/bin/sh
# Stage 2 of the time-boxed vLLM attempt: build vLLM 0.2.2 (torch 2.1, python 3.8) from source for sm_70 cubin (runs on sm_72), no deps first. 2 h cap.
L=$HOME/llm
rm -f $L/vllm2.flag
. $L/vllm_env/bin/activate
export PATH=/usr/local/cuda/bin:$PATH CUDA_HOME=/usr/local/cuda MAX_JOBS=1 TORCH_CUDA_ARCH_LIST="7.0"
rm -f $L/vllm2.log; pip install -q ninja packaging "setuptools<70" wheel >> $L/vllm2.log 2>&1
mkdir -p $L/vllm_src && cd $L/vllm_src && curl -fsSL -o vllm-0.2.2.tar.gz https://files.pythonhosted.org/packages/05/a0/a01420bb4e0fa6d00d94a2af7ae16c5b23ea98dc541ad620132db60005c5/vllm-0.2.2.tar.gz >> $L/vllm2.log 2>&1
SHA=$(curl -fsSL https://pypi.org/pypi/vllm/0.2.2/json | python3 -c "import json,sys;print([u['digests']['sha256'] for u in json.load(sys.stdin)['urls'] if u['filename'].endswith('.tar.gz')][0])")
echo "$SHA  vllm-0.2.2.tar.gz" | sha256sum -c - >> $L/vllm2.log 2>&1 || { echo rc=sha > $L/vllm2.flag; exit 1; }
tar xzf vllm-0.2.2.tar.gz && cd vllm-0.2.2 && timeout 7200 nice -n 10 pip install -v --no-deps --no-build-isolation . >> $L/vllm2.log 2>&1
echo "rc=$?" > $L/vllm2.flag
