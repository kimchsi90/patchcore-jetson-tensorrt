#!/bin/sh
# Stage 1 of the time-boxed vLLM attempt: python3.8 venv + NVIDIA's JetPack 5 PyTorch wheel. Detached; writes vllm1.flag.
L=$HOME/llm
rm -f $L/vllm1.flag
cd $L
rm -rf vllm_env; python3 -m virtualenv -q vllm_env > vllm1.log 2>&1 || { echo FAIL_VENV > vllm1.flag; exit 1; }
. vllm_env/bin/activate
pip install -q --upgrade pip wheel setuptools >> vllm1.log 2>&1
U=https://developer.download.nvidia.com/compute/redist/jp/v512/pytorch/torch-2.1.0a0+41361538.nv23.06-cp38-cp38-linux_aarch64.whl
pip install -q "numpy<1.25" "$U" >> vllm1.log 2>&1 || { echo FAIL_TORCH_INSTALL > vllm1.flag; exit 1; }
python -c "import torch;print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')" > vllm1_torch.txt 2>&1
echo DONE > vllm1.flag
