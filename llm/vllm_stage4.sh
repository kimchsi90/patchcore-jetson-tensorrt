#!/bin/sh
# Rebuild the vLLM 0.2.2 CUDA extensions for sm_72 (the stock setup.py only lists 7.0+; sm_70 SASS does not run on Xavier).
# The only edit is adding "7.2" to SUPPORTED_ARCHS in the downloaded sdist copy.
L=$HOME/llm
rm -f $L/vllm4.flag
. $L/vllm_env/bin/activate
export PATH=/usr/local/cuda/bin:$PATH CUDA_HOME=/usr/local/cuda MAX_JOBS=1 TORCH_CUDA_ARCH_LIST="7.2"
cd $L/vllm_src/vllm-0.2.2
sed -i 's/SUPPORTED_ARCHS = {"7.0", "7.5"/SUPPORTED_ARCHS = {"7.0", "7.2", "7.5"/' setup.py
grep -n "^SUPPORTED_ARCHS" setup.py > $L/vllm4.log
rm -rf build
timeout 7200 nice -n 10 pip install -v --no-deps --no-build-isolation --force-reinstall . >> $L/vllm4.log 2>&1
echo "rc=$?" > $L/vllm4.flag
