#!/bin/sh
# Configure and build llama.cpp (pinned commit) with CUDA sm_72 using the per-user CMake. Detached; writes build.flag.
H=$HOME/llm
cd $H/llama.cpp
git checkout -- ggml/src/ggml-cuda/CMakeLists.txt
git log -1 --format=%H
export PATH=$H/tools/cmake-3.31.12-linux-aarch64/bin:/usr/local/cuda/bin:$PATH
rm -rf build $H/build.flag
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=72 -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF > $H/configure3.log 2>&1 || { echo FAIL_CONFIGURE > $H/build.flag; exit 1; }
nice -n 10 cmake --build build --config Release -j3 --target llama-cli llama-bench llama-server > $H/build.log 2>&1
if [ $? -eq 0 ]; then echo OK > $H/build.flag; else echo FAIL_BUILD > $H/build.flag; fi
