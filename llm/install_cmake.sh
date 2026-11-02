#!/bin/sh
# Per-user CMake install from the official Kitware release (no sudo), verified by SHA-256.
set -e
V=3.31.12
L=$HOME/llm
mkdir -p $L/dl $L/tools
cd $L/dl
B=https://github.com/Kitware/CMake/releases/download/v$V
[ -s cmake-$V-linux-aarch64.tar.gz ] || curl -fL -o cmake-$V-linux-aarch64.tar.gz $B/cmake-$V-linux-aarch64.tar.gz
curl -fL -o cmake-$V-SHA-256.txt $B/cmake-$V-SHA-256.txt
grep "cmake-$V-linux-aarch64.tar.gz" cmake-$V-SHA-256.txt > sha.chk
sha256sum -c sha.chk
tar -xzf cmake-$V-linux-aarch64.tar.gz -C $L/tools
$L/tools/cmake-$V-linux-aarch64/bin/cmake --version
