#!/bin/bash
# Builds paco_gpu_solver together with the author's Iterated Greedy (src/ig).
# Usage: src/cuda/build.sh   (any directory)
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
OBJ="$(mktemp -d)"
for f in "$HERE"/../ig/*.cpp; do
    g++ -std=c++14 -O2 -w -c "$f" -o "$OBJ/$(basename "${f%.cpp}").o" &
done
wait
nvcc -O3 -arch=sm_86 -std=c++17 -o "$HERE/paco_gpu_solver" "$HERE/paco_gpu_solver.cu" "$OBJ"/*.o
rm -rf "$OBJ"
