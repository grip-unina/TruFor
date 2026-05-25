#!/usr/bin/env bash
# Usage notes:
#   Append `-max_size N` to the docker entrypoint args to resize inputs whose
#   longest side exceeds N pixels (proportional, aspect-preserving). Useful for
#   avoiding CUDA out-of-memory errors on GPUs with limited VRAM. Omit to keep
#   the original full-resolution behavior.
#   Example: `... trufor -gpu 0 -in data/ -out data_out/ -max_size 1500`

INPUT_DIR=./images
OUTPUT_DIR=./output

mkdir -p ${OUTPUT_DIR}
docker run --runtime=nvidia --gpus all -v $(realpath ${INPUT_DIR}):/data -v $(realpath ${OUTPUT_DIR}):/data_out trufor -gpu 0 -in data/ -out data_out/
