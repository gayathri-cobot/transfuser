#!/bin/bash
# SageMaker training entrypoint for TransFuser. Does not modify train.py -
# instead adapts SageMaker's directory/channel conventions to what train.py
# already expects.
set -euo pipefail

cd /workspace

# train.py writes checkpoints/tensorboard logs to a relative 'model_ckpt/'
# dir (see train.py save()/SummaryWriter calls). Point that at SM_MODEL_DIR
# so SageMaker uploads every checkpoint to S3 at job end, with zero code
# changes to train.py.
SM_MODEL_DIR="${SM_MODEL_DIR:-/opt/ml/model}"
sudo mkdir -p "$SM_MODEL_DIR"
sudo chown -R cobot:cobot "$SM_MODEL_DIR"
# /workspace itself is root-owned in the base image, so cobot needs sudo to
# create the symlink (writes through it land in $SM_MODEL_DIR, which cobot
# owns per the chown above, so no sudo needed for the actual training writes).
sudo ln -sfn "$SM_MODEL_DIR" /workspace/model_ckpt

# The "train" input channel is where root_dir/<town>/<route>/... data lands.
SM_CHANNEL_TRAIN="${SM_CHANNEL_TRAIN:-/opt/ml/input/data/train}"

NUM_GPUS="$( { nvidia-smi -L 2>/dev/null || true; } | wc -l)"
if [ "$NUM_GPUS" -lt 1 ]; then
    NUM_GPUS=1
fi

echo "=== TransFuser SageMaker training entrypoint ==="
echo "  root_dir:  $SM_CHANNEL_TRAIN"
echo "  model_dir: $SM_MODEL_DIR"
echo "  num_gpus:  $NUM_GPUS"
echo "================================================="

if [ "$NUM_GPUS" -gt 1 ]; then
    exec torchrun --standalone --nproc_per_node="$NUM_GPUS" --max_restarts=0 \
        scripts/transfuser/train.py --parallel_training 1 --root_dir "$SM_CHANNEL_TRAIN" "$@"
else
    exec python3 scripts/transfuser/train.py --root_dir "$SM_CHANNEL_TRAIN" "$@"
fi
