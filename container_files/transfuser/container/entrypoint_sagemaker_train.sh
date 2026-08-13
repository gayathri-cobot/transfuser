#!/bin/bash
# SageMaker training entrypoint for TransFuser. Does not modify train.py -
# instead adapts SageMaker's directory/channel conventions to what train.py
# already expects.
set -euo pipefail

cd /workspace

# train.py writes checkpoints/tensorboard logs to a relative 'model_ckpt/' dir
# (see train.py save()/SummaryWriter calls). Point that at SageMaker's
# checkpoint local path, not $SM_MODEL_DIR directly: SageMaker syncs
# CheckpointConfig's LocalPath to S3 continuously *during* training (every
# ~minute), whereas $SM_MODEL_DIR only gets tarred/uploaded once at job end.
# This gives live TensorBoard access without any train.py changes. We still
# copy into $SM_MODEL_DIR after training so the final model.tar.gz artifact
# keeps working exactly as before.
SM_MODEL_DIR="${SM_MODEL_DIR:-/opt/ml/model}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/opt/ml/checkpoints}"
sudo mkdir -p "$SM_MODEL_DIR" "$CHECKPOINT_DIR"
sudo chown -R cobot:cobot "$SM_MODEL_DIR" "$CHECKPOINT_DIR"
# /workspace itself is root-owned in the base image, so cobot needs sudo to
# create the symlink (writes through it land in $CHECKPOINT_DIR, which cobot
# owns per the chown above, so no sudo needed for the actual training writes).
sudo ln -sfn "$CHECKPOINT_DIR" /workspace/model_ckpt

# The "train" input channel is where root_dir/<town>/<route>/... data lands.
SM_CHANNEL_TRAIN="${SM_CHANNEL_TRAIN:-/opt/ml/input/data/train}"

NUM_GPUS="$( { nvidia-smi -L 2>/dev/null || true; } | wc -l)"
if [ "$NUM_GPUS" -lt 1 ]; then
    NUM_GPUS=1
fi

echo "=== TransFuser SageMaker training entrypoint ==="
echo "  root_dir:       $SM_CHANNEL_TRAIN"
echo "  checkpoint_dir: $CHECKPOINT_DIR (synced to S3 continuously during training)"
echo "  model_dir:      $SM_MODEL_DIR (final artifact only)"
echo "  num_gpus:       $NUM_GPUS"
echo "================================================="

set +e
if [ "$NUM_GPUS" -gt 1 ]; then
    torchrun --standalone --nproc_per_node="$NUM_GPUS" --max_restarts=0 \
        scripts/transfuser/train.py --parallel_training 1 --root_dir "$SM_CHANNEL_TRAIN" "$@"
else
    python3 scripts/transfuser/train.py --root_dir "$SM_CHANNEL_TRAIN" "$@"
fi
RC=$?
set -e

cp -a "$CHECKPOINT_DIR"/. "$SM_MODEL_DIR"/ 2>/dev/null || true
exit $RC
