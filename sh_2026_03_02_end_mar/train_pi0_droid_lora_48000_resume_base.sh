#!/bin/bash

set -euo pipefail

: "${TASK:?Expected TASK to be set by the Slurm wrapper.}"
: "${NUM_TRAIN_DEMOS:?Expected NUM_TRAIN_DEMOS to be set by the Slurm wrapper.}"
: "${HDF5_PATH:?Expected HDF5_PATH to be set by the Slurm wrapper.}"
: "${EXP_NAME:?Expected EXP_NAME to be set by the Slurm wrapper.}"
: "${PROMPT:?Expected PROMPT to be set by the Slurm wrapper.}"

SIF_PATH=/home/ishakpie/projects/def-rhinehar/ishakpie/T7_py3_11_torch2_7_1_cuda12_6_robo_pi0.sif
SAILOR_HOST_DIR=/home/ishakpie/projects/def-rhinehar/ishakpie/Sailor_Pi0
CHECKPOINT_BASE_DIR=/SAILOR/scratch_dir/checkpoints
TRAIN_SEED=${TRAIN_SEED:-43}
NUM_TRAIN_STEPS=${NUM_TRAIN_STEPS:-48000}
BATCH_SIZE=${BATCH_SIZE:-8}
NUM_WORKERS=${NUM_WORKERS:-0}
LOG_INTERVAL=${LOG_INTERVAL:-10}
SAVE_INTERVAL=${SAVE_INTERVAL:-12000}
KEEP_PERIOD=${KEEP_PERIOD:-12000}
ACTION_HORIZON=${ACTION_HORIZON:-10}
FSDP_DEVICES=${FSDP_DEVICES:-1}

module load apptainer
export TASK NUM_TRAIN_DEMOS HDF5_PATH EXP_NAME PROMPT CHECKPOINT_BASE_DIR
export TRAIN_SEED NUM_TRAIN_STEPS BATCH_SIZE NUM_WORKERS LOG_INTERVAL SAVE_INTERVAL
export KEEP_PERIOD ACTION_HORIZON FSDP_DEVICES

apptainer exec --nv \
  --no-home \
  --fakeroot \
  --contain \
  --bind ${SAILOR_HOST_DIR}:/SAILOR \
  "$SIF_PATH" \
  bash -lc '
    set -euo pipefail

    source /opt/conda/etc/profile.d/conda.sh
    conda activate robo_pi0

    unset SSL_CERT_FILE
    unset CURL_CA_BUNDLE
    unset REQUESTS_CA_BUNDLE
    export SSL_CERT_FILE="$(python3 -c "import certifi; print(certifi.where())")"
    export CURL_CA_BUNDLE="$SSL_CERT_FILE"
    export REQUESTS_CA_BUNDLE="$SSL_CERT_FILE"

    export XDG_CACHE_HOME=/SAILOR/.cache
    export OPENPI_DATA_HOME=/SAILOR/.cache/openpi
    export HF_HOME=/SAILOR/.cache/huggingface
    export HUGGINGFACE_HUB_CACHE=/SAILOR/.cache/huggingface/hub
    export TORCH_HOME=/SAILOR/.cache/torch
    export WANDB_MODE=disabled
    export PYTHONUNBUFFERED=1
    export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90
    export HOME=/tmp/home
    mkdir -p "$XDG_CACHE_HOME" "$OPENPI_DATA_HOME" "$HF_HOME" "$HUGGINGFACE_HUB_CACHE" "$TORCH_HOME" "$HOME"

    cd /SAILOR
    export PYTHONPATH=/SAILOR:${PYTHONPATH:-}

    python3 train_pi0_droid_lora_robomimic.py \
      --hdf5-path "$HDF5_PATH" \
      --task "$TASK" \
      --num-train-demos "$NUM_TRAIN_DEMOS" \
      --num-val-demos 0 \
      --seed "$TRAIN_SEED" \
      --action-horizon "$ACTION_HORIZON" \
      --prompt "$PROMPT" \
      --batch-size "$BATCH_SIZE" \
      --num-workers "$NUM_WORKERS" \
      --num-train-steps "$NUM_TRAIN_STEPS" \
      --exp-name "$EXP_NAME" \
      --checkpoint-base-dir "$CHECKPOINT_BASE_DIR" \
      --resume \
      --log-interval "$LOG_INTERVAL" \
      --save-interval "$SAVE_INTERVAL" \
      --keep-period "$KEEP_PERIOD" \
      --fsdp-devices "$FSDP_DEVICES"
  '
