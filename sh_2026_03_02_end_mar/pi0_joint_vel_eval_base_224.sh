#!/bin/bash

set -euo pipefail

: "${TASK:?Expected TASK to be set by the Slurm wrapper.}"
: "${PROMPT:?Expected PROMPT to be set by the Slurm wrapper.}"
: "${CHECKPOINT:?Expected CHECKPOINT to be set by the Slurm wrapper.}"

SIF_PATH=/home/ishakpie/projects/def-rhinehar/ishakpie/T7_py3_11_torch2_7_1_cuda12_6_robo_pi0.sif
PI_CONFIG_NAME=${PI_CONFIG_NAME:-pi0_droid}
PI0_EVAL_IMAGE_SIZE=${PI0_EVAL_IMAGE_SIZE:-224}

export TASK PI_CONFIG_NAME CHECKPOINT PROMPT PI0_EVAL_IMAGE_SIZE

module load apptainer

apptainer exec --nv \
  --no-home \
  --fakeroot \
  --contain \
  --bind /home/ishakpie/projects/def-rhinehar/ishakpie/Sailor_Pi0:/SAILOR \
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
    mkdir -p "$XDG_CACHE_HOME" "$OPENPI_DATA_HOME" "$HF_HOME" "$HUGGINGFACE_HUB_CACHE" "$TORCH_HOME/hub/checkpoints"

    export HOME=/tmp/home
    export MPLCONFIGDIR=/tmp/mplconfig
    export FC_CACHEDIR=/tmp/fontconfig
    export NUMBA_CACHE_DIR=/tmp/numba-cache
    export NUMBA_DISABLE_CACHING=1
    mkdir -p "$HOME" "$MPLCONFIGDIR" "$FC_CACHEDIR" "$NUMBA_CACHE_DIR"

    export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
    export LD_PRELOAD="$CONDA_PREFIX/lib/libpng16.so:$CONDA_PREFIX/lib/libjpeg.so.8:${LD_PRELOAD:-}"

    export WANDB_MODE=disabled
    export PYTHONUNBUFFERED=1
    export MUJOCO_GL=osmesa

    TRANSFORMERS_VERSION="$(python3 -c "import transformers; print(transformers.__version__)")"
    if [ "$TRANSFORMERS_VERSION" != "4.53.2" ]; then
      echo "ERROR: expected transformers==4.53.2, got $TRANSFORMERS_VERSION"
      exit 1
    fi
    TRANSFORMERS_DIR="$(python3 -c "import transformers, os; print(os.path.dirname(transformers.__file__))")"
    OPENPI_SITE_PATCH_DIR="/workspace/openpi/src/openpi/models_pytorch/transformers_replace"
    test -d "$OPENPI_SITE_PATCH_DIR" || { echo "ERROR: missing $OPENPI_SITE_PATCH_DIR"; exit 1; }
    test -d "$TRANSFORMERS_DIR" || { echo "ERROR: missing $TRANSFORMERS_DIR"; exit 1; }

    PATCH_HASH="$(find "$OPENPI_SITE_PATCH_DIR" -type f -print0 | sort -z | xargs -0 sha256sum | sha256sum | awk "{print \$1}")"
    TRANSFORMERS_OVERLAY_PARENT="/SAILOR/.cache/python-overlay"
    TRANSFORMERS_OVERLAY_DIR="$TRANSFORMERS_OVERLAY_PARENT/transformers"
    OVERLAY_SIGNATURE_FILE="$TRANSFORMERS_OVERLAY_PARENT/.overlay_signature"
    OVERLAY_SIGNATURE="$TRANSFORMERS_VERSION|$PATCH_HASH"
    mkdir -p "$TRANSFORMERS_OVERLAY_PARENT"

    if [ -d "$TRANSFORMERS_OVERLAY_DIR" ] && [ -f "$OVERLAY_SIGNATURE_FILE" ] && [ "$(cat "$OVERLAY_SIGNATURE_FILE")" = "$OVERLAY_SIGNATURE" ]; then
      echo "Using cached patched overlay at $TRANSFORMERS_OVERLAY_DIR"
    else
      echo "Building patched overlay at $TRANSFORMERS_OVERLAY_DIR"
      rm -rf "$TRANSFORMERS_OVERLAY_DIR"
      cp -a "$TRANSFORMERS_DIR" "$TRANSFORMERS_OVERLAY_DIR"
      cp -rv "$OPENPI_SITE_PATCH_DIR"/. "$TRANSFORMERS_OVERLAY_DIR"/
      printf "%s\n" "$OVERLAY_SIGNATURE" > "$OVERLAY_SIGNATURE_FILE"
    fi

    export PYTHONPATH="$TRANSFORMERS_OVERLAY_PARENT:${PYTHONPATH:-}"
    python3 -c "import os, sys, transformers; p=os.path.dirname(transformers.__file__); ok=p.startswith(\"/SAILOR/.cache/python-overlay\"); print(\"overlay in use:\", ok); print(\"active transformers path:\", p); sys.exit(0 if ok else 1)"

    EXPECTED_NORM_STATS="$CHECKPOINT/assets/droid/norm_stats.json"
    SOURCE_ASSETS_DIR="/SAILOR/.cache/openpi/openpi-assets/checkpoints/$PI_CONFIG_NAME/assets"
    if [ -d "$CHECKPOINT" ] && [ ! -f "$EXPECTED_NORM_STATS" ]; then
      test -d "$SOURCE_ASSETS_DIR" || { echo "ERROR: missing source assets at $SOURCE_ASSETS_DIR"; exit 1; }
      echo "Backfilling missing checkpoint assets from $SOURCE_ASSETS_DIR"
      rm -rf "$CHECKPOINT/assets"
      cp -a "$SOURCE_ASSETS_DIR" "$CHECKPOINT/assets"
    fi

    cd /SAILOR
    python3 pi0_joint_vel_final_simple_fix_pytorch_lora_eval224_full.py \
      --task "$TASK" \
      --num-envs 10 \
      --eval-num-runs 50 \
      --open-loop-horizon-pct 80 \
      --save-video \
      --prompt "$PROMPT" \
      --video-dir "/SAILOR/scratch_dir/rollouts/${TASK}/${SLURM_JOB_NAME}_${SLURM_JOB_ID}" \
      --pi-config-name "$PI_CONFIG_NAME" \
      --env-image-size "$PI0_EVAL_IMAGE_SIZE" \
      --checkpoint "$CHECKPOINT"
  '
