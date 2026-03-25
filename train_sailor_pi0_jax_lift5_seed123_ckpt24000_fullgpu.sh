#!/bin/bash
#SBATCH --job-name=sailor_pi0_lift5_ckpt24000_full_8d_diff
#SBATCH --output=/home/ishakpie/projects/def-rhinehar/ishakpie/logs/sailor_pi0_lift5_ckpt24000_full_8d_diff_%j.out
#SBATCH --error=/home/ishakpie/projects/def-rhinehar/ishakpie/logs/sailor_pi0_lift5_ckpt24000_full_8d_diff_%j.err
#SBATCH --time=72:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=80G
#SBATCH --gres=gpu:h100:1
#SBATCH --account=def-rhinehar
#SBATCH --mail-user=pierreishak2003@gmail.com
#SBATCH --mail-type=BEGIN,END,FAIL

set -euo pipefail

SIF_PATH=/home/ishakpie/projects/def-rhinehar/ishakpie/T7_py3_11_torch2_7_1_cuda12_6_robo_pi0.sif
SAILOR_HOST_DIR=/home/ishakpie/projects/def-rhinehar/ishakpie/Sailor_Pi0
HOST_LOG_DIR=/home/ishakpie/projects/def-rhinehar/ishakpie/logs

CHECKPOINT=/SAILOR/scratch_dir/checkpoints/pi0_droid_robomimic_hdf5/lift5_seed123_48000/24000
SELECTED_DEMOS_MANIFEST=/SAILOR/scratch_dir/checkpoints/pi0_droid_robomimic_hdf5/lift5_seed123_48000/selected_demos.json
DATASET=/SAILOR/datasets/robomimic_datasets/lift/ph/image_224_shaped_done1_v141.hdf5
TASK=robomimic__lift
PROMPT="Lift block above the table."
TRAIN_SEED=28

export CHECKPOINT
export SELECTED_DEMOS_MANIFEST
export DATASET
export TASK
export PROMPT
export TRAIN_SEED

for required_path in \
  "$SIF_PATH" \
  "$SAILOR_HOST_DIR/third_party/SAILOR/train_sailor.py" \
  "$SAILOR_HOST_DIR/scratch_dir/checkpoints/pi0_droid_robomimic_hdf5/lift5_seed123_48000/24000" \
  "$SAILOR_HOST_DIR/scratch_dir/checkpoints/pi0_droid_robomimic_hdf5/lift5_seed123_48000/selected_demos.json" \
  "$SAILOR_HOST_DIR/datasets/robomimic_datasets/lift/ph/image_224_shaped_done1_v141.hdf5"; do
  if [ ! -e "$required_path" ]; then
    echo "ERROR: missing required path: $required_path"
    exit 1
  fi
done

module load apptainer

apptainer exec --nv \
  --no-home \
  --fakeroot \
  --contain \
  --bind "${SAILOR_HOST_DIR}:/SAILOR" \
  --bind "${HOST_LOG_DIR}:/JOB_LOGS" \
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
    mkdir -p \
      "$XDG_CACHE_HOME" \
      "$OPENPI_DATA_HOME" \
      "$HF_HOME" \
      "$HUGGINGFACE_HUB_CACHE" \
      "$TORCH_HOME/hub/checkpoints"

    export HOME=/tmp/home
    export MPLCONFIGDIR=/tmp/mplconfig
    export FC_CACHEDIR=/tmp/fontconfig
    export NUMBA_CACHE_DIR=/tmp/numba-cache
    export NUMBA_DISABLE_CACHING=1
    mkdir -p "$HOME" "$MPLCONFIGDIR" "$FC_CACHEDIR" "$NUMBA_CACHE_DIR"

    export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
    export LD_PRELOAD="$CONDA_PREFIX/lib/libpng16.so:$CONDA_PREFIX/lib/libjpeg.so.8:${LD_PRELOAD:-}"
    export WANDB_MODE=disabled
    export WANDB_DISABLED=true
    export PYTHONUNBUFFERED=1
    export MUJOCO_GL=osmesa
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    export XLA_PYTHON_CLIENT_PREALLOCATE=false

    mkdir -p /JOB_LOGS
    if command -v nvidia-smi >/dev/null 2>&1; then
      GPU_LOG_FILE="/JOB_LOGS/${SLURM_JOB_NAME}_${SLURM_JOB_ID}_gpu_max.txt"
      echo "Logging peak GPU usage to $GPU_LOG_FILE"
      (
        max_util=0
        max_mem_used=0
        max_timestamp=""
        mem_total=""
        while true; do
          line="$(nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits | head -n 1 || true)"
          if [ -n "$line" ]; then
            IFS=, read -r timestamp util mem_used mem_total_raw <<< "$line"
            timestamp="$(printf "%s" "$timestamp" | xargs)"
            util="$(printf "%s" "$util" | xargs)"
            mem_used="$(printf "%s" "$mem_used" | xargs)"
            mem_total="$(printf "%s" "$mem_total_raw" | xargs)"
            if [ -n "$util" ] && [ "$util" -gt "$max_util" ]; then
              max_util="$util"
              max_timestamp="$timestamp"
            fi
            if [ -n "$mem_used" ] && [ "$mem_used" -gt "$max_mem_used" ]; then
              max_mem_used="$mem_used"
              max_timestamp="$timestamp"
            fi
            printf "job_name=%s\njob_id=%s\nsample_interval_sec=60\nmax_gpu_util_pct=%s\nmax_gpu_mem_used_mib=%s\ngpu_mem_total_mib=%s\npeak_seen_at=%s\nlast_sample_at=%s\n" \
              "$SLURM_JOB_NAME" \
              "$SLURM_JOB_ID" \
              "$max_util" \
              "$max_mem_used" \
              "$mem_total" \
              "$max_timestamp" \
              "$timestamp" > "$GPU_LOG_FILE"
          fi
          sleep 60
        done
      ) &
      GPU_MONITOR_PID=$!
      trap "kill $GPU_MONITOR_PID 2>/dev/null || true" EXIT
    fi

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
    SOURCE_ASSETS_DIR="/SAILOR/.cache/openpi/openpi-assets/checkpoints/pi0_droid/assets"
    if [ -d "$CHECKPOINT" ] && [ ! -f "$EXPECTED_NORM_STATS" ]; then
      test -d "$SOURCE_ASSETS_DIR" || { echo "ERROR: missing source assets at $SOURCE_ASSETS_DIR"; exit 1; }
      echo "Backfilling missing checkpoint assets from $SOURCE_ASSETS_DIR"
      rm -rf "$CHECKPOINT/assets"
      cp -a "$SOURCE_ASSETS_DIR" "$CHECKPOINT/assets"
    fi

    test -f "$SELECTED_DEMOS_MANIFEST" || { echo "ERROR: missing selected demos manifest at $SELECTED_DEMOS_MANIFEST"; exit 1; }
    test -f "$DATASET" || { echo "ERROR: missing dataset at $DATASET"; exit 1; }

    cd /SAILOR
    python3 ./third_party/SAILOR/train_sailor.py \
      --configs cfg_dp_mppi robomimic \
      --task "$TASK" \
      --num_exp_trajs 5 \
      --num_exp_val_trajs 0 \
      --seed "$TRAIN_SEED" \
      --wandb_exp_name lift5_24000 \
      --set base_policy_backend pi0_jax \
      --set pi0.checkpoint "$CHECKPOINT" \
      --set pi0.pi_config_name pi0_droid \
      --set pi0.prompt "$PROMPT" \
      --set pi0.env_image_size 224 \
      --set use_wandb False \
      --set eval_num_runs 50 \
      --set num_envs 10 \
      --set visualize_eval True \
      --set train_dp_mppi_params.eval_every_round 10 \
      --set train_dp_mppi_params.data_collect_noise_std 0.1 \
      --set logging.write_pi0_round_debug True
  '
