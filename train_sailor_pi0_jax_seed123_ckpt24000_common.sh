#!/bin/bash
set -euo pipefail

: "${JOB_NAME:=}"
: "${CHECKPOINT:=}"
: "${SELECTED_DEMOS_MANIFEST:=}"
: "${DATASET:=}"
: "${TASK:=}"
: "${PROMPT:=}"
: "${TRAIN_SEED:=}"
: "${WANDB_EXP_NAME:=}"
: "${RESUME_RUN_LOGDIR:=}"
: "${WORKER_SCRIPT_PATH:=}"
: "${SAILOR_AUTO_CHAIN:=1}"
: "${SAILOR_WARMSTART_INITIAL_HOURS:=12}"
: "${SAILOR_WARMSTART_INCREMENT_HOURS:=4}"
: "${SAILOR_RESUME_HOURS:=3}"
: "${SAILOR_RESUME_LONG_HOURS:=6}"
: "${SAILOR_INITIAL_MEM_GB:=128}"
: "${SAILOR_OOM_MEM_FACTOR:=1.2}"
: "${SAILOR_HELPER_PARTITION:=}"

for required_var in \
  JOB_NAME \
  CHECKPOINT \
  SELECTED_DEMOS_MANIFEST \
  DATASET \
  TASK \
  PROMPT \
  TRAIN_SEED \
  WANDB_EXP_NAME; do
  if [ -z "${!required_var}" ]; then
    echo "ERROR: missing required variable: ${required_var}"
    exit 1
  fi
done

: "${SIF_PATH:=/home/ishakpie/projects/def-rhinehar/ishakpie/T7_py3_11_torch2_7_1_cuda12_6_robo_pi0.sif}"
: "${SAILOR_HOST_DIR:=/home/ishakpie/projects/def-rhinehar/ishakpie/Sailor_Pi0}"
: "${HOST_LOG_DIR:=/home/ishakpie/projects/def-rhinehar/ishakpie/logs}"
: "${SLURM_ACCOUNT:=def-rhinehar}"

if [ -n "${SLURM_JOB_NAME:-}" ] && [ "${SLURM_JOB_NAME}" != "${JOB_NAME}" ]; then
  echo "ERROR: JOB_NAME (${JOB_NAME}) does not match SLURM_JOB_NAME (${SLURM_JOB_NAME})"
  exit 1
fi

CHECKPOINT_FAMILY="$(basename "$(dirname "${CHECKPOINT}")")"
if [[ "${CHECKPOINT_FAMILY}" =~ ^(lift|can|square)([0-9]+)_seed123_48000$ ]]; then
  CHECKPOINT_TASK="${BASH_REMATCH[1]}"
  NUM_EXP_TRAJS="${BASH_REMATCH[2]}"
else
  echo "ERROR: unsupported checkpoint family: ${CHECKPOINT_FAMILY}"
  exit 1
fi

EXPECTED_TASK="robomimic__${CHECKPOINT_TASK}"
if [ "${TASK}" != "${EXPECTED_TASK}" ]; then
  echo "ERROR: TASK (${TASK}) does not match checkpoint family (${CHECKPOINT_FAMILY})"
  exit 1
fi

to_host_path() {
  case "$1" in
    /SAILOR/*)
      printf '%s%s\n' "${SAILOR_HOST_DIR}" "${1#/SAILOR}"
      ;;
    *)
      printf '%s\n' "$1"
      ;;
  esac
}

run_state_training_complete() {
  python3 - "$1" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
if not path.exists():
    print("0")
    raise SystemExit(0)
data = json.loads(path.read_text())
done = bool(data.get("training_complete")) or data.get("last_completed_stage") == "training_complete"
print("1" if done else "0")
PY
}

slurm_duration_to_hours() {
  python3 - "$1" <<'PY'
import math
import sys

value = sys.argv[1].strip()
days = 0
if "-" in value:
    day_str, value = value.split("-", 1)
    days = int(day_str)
parts = [int(x) for x in value.split(":")]
while len(parts) < 3:
    parts.insert(0, 0)
hours = days * 24 + parts[0] + parts[1] / 60 + parts[2] / 3600
print(int(math.ceil(hours)))
PY
}

slurm_reqmem_to_gib() {
  python3 - "$1" <<'PY'
import math
import re
import sys

raw = sys.argv[1].strip()
if not raw or raw == "0":
    print(0)
    raise SystemExit(0)
match = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)([KMGTP])(?:[cn])?", raw, re.IGNORECASE)
if not match:
    raise SystemExit(f"Unsupported ReqMem value: {raw}")
value = float(match.group(1))
unit = match.group(2).upper()
scale = {"K": 1 / (1024 * 1024), "M": 1 / 1024, "G": 1, "T": 1024, "P": 1024 * 1024}
print(int(math.ceil(value * scale[unit])))
PY
}

submit_followup_helper() {
  if [ "${SAILOR_AUTO_CHAIN}" != "1" ] || [ -z "${SLURM_JOB_ID:-}" ]; then
    return
  fi

  local helper_script="${SAILOR_HOST_DIR}/continue_train_sailor_pi0_jax_seed123_ckpt24000.sh"
  if [ ! -f "${helper_script}" ]; then
    echo "ERROR: missing follow-up helper script: ${helper_script}"
    exit 1
  fi

  local job_details submitted_command resolved_worker_script_path
  local current_time_limit current_req_mem current_walltime_hours current_mem_gb
  job_details="$(scontrol show job "${SLURM_JOB_ID}" | tr '\n' ' ')"
  submitted_command="$(printf '%s\n' "${job_details}" | sed -n 's/.*Command=\([^ ]*\).*/\1/p')"
  resolved_worker_script_path="${WORKER_SCRIPT_PATH:-}"
  if [ -n "${submitted_command}" ] && [ -f "${submitted_command}" ]; then
    resolved_worker_script_path="${submitted_command}"
  fi
  if [ -z "${resolved_worker_script_path}" ] || [ ! -f "${resolved_worker_script_path}" ]; then
    echo "ERROR: unable to resolve worker script path for chain submission: ${resolved_worker_script_path:-<empty>}"
    exit 1
  fi
  current_time_limit="$(printf '%s\n' "${job_details}" | sed -n 's/.*TimeLimit=\([^ ]*\).*/\1/p')"
  current_req_mem="$(printf '%s\n' "${job_details}" | sed -n 's/.*ReqMem=\([^ ]*\).*/\1/p')"
  if [ -z "${current_req_mem}" ]; then
    current_req_mem="$(printf '%s\n' "${job_details}" | sed -n 's/.*MinMemoryNode=\([^ ]*\).*/\1/p')"
  fi
  if [ -z "${current_req_mem}" ]; then
    current_req_mem="$(printf '%s\n' "${job_details}" | sed -n 's/.*ReqTRES=[^ ]*mem=\([^, ]*\).*/\1/p')"
  fi
  current_walltime_hours="$(slurm_duration_to_hours "${current_time_limit}")"
  if [ -n "${current_req_mem}" ] && [ "${current_req_mem}" != "0" ]; then
    current_mem_gb="$(slurm_reqmem_to_gib "${current_req_mem}")"
  else
    current_mem_gb="${SAILOR_INITIAL_MEM_GB}"
  fi

  mkdir -p "${HOST_LOG_DIR}"

  local helper_job_id
  if [ -n "${SAILOR_HELPER_PARTITION}" ]; then
    helper_job_id="$(
      sbatch --parsable \
        --partition="${SAILOR_HELPER_PARTITION}" \
        --time=00:05:00 \
        --cpus-per-task=1 \
        --mem=512M \
        --account="${SLURM_ACCOUNT}" \
        --job-name="${SLURM_JOB_NAME}_continue" \
        --output="${HOST_LOG_DIR}/${SLURM_JOB_NAME}_${SLURM_JOB_ID}_continue_%j.out" \
        --error="${HOST_LOG_DIR}/${SLURM_JOB_NAME}_${SLURM_JOB_ID}_continue_%j.err" \
        --dependency="afterany:${SLURM_JOB_ID}" \
        --export=ALL,PREVIOUS_WORKER_JOB_ID="${SLURM_JOB_ID}",PREVIOUS_WALLTIME_HOURS="${current_walltime_hours}",PREVIOUS_MEM_GB="${current_mem_gb}",WORKER_SCRIPT_PATH="${resolved_worker_script_path}",RUN_LOGDIR_HOST="${RUN_LOGDIR_HOST}",SAILOR_HOST_DIR="${SAILOR_HOST_DIR}",HOST_LOG_DIR="${HOST_LOG_DIR}",JOB_NAME="${JOB_NAME}",SLURM_ACCOUNT="${SLURM_ACCOUNT}",SAILOR_AUTO_CHAIN="${SAILOR_AUTO_CHAIN}",SAILOR_WARMSTART_INITIAL_HOURS="${SAILOR_WARMSTART_INITIAL_HOURS}",SAILOR_WARMSTART_INCREMENT_HOURS="${SAILOR_WARMSTART_INCREMENT_HOURS}",SAILOR_RESUME_HOURS="${SAILOR_RESUME_HOURS}",SAILOR_RESUME_LONG_HOURS="${SAILOR_RESUME_LONG_HOURS}",SAILOR_INITIAL_MEM_GB="${SAILOR_INITIAL_MEM_GB}",SAILOR_OOM_MEM_FACTOR="${SAILOR_OOM_MEM_FACTOR}" \
        "${helper_script}"
    )"
  else
    helper_job_id="$(
      sbatch --parsable \
        --time=00:05:00 \
        --cpus-per-task=1 \
        --mem=512M \
        --account="${SLURM_ACCOUNT}" \
        --job-name="${SLURM_JOB_NAME}_continue" \
        --output="${HOST_LOG_DIR}/${SLURM_JOB_NAME}_${SLURM_JOB_ID}_continue_%j.out" \
        --error="${HOST_LOG_DIR}/${SLURM_JOB_NAME}_${SLURM_JOB_ID}_continue_%j.err" \
        --dependency="afterany:${SLURM_JOB_ID}" \
        --export=ALL,PREVIOUS_WORKER_JOB_ID="${SLURM_JOB_ID}",PREVIOUS_WALLTIME_HOURS="${current_walltime_hours}",PREVIOUS_MEM_GB="${current_mem_gb}",WORKER_SCRIPT_PATH="${resolved_worker_script_path}",RUN_LOGDIR_HOST="${RUN_LOGDIR_HOST}",SAILOR_HOST_DIR="${SAILOR_HOST_DIR}",HOST_LOG_DIR="${HOST_LOG_DIR}",JOB_NAME="${JOB_NAME}",SLURM_ACCOUNT="${SLURM_ACCOUNT}",SAILOR_AUTO_CHAIN="${SAILOR_AUTO_CHAIN}",SAILOR_WARMSTART_INITIAL_HOURS="${SAILOR_WARMSTART_INITIAL_HOURS}",SAILOR_WARMSTART_INCREMENT_HOURS="${SAILOR_WARMSTART_INCREMENT_HOURS}",SAILOR_RESUME_HOURS="${SAILOR_RESUME_HOURS}",SAILOR_RESUME_LONG_HOURS="${SAILOR_RESUME_LONG_HOURS}",SAILOR_INITIAL_MEM_GB="${SAILOR_INITIAL_MEM_GB}",SAILOR_OOM_MEM_FACTOR="${SAILOR_OOM_MEM_FACTOR}" \
        "${helper_script}"
    )"
  fi
  echo "Submitted follow-up helper ${helper_job_id} for worker ${SLURM_JOB_ID}"
}

DEFAULT_RUN_LOGDIR_HOST="${SAILOR_HOST_DIR}/scratch_dir/logs/${TASK,,}/${WANDB_EXP_NAME}_demos${NUM_EXP_TRAJS}/seed${TRAIN_SEED}"
if [ -z "${RESUME_RUN_LOGDIR}" ] && [ -f "${DEFAULT_RUN_LOGDIR_HOST}/run_state.json" ]; then
  RESUME_RUN_LOGDIR="${DEFAULT_RUN_LOGDIR_HOST}"
  echo "Auto-resuming from ${RESUME_RUN_LOGDIR}"
fi

RUN_LOGDIR_HOST="$(to_host_path "${RESUME_RUN_LOGDIR:-${DEFAULT_RUN_LOGDIR_HOST}}")"
RUN_STATE_PATH_HOST="${RUN_LOGDIR_HOST}/run_state.json"
if [ -f "${RUN_STATE_PATH_HOST}" ] && [ "$(run_state_training_complete "${RUN_STATE_PATH_HOST}")" = "1" ]; then
  echo "Training already complete for ${RUN_LOGDIR_HOST}; exiting without launching worker."
  exit 0
fi

if [ -n "${RESUME_RUN_LOGDIR}" ] && [[ "${RESUME_RUN_LOGDIR}" == "${SAILOR_HOST_DIR}"* ]]; then
  RESUME_RUN_LOGDIR="/SAILOR${RESUME_RUN_LOGDIR#${SAILOR_HOST_DIR}}"
fi

for required_path in \
  "${SIF_PATH}" \
  "${SAILOR_HOST_DIR}/third_party/SAILOR/train_sailor.py" \
  "$(to_host_path "${CHECKPOINT}")" \
  "$(to_host_path "${SELECTED_DEMOS_MANIFEST}")" \
  "$(to_host_path "${DATASET}")"; do
  if [ ! -e "${required_path}" ]; then
    echo "ERROR: missing required path: ${required_path}"
    exit 1
  fi
done

export JOB_NAME
export CHECKPOINT
export SELECTED_DEMOS_MANIFEST
export DATASET
export TASK
export PROMPT
export TRAIN_SEED
export WANDB_EXP_NAME
export NUM_EXP_TRAJS
export RESUME_RUN_LOGDIR
export WORKER_SCRIPT_PATH
export SAILOR_AUTO_CHAIN
export SAILOR_WARMSTART_INITIAL_HOURS
export SAILOR_WARMSTART_INCREMENT_HOURS
export SAILOR_RESUME_HOURS
export SAILOR_RESUME_LONG_HOURS
export SAILOR_INITIAL_MEM_GB
export SAILOR_OOM_MEM_FACTOR

submit_followup_helper

APPTAINER_WORKDIR="${SAILOR_HOST_DIR}/scratch_dir/apptainer_workdir/${SLURM_JOB_ID:-manual}"
mkdir -p "${APPTAINER_WORKDIR}"

module load apptainer

apptainer exec --nv \
  --no-home \
  --fakeroot \
  --contain \
  --workdir "${APPTAINER_WORKDIR}" \
  --bind "${SAILOR_HOST_DIR}:/SAILOR" \
  --bind "${HOST_LOG_DIR}:/JOB_LOGS" \
  "${SIF_PATH}" \
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
    resume_args=()
    if [ -n "${RESUME_RUN_LOGDIR:-}" ]; then
      resume_args=(--resume-run "$RESUME_RUN_LOGDIR")
    fi
    python3 ./third_party/SAILOR/train_sailor.py \
      "${resume_args[@]}" \
      --configs cfg_dp_mppi robomimic \
      --task "$TASK" \
      --num_exp_trajs "$NUM_EXP_TRAJS" \
      --num_exp_val_trajs 0 \
      --seed "$TRAIN_SEED" \
      --wandb_exp_name "$WANDB_EXP_NAME" \
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
