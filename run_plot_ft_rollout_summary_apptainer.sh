#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

: "${SIF_PATH:=/home/ishakpie/projects/def-rhinehar/ishakpie/T7_py3_11_torch2_7_1_cuda12_6_robo_pi0.sif}"
: "${SAILOR_HOST_DIR:=${SCRIPT_DIR}}"
: "${HOST_LOG_DIR:=/home/ishakpie/projects/def-rhinehar/ishakpie/logs}"
: "${APPTAINER_WORKDIR:=${SAILOR_HOST_DIR}/scratch_dir/apptainer_workdir/plot_ft_rollout_summary}"
: "${APPTAINER_USE_FAKEROOT:=0}"

mkdir -p "${APPTAINER_WORKDIR}" "${HOST_LOG_DIR}"

module load apptainer

apptainer_args=(
  exec
  --nv
  --no-home
  --contain
  --workdir "${APPTAINER_WORKDIR}"
  --bind "${SAILOR_HOST_DIR}:/SAILOR"
  --bind "${HOST_LOG_DIR}:/JOB_LOGS"
)

if [ "${APPTAINER_USE_FAKEROOT}" = "1" ]; then
  apptainer_args+=(--fakeroot)
fi

apptainer "${apptainer_args[@]}" "${SIF_PATH}" bash -lc '
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
    mkdir -p "$HOME"

    cd /SAILOR
    python3 /SAILOR/plot_ft_rollout_summary.py "$@"
  ' bash "$@"
