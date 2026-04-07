#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

: "${SIF_PATH:=/home/ishakpie/projects/def-rhinehar/ishakpie/T7_py3_11_torch2_7_1_cuda12_6_robo_pi0.sif}"
: "${SAILOR_HOST_DIR:=/home/ishakpie/projects/def-rhinehar/ishakpie/Sailor_Pi0}"
: "${OUR_RESULTS_ROOT:=/home/ishakpie/scratch/ishakpie_scratch}"
: "${OFFICIAL_RESULTS_ROOT:=/home/ishakpie/projects/def-rhinehar/ishakpie/SAILOR_fork/scratch_dir/logs}"
: "${OUTPUT_ROOT:=${OUR_RESULTS_ROOT}/comparison_exports}"
: "${APPTAINER_WORKDIR:=${SAILOR_HOST_DIR}/scratch_dir/apptainer_workdir/plot_round_eval_comparisons}"
: "${APPTAINER_USE_FAKEROOT:=0}"

mkdir -p "${APPTAINER_WORKDIR}" "${OUTPUT_ROOT}"

module load apptainer

apptainer_args=(
  exec
  --nv
  --no-home
  --contain
  --workdir "${APPTAINER_WORKDIR}"
  --bind "${SAILOR_HOST_DIR}:/SAILOR"
  --bind "${OUR_RESULTS_ROOT}:/OURS"
  --bind "${OFFICIAL_RESULTS_ROOT}:/OFFICIAL"
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
  python3 /SAILOR/plot_round_eval_comparisons.py \
    --source-root /OURS \
    --official-root /OFFICIAL \
    --output-root /OURS/comparison_exports \
    "$@"
' bash "$@"
