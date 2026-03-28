#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

jobs=(
  "train_sailor_pi0_jax_lift5_seed123_ckpt24000_fullgpu.sh"
  "train_sailor_pi0_jax_lift10_seed123_ckpt24000_fullgpu.sh"
  "train_sailor_pi0_jax_lift15_seed123_ckpt24000_fullgpu.sh"
  "train_sailor_pi0_jax_can5_seed123_ckpt24000_fullgpu.sh"
  "train_sailor_pi0_jax_can10_seed123_ckpt24000_fullgpu.sh"
  "train_sailor_pi0_jax_can15_seed123_ckpt24000_fullgpu.sh"
  "train_sailor_pi0_jax_square50_seed123_ckpt24000_fullgpu.sh"
  "train_sailor_pi0_jax_square75_seed123_ckpt24000_fullgpu.sh"
  "train_sailor_pi0_jax_square100_seed123_ckpt24000_fullgpu.sh"
)

for job_script in "${jobs[@]}"; do
  script_path="${SCRIPT_DIR}/${job_script}"
  if [ ! -f "${script_path}" ]; then
    echo "ERROR: missing submit script: ${script_path}" >&2
    exit 1
  fi
done

for job_script in "${jobs[@]}"; do
  script_path="${SCRIPT_DIR}/${job_script}"
  job_id="$(sbatch --parsable "${script_path}")"
  echo "Submitted ${job_script} as ${job_id}"
done
