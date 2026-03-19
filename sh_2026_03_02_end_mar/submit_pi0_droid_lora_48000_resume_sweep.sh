#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

scripts=(
  train_pi0_droid_lora_lift5_resume_48000.sh
  train_pi0_droid_lora_lift10_resume_48000.sh
  train_pi0_droid_lora_lift15_resume_48000.sh
  train_pi0_droid_lora_can5_resume_48000.sh
  train_pi0_droid_lora_can10_resume_48000.sh
  train_pi0_droid_lora_can15_resume_48000.sh
  train_pi0_droid_lora_square50_resume_48000.sh
  train_pi0_droid_lora_square75_resume_48000.sh
  train_pi0_droid_lora_square100_resume_48000.sh
)

for script in "${scripts[@]}"; do
  job_id="$(sbatch --parsable "$SCRIPT_DIR/$script")"
  echo "Submitted $script -> $job_id"
done
