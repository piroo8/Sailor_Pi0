#!/bin/bash

set -euo pipefail

LAUNCHER_DIR=/home/ishakpie/projects/def-rhinehar/ishakpie/Sailor_Pi0/sh_2026_03_02_end_mar
LOG_DIR=/home/ishakpie/projects/def-rhinehar/ishakpie/logs
BASE_CHECKPOINT=/SAILOR/.cache/openpi/openpi-assets/checkpoints/pi0_droid
FT_BASE=/SAILOR/scratch_dir/checkpoints/pi0_droid_robomimic_hdf5

submit_eval() {
  local job_name="$1"
  local task="$2"
  local checkpoint="$3"
  local prompt="$4"

  sbatch \
    --job-name="$job_name" \
    --output="$LOG_DIR/${job_name}_%j.out" \
    --error="$LOG_DIR/${job_name}_%j.err" \
    --time=00:30:00 \
    --cpus-per-task=1 \
    --mem=24G \
    --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1 \
    --account=def-rhinehar \
    --mail-user=pierreishak2003@gmail.com \
    --mail-type=FAIL \
    --export=ALL,TASK="$task",PROMPT="$prompt",CHECKPOINT="$checkpoint",PI0_EVAL_IMAGE_SIZE=224 \
    --wrap="bash '$LAUNCHER_DIR/pi0_joint_vel_eval_base_224.sh'"
}

#submit_eval "pi0_cached_lift_224" "robomimic__lift" "$BASE_CHECKPOINT" "Lift block above the table."
submit_eval "pi0_cached_can_224" "robomimic__can" "$BASE_CHECKPOINT" "Lift can and place in correct bin."
submit_eval "pi0_cached_square_224" "robomimic__square" "$BASE_CHECKPOINT" "Pick square tool and insert in slot."

#submit_eval "pi0_ft_lift5_23999_224" "robomimic__lift" "$FT_BASE/lift5_seed43_24000/23999" "Lift block above the table."
#submit_eval "pi0_ft_lift10_23999_224" "robomimic__lift" "$FT_BASE/lift10_seed43_24000/23999" "Lift block above the table."
#submit_eval "pi0_ft_lift15_23999_224" "robomimic__lift" "$FT_BASE/lift15_seed43_24000/23999" "Lift block above the table."

submit_eval "pi0_ft_can5_23999_224" "robomimic__can" "$FT_BASE/can5_seed43_24000/23999" "Lift can and place in correct bin."
submit_eval "pi0_ft_can10_23999_224" "robomimic__can" "$FT_BASE/can10_seed43_24000/23999" "Lift can and place in correct bin."
submit_eval "pi0_ft_can15_23999_224" "robomimic__can" "$FT_BASE/can15_seed43_24000/23999" "Lift can and place in correct bin."

submit_eval "pi0_ft_square50_23999_224" "robomimic__square" "$FT_BASE/square50_seed43_24000/23999" "Pick square tool and insert in slot."
submit_eval "pi0_ft_square75_23999_224" "robomimic__square" "$FT_BASE/square75_seed43_24000/23999" "Pick square tool and insert in slot."
submit_eval "pi0_ft_square100_23999_224" "robomimic__square" "$FT_BASE/square100_seed43_24000/23999" "Pick square tool and insert in slot."
