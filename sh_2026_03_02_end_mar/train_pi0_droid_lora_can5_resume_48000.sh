#!/bin/bash
#SBATCH --job-name=pi0_lora_can5_resume_48000
#SBATCH --output=/home/ishakpie/projects/def-rhinehar/ishakpie/logs/pi0_lora_can5_resume_48000_%j.out
#SBATCH --error=/home/ishakpie/projects/def-rhinehar/ishakpie/logs/pi0_lora_can5_resume_48000_%j.err
#SBATCH --time=4:59:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --account=def-rhinehar
#SBATCH --mail-user=pierreishak2003@gmail.com
#SBATCH --mail-type=BEGIN,END,FAIL

set -euo pipefail
LAUNCHER_DIR=/home/ishakpie/projects/def-rhinehar/ishakpie/Sailor_Pi0/sh_2026_03_02_end_mar

export TASK=can
export NUM_TRAIN_DEMOS=5
export HDF5_PATH=/SAILOR/datasets/robomimic_datasets/can/ph/image_224_shaped_done1_v141.hdf5
export EXP_NAME=can5_seed43_24000
export PROMPT="Lift can and place in correct bin."

bash "$LAUNCHER_DIR/train_pi0_droid_lora_48000_resume_base.sh"
