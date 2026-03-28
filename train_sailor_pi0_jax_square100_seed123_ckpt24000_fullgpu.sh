#!/bin/bash
#SBATCH --job-name=sailor_pi0_square100_ckpt24000_full_8d_diff
#SBATCH --output=/home/ishakpie/projects/def-rhinehar/ishakpie/logs/sailor_pi0_square100_ckpt24000_full_8d_diff_%j.out
#SBATCH --error=/home/ishakpie/projects/def-rhinehar/ishakpie/logs/sailor_pi0_square100_ckpt24000_full_8d_diff_%j.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --gres=gpu:h100:1
#SBATCH --account=def-rhinehar
#SBATCH --mail-user=pierreishak2003@gmail.com
#SBATCH --mail-type=BEGIN,END,FAIL

set -euo pipefail

# Resolve shared scripts against the stable host checkout, not Slurm's spool copy.
SAILOR_HOST_DIR=/home/ishakpie/projects/def-rhinehar/ishakpie/Sailor_Pi0
COMMON_SCRIPT_PATH="${SAILOR_HOST_DIR}/train_sailor_pi0_jax_seed123_ckpt24000_common.sh"

JOB_NAME=sailor_pi0_square100_ckpt24000_full_8d_diff
CHECKPOINT_ROOT=/SAILOR/scratch_dir/checkpoints/pi0_droid_robomimic_hdf5/square100_seed123_48000
CHECKPOINT="${CHECKPOINT_ROOT}/24000"
SELECTED_DEMOS_MANIFEST="${CHECKPOINT_ROOT}/selected_demos.json"
DATASET=/SAILOR/datasets/robomimic_datasets/square/ph/image_224_shaped_done1_v141.hdf5
TASK=robomimic__square
PROMPT="Pick square tool and insert in slot."
: "${TRAIN_SEED:=108}"
WANDB_EXP_NAME=square100_24000
: "${RESUME_RUN_LOGDIR:=}"
: "${SAILOR_AUTO_CHAIN:=1}"
: "${SAILOR_WARMSTART_INITIAL_HOURS:=12}"
: "${SAILOR_WARMSTART_INCREMENT_HOURS:=4}"
: "${SAILOR_RESUME_HOURS:=3}"
: "${SAILOR_RESUME_LONG_HOURS:=6}"
: "${SAILOR_INITIAL_MEM_GB:=128}"
: "${SAILOR_OOM_MEM_FACTOR:=1.2}"
WORKER_SCRIPT_PATH="${SAILOR_HOST_DIR}/train_sailor_pi0_jax_square100_seed123_ckpt24000_fullgpu.sh"

export JOB_NAME CHECKPOINT SELECTED_DEMOS_MANIFEST DATASET TASK PROMPT TRAIN_SEED WANDB_EXP_NAME RESUME_RUN_LOGDIR WORKER_SCRIPT_PATH SAILOR_HOST_DIR
export SAILOR_AUTO_CHAIN SAILOR_WARMSTART_INITIAL_HOURS SAILOR_WARMSTART_INCREMENT_HOURS SAILOR_RESUME_HOURS
export SAILOR_RESUME_LONG_HOURS SAILOR_INITIAL_MEM_GB SAILOR_OOM_MEM_FACTOR
exec bash "${COMMON_SCRIPT_PATH}"
