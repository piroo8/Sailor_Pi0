#!/bin/bash
#SBATCH --job-name=pi0_lora_lift5_24000
#SBATCH --output=/home/ishakpie/projects/def-rhinehar/ishakpie/logs/pi0_lora_lift5_24000_%j.out
#SBATCH --error=/home/ishakpie/projects/def-rhinehar/ishakpie/logs/pi0_lora_lift5_24000_%j.err
#SBATCH --time=5:59:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --account=def-rhinehar
#SBATCH --mail-user=pierreishak2003@gmail.com
#SBATCH --mail-type=BEGIN,END,FAIL

set -euo pipefail

# This wrapper mirrors the working inference container path on purpose so the
# local OpenPI bridge runs inside the same conda environment and cache layout.
# It does not launch any SAILOR diffusion code; the shared directory name is
# only the bind mount path that contains these local scripts.

SIF_PATH=/home/ishakpie/projects/def-rhinehar/ishakpie/T7_py3_11_torch2_7_1_cuda12_6_robo_pi0.sif
SAILOR_HOST_DIR=/home/ishakpie/projects/def-rhinehar/ishakpie/Sailor_Pi0
HDF5_PATH=/SAILOR/datasets/robomimic_datasets/lift/ph/image_224_shaped_done1_v141.hdf5
EXP_NAME=lift5_seed43_24000
PROMPT="Lift block above the table."
CHECKPOINT_BASE_DIR=/SAILOR/scratch_dir/checkpoints

module load apptainer
export HDF5_PATH EXP_NAME PROMPT CHECKPOINT_BASE_DIR

apptainer exec --nv \
  --no-home \
  --fakeroot \
  --contain \
  --bind ${SAILOR_HOST_DIR}:/SAILOR \
  "$SIF_PATH" \
  bash -lc '
    set -euo pipefail

    # Step 1: Activate the same container and conda environment used by the working eval path.
    source /opt/conda/etc/profile.d/conda.sh
    conda activate robo_pi0

    # Step 2: Recreate the cache and SSL environment that OpenPI already expects inside the container.
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
    export WANDB_MODE=disabled
    export PYTHONUNBUFFERED=1
    export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90
    export HOME=/tmp/home
    # Keep transient home/cache state writable even though the container runs
    # without the real home directory mounted inside.
    mkdir -p "$XDG_CACHE_HOME" "$OPENPI_DATA_HOME" "$HF_HOME" "$HUGGINGFACE_HUB_CACHE" "$TORCH_HOME" "$HOME"

    cd /SAILOR
    # The local bridge files live directly under /SAILOR, so prepend that path
    # instead of depending on any editable OpenPI checkout in the container.
    export PYTHONPATH=/SAILOR:${PYTHONPATH:-}

    # Step 3: Launch a longer LoRA training run using exactly 5 demos total.
    # No validation demo is reserved; checkpoint choice happens via rollout
    # evaluation after training. The 24000-step target matches the Sailor
    # RoboMimic diffusion train_steps.
    python3 train_pi0_droid_lora_robomimic.py \
      --hdf5-path "$HDF5_PATH" \
      --task lift \
      --num-train-demos 5 \
      --num-val-demos 0 \
      --seed 43 \
      --action-horizon 10 \
      --prompt "$PROMPT" \
      --batch-size 8 \
      --num-workers 0 \
      --num-train-steps 24000 \
      --exp-name "$EXP_NAME" \
      --checkpoint-base-dir "$CHECKPOINT_BASE_DIR" \
      --overwrite \
      --log-interval 10 \
      --save-interval 1000 \
      --keep-period 1000 \
      --fsdp-devices 1
  '
