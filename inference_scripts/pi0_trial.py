#!/usr/bin/env python3

# =========================
# STEP 0 — BASIC SYSTEM INFO
# =========================
import sys
import torch

print("\n==============================")
print("STEP 0: System & CUDA Check")
print("==============================")
print("Python executable:", sys.executable)
print("Python version:", sys.version)
print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("CUDA device name:", torch.cuda.get_device_name(0))
else:
    print("Running on CPU ❗")

print("System check complete.\n")


# =========================
# STEP 1 — IMPORTS
# =========================
print("==============================")
print("STEP 1: Importing libraries")
print("==============================")

import dataclasses
import jax

print("Imported: dataclasses")
print("Imported: jax")

from openpi.models import model as _model
from openpi.policies import droid_policy
from openpi.policies import policy_config as _policy_config
from openpi.shared import download
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader

print("Imported: openpi modules")
print("Imports complete.\n")


# =========================
# STEP 2 — LOAD CONFIG
# =========================
print("==============================")
print("STEP 2: Loading π0-DROID config")
print("==============================")

CONFIG_NAME = "pi0_droid"   # <-- THIS IS THE REAL π0-DROID MODEL
print("Config name:", CONFIG_NAME)

config = _config.get_config(CONFIG_NAME)

print("Config loaded successfully.")
print("Config object type:", type(config))
print("Config summary:")
print(config)
print()


# =========================
# STEP 3 — DOWNLOAD CHECKPOINT
# =========================
print("==============================")
print("STEP 3: Downloading model checkpoint")
print("==============================")

CHECKPOINT_PATH = "gs://openpi-assets/checkpoints/pi0_droid"
print("Checkpoint path:", CHECKPOINT_PATH)

checkpoint_dir = download.maybe_download(CHECKPOINT_PATH)

print("Checkpoint directory:", checkpoint_dir)
print("Checkpoint download complete.\n")


# =========================
# STEP 4 — CREATE TRAINED POLICY
# =========================
print("==============================")
print("STEP 4: Creating trained policy")
print("==============================")

policy = _policy_config.create_trained_policy(config, checkpoint_dir)

print("Policy object created.")
print("Policy type:", type(policy))
print("Policy ready for inference.\n")


# =========================
# STEP 5 — CREATE DROID INPUT
# =========================
print("==============================")
print("STEP 5: Creating DROID example input")
print("==============================")

example = droid_policy.make_droid_example()

print("DROID example created.")
print("Example type:", type(example))

# Inspect structure
if isinstance(example, dict):
    print("Example keys:", example.keys())
else:
    print("Example structure:", example)

print()


# =========================
# STEP 6 — RUN INFERENCE
# =========================
print("==============================")
print("STEP 6: Running π0-DROID inference")
print("==============================")

print("Running policy.infer(example)...")
result = policy.infer(example)

print("Inference complete.")
print("Result type:", type(result))

if isinstance(result, dict):
    print("Result keys:", result.keys())

print()


# =========================
# STEP 7 — INSPECT OUTPUT
# =========================
print("==============================")
print("STEP 7: Output inspection")
print("==============================")

if "actions" in result:
    actions = result["actions"]
    print("Actions found ✅")
    print("Actions type:", type(actions))
    print("Actions shape:", actions.shape)
    print("Actions dtype:", actions.dtype)
    print("Sample actions (first row):")
    print(actions[0])
else:
    print("No 'actions' key found in result ❌")

print()


# =========================
# STEP 8 — MEMORY CLEANUP
# =========================
print("==============================")
print("STEP 8: Cleanup")
print("==============================")

del policy
print("Policy object deleted.")
print("Memory cleanup complete.\n")


print("======================================")
print("π0-DROID pipeline execution COMPLETE ✅")
print("======================================")