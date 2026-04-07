# Container Provenance

This folder contains a lightweight reproducibility snapshot extracted from the Apptainer image used for the project runtime.

It does **not** contain the actual `.sif` image. The image is an external file that is too large to store in the Git repository.

## What Is Included

- `T7_py3_11_torch2_7_1_cuda12_6_robo_pi0.inspect.def`
  - embedded Apptainer definition metadata extracted with `apptainer inspect --deffile`
- `T7_py3_11_torch2_7_1_cuda12_6_robo_pi0.runscript.sh`
  - image runscript extracted with `apptainer inspect --runscript`
- `T7_py3_11_torch2_7_1_cuda12_6_robo_pi0.environment.sh`
  - image environment metadata extracted with `apptainer inspect --environment`
- `robo_pi0_environment.yml`
  - conda environment export for the `robo_pi0` environment
- `robo_pi0_conda_list.txt`
  - full `conda list` snapshot for the same environment

## How To Interpret These Files

These files are useful for:

- documenting the runtime environment used in the project
- understanding dependency versions
- partially recreating the `robo_pi0` conda environment
- debugging differences between systems

These files are **not** a full self-sufficient container source tree.

In particular, the embedded `.def` file only records a parent local image path, so it should be treated as **provenance** rather than a portable rebuild recipe.

## Recreating the Environment

If you do not have the original `.sif`, the safest path is:

1. Start from a compatible CUDA + conda base image.
2. Recreate the `robo_pi0` conda environment from `robo_pi0_environment.yml`.
3. Use `robo_pi0_conda_list.txt` to check exact package versions when needed.
4. Point the repo's Apptainer launchers at your own rebuilt image with `SIF_PATH`.

## Notes

The actual runtime scripts in the repository assume an external Apptainer image and a bound repo path. See the root [README.md](../README.md) for the main usage instructions.
