=== /.singularity.d/env/10-docker2singularity.sh ===
#!/bin/sh
export PATH="/usr/local/nvidia/bin:/usr/local/cuda/bin:/opt/conda/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
export NVARCH="${NVARCH:-"x86_64"}"
export NVIDIA_REQUIRE_CUDA="${NVIDIA_REQUIRE_CUDA:-"cuda>=12.4 brand=tesla,driver>=470,driver<471 brand=unknown,driver>=470,driver<471 brand=nvidia,driver>=470,driver<471 brand=nvidiartx,driver>=470,driver<471 brand=geforce,driver>=470,driver<471 brand=geforcertx,driver>=470,driver<471 brand=quadro,driver>=470,driver<471 brand=quadrortx,driver>=470,driver<471 brand=titan,driver>=470,driver<471 brand=titanrtx,driver>=470,driver<471 brand=tesla,driver>=525,driver<526 brand=unknown,driver>=525,driver<526 brand=nvidia,driver>=525,driver<526 brand=nvidiartx,driver>=525,driver<526 brand=geforce,driver>=525,driver<526 brand=geforcertx,driver>=525,driver<526 brand=quadro,driver>=525,driver<526 brand=quadrortx,driver>=525,driver<526 brand=titan,driver>=525,driver<526 brand=titanrtx,driver>=525,driver<526 brand=tesla,driver>=535,driver<536 brand=unknown,driver>=535,driver<536 brand=nvidia,driver>=535,driver<536 brand=nvidiartx,driver>=535,driver<536 brand=geforce,driver>=535,driver<536 brand=geforcertx,driver>=535,driver<536 brand=quadro,driver>=535,driver<536 brand=quadrortx,driver>=535,driver<536 brand=titan,driver>=535,driver<536 brand=titanrtx,driver>=535,driver<536"}"
export NV_CUDA_CUDART_VERSION="${NV_CUDA_CUDART_VERSION:-"12.4.127-1"}"
export NV_CUDA_COMPAT_PACKAGE="${NV_CUDA_COMPAT_PACKAGE:-"cuda-compat-12-4"}"
export CUDA_VERSION="${CUDA_VERSION:-"12.4.1"}"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-"/usr/local/nvidia/lib:/usr/local/nvidia/lib64"}"
export NVIDIA_VISIBLE_DEVICES="${NVIDIA_VISIBLE_DEVICES:-"all"}"
export NVIDIA_DRIVER_CAPABILITIES="${NVIDIA_DRIVER_CAPABILITIES:-"compute,utility"}"
export NV_CUDA_LIB_VERSION="${NV_CUDA_LIB_VERSION:-"12.4.1-1"}"
export NV_NVTX_VERSION="${NV_NVTX_VERSION:-"12.4.127-1"}"
export NV_LIBNPP_VERSION="${NV_LIBNPP_VERSION:-"12.2.5.30-1"}"
export NV_LIBNPP_PACKAGE="${NV_LIBNPP_PACKAGE:-"libnpp-12-4=12.2.5.30-1"}"
export NV_LIBCUSPARSE_VERSION="${NV_LIBCUSPARSE_VERSION:-"12.3.1.170-1"}"
export NV_LIBCUBLAS_PACKAGE_NAME="${NV_LIBCUBLAS_PACKAGE_NAME:-"libcublas-12-4"}"
export NV_LIBCUBLAS_VERSION="${NV_LIBCUBLAS_VERSION:-"12.4.5.8-1"}"
export NV_LIBCUBLAS_PACKAGE="${NV_LIBCUBLAS_PACKAGE:-"libcublas-12-4=12.4.5.8-1"}"
export NV_LIBNCCL_PACKAGE_NAME="${NV_LIBNCCL_PACKAGE_NAME:-"libnccl2"}"
export NV_LIBNCCL_PACKAGE_VERSION="${NV_LIBNCCL_PACKAGE_VERSION:-"2.21.5-1"}"
export NCCL_VERSION="${NCCL_VERSION:-"2.21.5-1"}"
export NV_LIBNCCL_PACKAGE="${NV_LIBNCCL_PACKAGE:-"libnccl2=2.21.5-1+cuda12.4"}"
export NVIDIA_PRODUCT_NAME="${NVIDIA_PRODUCT_NAME:-"CUDA"}"
export NV_CUDA_CUDART_DEV_VERSION="${NV_CUDA_CUDART_DEV_VERSION:-"12.4.127-1"}"
export NV_NVML_DEV_VERSION="${NV_NVML_DEV_VERSION:-"12.4.127-1"}"
export NV_LIBCUSPARSE_DEV_VERSION="${NV_LIBCUSPARSE_DEV_VERSION:-"12.3.1.170-1"}"
export NV_LIBNPP_DEV_VERSION="${NV_LIBNPP_DEV_VERSION:-"12.2.5.30-1"}"
export NV_LIBNPP_DEV_PACKAGE="${NV_LIBNPP_DEV_PACKAGE:-"libnpp-dev-12-4=12.2.5.30-1"}"
export NV_LIBCUBLAS_DEV_VERSION="${NV_LIBCUBLAS_DEV_VERSION:-"12.4.5.8-1"}"
export NV_LIBCUBLAS_DEV_PACKAGE_NAME="${NV_LIBCUBLAS_DEV_PACKAGE_NAME:-"libcublas-dev-12-4"}"
export NV_LIBCUBLAS_DEV_PACKAGE="${NV_LIBCUBLAS_DEV_PACKAGE:-"libcublas-dev-12-4=12.4.5.8-1"}"
export NV_CUDA_NSIGHT_COMPUTE_VERSION="${NV_CUDA_NSIGHT_COMPUTE_VERSION:-"12.4.1-1"}"
export NV_CUDA_NSIGHT_COMPUTE_DEV_PACKAGE="${NV_CUDA_NSIGHT_COMPUTE_DEV_PACKAGE:-"cuda-nsight-compute-12-4=12.4.1-1"}"
export NV_NVPROF_VERSION="${NV_NVPROF_VERSION:-"12.4.127-1"}"
export NV_NVPROF_DEV_PACKAGE="${NV_NVPROF_DEV_PACKAGE:-"cuda-nvprof-12-4=12.4.127-1"}"
export NV_LIBNCCL_DEV_PACKAGE_NAME="${NV_LIBNCCL_DEV_PACKAGE_NAME:-"libnccl-dev"}"
export NV_LIBNCCL_DEV_PACKAGE_VERSION="${NV_LIBNCCL_DEV_PACKAGE_VERSION:-"2.21.5-1"}"
export NV_LIBNCCL_DEV_PACKAGE="${NV_LIBNCCL_DEV_PACKAGE:-"libnccl-dev=2.21.5-1+cuda12.4"}"
export LIBRARY_PATH="${LIBRARY_PATH:-"/usr/local/cuda/lib64/stubs"}"
export PYTORCH_VERSION="${PYTORCH_VERSION:-"2.6.0"}"

=== /.singularity.d/env/90-environment.sh ===
#!/bin/sh
# Copyright (c) Contributors to the Apptainer project, established as
#   Apptainer a Series of LF Projects LLC.
#   For website terms of use, trademark policy, privacy policy and other
#   project policies see https://lfprojects.org/policies
# Copyright (c) 2018-2021, Sylabs Inc. All rights reserved.
# This software is licensed under a 3-clause BSD license. Please consult
# https://github.com/apptainer/apptainer/blob/main/LICENSE.md regarding your
# rights to use or distribute this software.

# Custom environment shell code should follow

