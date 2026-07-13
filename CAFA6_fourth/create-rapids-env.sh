#!/bin/bash
set -e

conda --version

# 创建rapids环境（明确指定支持CUDA 12.8的版本）
# RAPIDS 23.10支持CUDA 12.0-12.2，24.02支持CUDA 12.2-12.4，24.06支持CUDA 12.4+
# 若CUDA 12.8无直接匹配版本，优先选24.06（向下兼容）
conda create --solver=libmamba -p $1/rapids-env -c rapidsai -c conda-forge -c nvidia \
    rapids=24.06 python=3.10 cuda-version=12.4 -y  # 12.4向下兼容12.8
eval "$(conda shell.bash hook)"
conda activate $1/rapids-env
which python

# 卸载冲突依赖，重装适配CUDA 12.8的版本
pip uninstall -y cupy numba
pip install --upgrade pip
pip install tqdm cupy-cuda12x==13.0.0 numba==0.58.1 py-boost==0.4.3

# 验证RAPIDS/CUDA是否可用
echo "验证RAPIDS/cupy是否可用："
python -c "import cudf, cupy; print(f'cudf版本: {cudf.__version__}'); print(f'cupy CUDA版本: {cupy.cuda.runtime.runtimeGetVersion()}')"