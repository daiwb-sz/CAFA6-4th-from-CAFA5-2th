#!/bin/bash
set -e  # 出错时终止脚本，方便排查

# 打印当前路径和conda版本
pwd
conda --version

# 创建pytorch环境（Python 3.10更适配CUDA 12.8，避免3.9的兼容性问题）
conda create --solver=libmamba -p $1/pytorch-env python=3.10 -y
eval "$(conda shell.bash hook)"
conda activate $1/pytorch-env

# 关键：不通过conda装pytorch（避免CUDA版本冲突），直接用你需要的pip命令装cu128版本
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# 安装其他依赖（升级兼容CUDA 12.8的版本）
# conda install --solver=libmamba -c conda-forge cupy-cuda12x -y  # cupy适配CUDA 12.x
pip install cupy-cuda12x==13.0.0
pip install --upgrade pip
pip install joblib tqdm pandas==2.1.0 pyyaml pyarrow numba==0.58.1 scikit-learn==1.3.0 numpy scipy fair-esm
pip install obonet pyvis transformers torchmetrics torchsummary sentencepiece psutil

# 验证安装
echo "验证PyTorch CUDA是否可用："
python -c "import torch; print(f'CUDA可用: {torch.cuda.is_available()}'); print(f'CUDA版本: {torch.version.cuda}')"