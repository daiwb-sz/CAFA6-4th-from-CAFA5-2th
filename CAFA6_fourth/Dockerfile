# 匹配CUDA12.8 RTX5090
FROM pytorch/pytorch:2.11.0-cuda12.8-cudnn9-runtime
WORKDIR /app

# 修复包名：blast-plus改为ncbi-blast+，移除hhsuite
RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget curl gcc g++ vim unzip openssl libssl-dev hhsuite hmmer ncbi-blast+ \
    && rm -rf /var/lib/apt/lists/*

# 安装 Miniforge
RUN curl -fsSL https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -o miniforge.sh \
    && bash miniforge.sh -b -p /opt/conda \
    && rm miniforge.sh

# 让 conda 可用
ENV PATH="/opt/conda/bin:$PATH"

# ========== 新增开始 ==========
RUN echo "source /opt/conda/etc/profile.d/conda.sh" >> /root/.bashrc \
    && echo "conda init bash" >> /root/.bashrc
# ========== 新增结束 ==========


# 安装python依赖
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# 复制项目代码
COPY . ./

CMD ["python", "run_pipline.py"]