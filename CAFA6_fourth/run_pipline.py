#!/usr/bin/env python
# coding: utf-8
"""
CAFA 蛋白质功能预测完整流水线复现脚本
整合原 Jupyter Notebook 全流程，模块化重构，增加异常处理与路径校验
"""
import sys
import os
import glob
import yaml
import subprocess
import traceback
from typing import List, Optional

# ====================== 全局常量 & 工具函数 ======================
# 日志打印封装
def log(msg: str) -> None:
    """统一日志输出"""
    print(f"[CAFA-PIPE] {msg}", flush=True)

# 安全执行shell命令，捕获异常与返回码
def run_cmd(cmd: List[str], env_bin: Optional[str] = None) -> None:
    """
    执行外部脚本命令
    :param cmd: 命令分段列表
    :param env_bin: conda环境python解释器路径，非空则前置
    """
    full_cmd = []
    if env_bin is not None and os.path.exists(env_bin):
        full_cmd.append(env_bin)
    full_cmd.extend(cmd)

    log(f"执行命令: {' '.join(full_cmd)}")
    try:
        proc = subprocess.run(
            full_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
        if proc.stdout.strip():
            log(f"命令输出:\n{proc.stdout}")
    except subprocess.CalledProcessError as e:
        log(f"命令执行失败！返回码: {e.returncode}")
        log(f"STDERR:\n{e.stderr}")
        raise RuntimeError(f"子进程命令失败: {' '.join(full_cmd)}") from e
    except Exception as e:
        log(f"执行命令未知异常: {traceback.format_exc()}")
        raise

# 路径安全拼接 + 自动创建目录
def safe_mkdir(path: str) -> str:
    """不存在则创建目录，返回完整路径"""
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
        log(f"创建目录: {path}")
    return path

# 校验文件是否存在，不存在直接抛异常阻断流程
def check_file_exists(file_path: str) -> None:
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"缺失关键文件: {file_path}")

# ====================== 配置加载模块 ======================
def load_config(config_yaml_path: str = "config.yaml") -> dict:
    """加载并校验config.yaml配置文件"""
    log("开始加载配置文件 config.yaml")
    check_file_exists(config_yaml_path)
    with open(config_yaml_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # 校验必要配置字段
    required_keys = ["base_path", "rapids-env", "pytorch-env"]
    for k in required_keys:
        if k not in cfg:
            raise ValueError(f"config.yaml 缺失必填字段: {k}")

    # 生成全局路径
    base_path = cfg["base_path"]
    cfg["CONFIG_PATH"] = os.path.join(base_path, "config.yaml")
    cfg["RAPIDS_ENV"] = os.path.join(base_path, cfg["rapids-env"])
    cfg["PYTORCH_ENV"] = os.path.join(base_path, cfg["pytorch-env"])

    return cfg

# ====================== 阶段1：环境初始化 ======================
def stage_1_setup_envs(cfg: dict) -> None:
    """1.1 创建Rapids、Pytorch虚拟环境"""
    log("===== 阶段1.1：初始化conda虚拟环境 =====")
    base = cfg["base_path"]
    rapids_sh = os.path.join(base, "create-rapids-env.sh")
    torch_sh = os.path.join(base, "create-pytorch-env.sh")
    check_file_exists(rapids_sh)
    check_file_exists(torch_sh)

    run_cmd([rapids_sh, base])
    run_cmd([torch_sh, base])
        # 校验环境解释器存在
    check_file_exists(cfg["RAPIDS_ENV"])
    check_file_exists(cfg["PYTORCH_ENV"])
    log("配置加载完成，路径校验通过")
    log("虚拟环境创建完成")

def stage_1_preprocess_helpers(cfg: dict) -> None:
    """1.3 解析FASTA、生成target辅助parquet数据"""
    log("===== 阶段1.3：预处理FASTA与目标标签辅助文件 =====")
    rapids_py = cfg["RAPIDS_ENV"]
    base = cfg["base_path"]
    config_path = cfg["CONFIG_PATH"]

    # 解析fasta
    run_cmd(
        [os.path.join(base, "protlib/scripts/parse_fasta.py"), "--config-path", config_path],
        env_bin=rapids_py
    )
    # 生成target helpers
    run_cmd(
        [
            os.path.join(base, "protlib/scripts/create_helpers.py"),
            "--config-path", config_path,
            "--batch-size", "10000",
            "--propagate", "true"
        ],
        env_bin=rapids_py
    )
    log("FASTA & target辅助文件生成完毕")

def stage_1_external_go_data(cfg: dict) -> None:
    """1.4 下载&解析外部GOA数据库、标签传播、复现MT数据集"""
    log("===== 阶段1.4：处理外部GOA时序数据 =====")
    rapids_py = cfg["RAPIDS_ENV"]
    base = cfg["base_path"]
    config_path = cfg["CONFIG_PATH"]
    temporal_folder = os.path.join(base, "temporal")
    safe_mkdir(temporal_folder)

    # 1.4.1 下载GOA原始文件
    dw_script = os.path.join(base, "protlib/scripts/downloads/dw_goant.py")
    run_cmd([dw_script, "--config-path", config_path], env_bin=rapids_py)

    # 1.4.2 解析两份GAF压缩包
    parse_args = [
        os.path.join(base, "protlib/scripts/parse_go_single.py"),
        "--config-path", config_path
    ]
    run_cmd(parse_args + ["--file", "goa_uniprot_all.gaf.228.gz"], env_bin=rapids_py)
    run_cmd(parse_args + ["--file", "goa_uniprot_all.gaf.226.gz", "--output", "old226"], env_bin=rapids_py)

    # 1.4.3 批量标签传播prop_tsv
    label_files = glob.glob(os.path.join(temporal_folder, "labels/train*")) + glob.glob(os.path.join(temporal_folder, "labels/test*"))
    obo_graph = os.path.join(base, "Train/go-basic.obo")
    check_file_exists(obo_graph)

    for f in label_files:
        out_name = os.path.join(temporal_folder, "labels", f"prop_{os.path.basename(f)}")
        prop_cmd = [
            os.path.join(base, "protlib/scripts/prop_tsv.py"),
            "--path", f,
            "--graph", obo_graph,
            "--output", out_name,
            "--batch_size", "30000",
            "--batch_inner", "5000"
        ]
        run_cmd(prop_cmd, env_bin=rapids_py)

    # 1.4.4 复现MT数据集 & QuickGO传播
    mt_script = os.path.join(base, "protlib/scripts/reproduce_mt.py")
    run_cmd([mt_script, "--path", temporal_folder, "--graph", obo_graph], env_bin=rapids_py)

    quickgo_in = os.path.join(temporal_folder, "quickgo51.tsv")
    quickgo_out = os.path.join(temporal_folder, "prop_quickgo51.tsv")
    run_cmd(
        [
            os.path.join(base, "protlib/scripts/prop_tsv.py"),
            "--path", quickgo_in,
            "--graph", obo_graph,
            "--output", quickgo_out,
            "--batch_size", "30000",
            "--batch_inner", "5000"
        ],
        env_bin=rapids_py
    )
    log("外部GO数据处理全部完成")

def stage_1_nn_prepare(cfg: dict) -> None:
    """1.5 神经网络训练辅助特征文件生成"""
    log("===== 阶段1.5：神经网络训练辅助数据准备 =====")
    torch_py = cfg["PYTORCH_ENV"]
    base = cfg["base_path"]
    config_path = cfg["CONFIG_PATH"]
    run_cmd(
        [os.path.join(base, "nn_solution/prepare.py"), "--config-path", config_path],
        env_bin=torch_py
    )
    log("NN辅助特征文件生成完成")

# ====================== 阶段2：蛋白质Embedding提取 ======================
def stage_2_extract_embeds(cfg: dict) -> None:
    log("===== 阶段2：T5 / ESM2 预训练Embedding提取 =====")
    torch_py = cfg["PYTORCH_ENV"]
    base = cfg["base_path"]
    config_path = cfg["CONFIG_PATH"]
    safe_mkdir(os.path.join(base, "embeds"))

    # T5 embedding
    run_cmd(
        [os.path.join(base, "nn_solution/t5.py"), "--config-path", config_path, "--device", "0"],
        env_bin=torch_py
    )
    # ESM2 small embedding
    run_cmd(
        [os.path.join(base, "nn_solution/esm2sm.py"), "--config-path", config_path, "--device", "0"],
        env_bin=torch_py
    )
    log("Embedding提取全部完成")

# ====================== 阶段3：基础模型训练（PyBoost / 线性LR / NN） ======================
def stage_3_train_base_models(cfg: dict) -> None:
    log("===== 阶段3：训练基础模型 PyBoost / LogReg / 神经网络 =====")
    rapids_py = cfg["RAPIDS_ENV"]
    torch_py = cfg["PYTORCH_ENV"]
    base = cfg["base_path"]
    config_path = cfg["CONFIG_PATH"]
    safe_mkdir(os.path.join(base, "models"))

    # 3.1 PyBoost GBDT模型
    pb_model_list = ["pb_t5esm4500_raw", "pb_t54500_raw", "pb_t54500_cond", "pb_t5esm4500_cond"]
    for m_name in pb_model_list:
        log(f"开始训练PyBoost模型: {m_name}")
        run_cmd(
            [
                os.path.join(base, "protlib/scripts/train_pb.py"),
                "--config-path", config_path,
                "--model-name", m_name,
                "--device", "0"
            ],
            env_bin=rapids_py
        )

    # 3.2 逻辑回归线性模型
    lin_model_list = ["lin_t5_raw", "lin_t5_cond"]
    for m_name in lin_model_list:
        log(f"开始训练LogReg模型: {m_name}")
        run_cmd(
            [
                os.path.join(base, "protlib/scripts/train_lin.py"),
                "--config-path", config_path,
                "--model-name", m_name,
                "--device", "0"
            ],
            env_bin=rapids_py
        )

    # 3.3 深度神经网络模型
    # 生成KFold
    run_cmd(
        [os.path.join(base, "protlib/scripts/create_gkf.py"), "--config-path", config_path],
        env_bin=rapids_py
    )
    # 训练NN
    run_cmd(
        [os.path.join(base, "nn_solution/train_models.py"), "--config-path", config_path, "--device", "0"],
        env_bin=torch_py
    )
    # NN推理
    run_cmd(
        [os.path.join(base, "nn_solution/inference_models.py"), "--config-path", config_path, "--device", "0"],
        env_bin=torch_py
    )
    # 输出统一pkl供堆叠使用
    run_cmd(
        [os.path.join(base, "nn_solution/make_pkl.py"), "--config-path", config_path],
        env_bin=torch_py
    )
    log("全部基础模型训练&推理完成")

# ====================== 阶段4：GCN堆叠模型 + 后处理生成提交文件 ======================
def stage_4_gcn_stack_and_submit(cfg: dict) -> None:
    log("===== 阶段4：GCN堆叠模型训练、TTA推理、后处理生成提交文件 =====")
    rapids_py = cfg["RAPIDS_ENV"]
    torch_py = cfg["PYTORCH_ENV"]
    base = cfg["base_path"]
    config_path = cfg["CONFIG_PATH"]
    safe_mkdir(os.path.join(base, "models/gcn"))
    safe_mkdir(os.path.join(base, "sub"))

    # 4.1 分本体训练GCN BP/MF/CC
    ont_list = ["bp", "mf", "cc"]
    for ont in ont_list:
        log(f"训练GCN {ont.upper()}")
        run_cmd(
            [
                os.path.join(base, "protnn/scripts/train_gcn.py"),
                "--config-path", config_path,
                "--ontology", ont,
                "--device", "4"
            ],
            env_bin=torch_py
        )

    # 4.2 GCN推理+TTA增强
    run_cmd(
        [os.path.join(base, "protnn/scripts/predict_gcn.py"), "--config-path", config_path, "--device", "0"],
        env_bin=torch_py
    )

    # 4.3 后处理流水线
    # 聚合所有TTA预测
    run_cmd(
        [
            os.path.join(base, "protlib/scripts/postproc/collect_ttas.py"),
            "--config-path", config_path,
            "--device", "4"
        ],
        env_bin=rapids_py
    )
    # min传播
    run_cmd(
        [
            os.path.join(base, "protlib/scripts/postproc/step.py"),
            "--config-path", config_path,
            "--device", "0",
            "--batch_size", "30000",
            "--batch_inner", "3000",
            "--lr", "0.7",
            "--direction", "min"
        ],
        env_bin=rapids_py
    )
    # max传播
    run_cmd(
        [
            os.path.join(base, "protlib/scripts/postproc/step.py"),
            "--config-path", config_path,
            "--device", "0",
            "--batch_size", "30000",
            "--batch_inner", "3000",
            "--lr", "0.7",
            "--direction", "max"
        ],
        env_bin=rapids_py
    )
    # 融合生成最终提交文件
    run_cmd(
        [
            os.path.join(base, "protlib/scripts/postproc/make_submission.py"),
            "--config-path", config_path,
            "--device", "0",
            "--max-rate", "0.5"
        ],
        env_bin=rapids_py
    )

    # 输出提交文件前10行预览
    sub_file = os.path.join(base, "sub/submission.tsv")
    check_file_exists(sub_file)
    log(f"流水线全部完成！最终提交文件路径: {sub_file}")
    log("===== 提交文件前10行预览 =====")
    with open(sub_file, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            print(line.strip())
            if idx >= 9:
                break

# ====================== 主流程入口 ======================
def main():
    try:
        # 加载全局配置
        cfg = load_config("config.yaml")

        # ============ 按需注释跳过耗时阶段 ============
        # 阶段1：环境与数据预处理
        stage_1_setup_envs(cfg)
        stage_1_preprocess_helpers(cfg)
        stage_1_external_go_data(cfg)
        stage_1_nn_prepare(cfg)

        
        ############################ 测试成功 ########################

        # 阶段2：提取T5/ESM Embedding
        stage_2_extract_embeds(cfg)

        # 阶段3：训练全部基础模型
        stage_3_train_base_models(cfg)

        # 阶段4：GCN堆叠 + 后处理生成提交
        stage_4_gcn_stack_and_submit(cfg)

        log("======== CAFA 完整流水线全部执行成功 ========")

    except KeyboardInterrupt:
        log("用户手动终止流水线")
        sys.exit(1)
    except Exception as e:
        log(f"流水线运行出现致命错误:\n{traceback.format_exc()}")
        sys.exit(2)

if __name__ == "__main__":
    main()