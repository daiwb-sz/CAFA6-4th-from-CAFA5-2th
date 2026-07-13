import numpy as np
import pandas as pd
import joblib
import glob
import os

def check_cafa5_data_detail(helpers_path, G, root_id):
    print("===== CAFA5数据详细检查（形状+内容）=====\n")
    
    # --------------------------
    # 1. 检查Parquet文件（标签数据）
    # --------------------------
    print("【1. Parquet文件（GO标注数据）】")
    flist = sorted(glob.glob(os.path.join(helpers_path, f'real_targets/{G.namespace}/part*')))
    print(f"- 文件数量：{len(flist)}")
    
    # 加载第一个文件看结构（避免加载全部，节省内存）
    first_parquet = pd.read_parquet(flist[0])
    print(f"- 单个Parquet文件形状：{first_parquet.shape}（样本数×GO术语列数）")
    print(f"- Parquet文件列名（前10个）：{first_parquet.columns[:10].tolist()}")
    print(f"- Parquet文件索引：{first_parquet.index[:5].tolist()}（前5个样本索引）")
    
    # 加载所有Parquet文件并合并
    all_parquet = [pd.read_parquet(f) for f in flist]
    target = pd.concat(all_parquet, ignore_index=True).fillna(0)
    print(f"- 合并后Parquet总形状：{target.shape}（总样本数×总GO术语列数）")
    
    # 打印Parquet前3行前5列的内容（直观看数据）
    print(f"- Parquet数据示例（前3行前5列）：")
    print(target.iloc[:3, :5])
    
    # 根节点相关信息
    if root_id in target.columns:
        print(f"- 根节点{root_id}列的值分布：{np.unique(target[root_id].values)}")
        train_sl = np.nonzero(target[root_id].values == 1)[0]
        print(f"- 根节点为1的样本数：{len(train_sl)}")
        # 打印根节点为1的前3个样本
        print(f"- 根节点为1的前3个样本（前5列）：")
        print(target.iloc[train_sl[:3], :5])
        
        # 【关键】移除根节点列，匹配prior/nulls的维度
        target = target.drop(columns=[root_id])
        print(f"- ⚠️ 已移除根节点列{root_id}，Parquet新形状：{target.shape}")
    else:
        print(f"- ❌ 根节点{root_id}不在Parquet列中！")
    
    print("\n" + "-"*60 + "\n")
    
    # --------------------------
    # 2. 检查prior.pkl（先验概率）
    # --------------------------
    print("【2. prior.pkl（先验概率数据）】")
    prior_path = os.path.join(helpers_path, f'real_targets/{G.namespace}/prior.pkl')
    if os.path.exists(prior_path):
        prior_cnd = joblib.load(prior_path)
        print(f"- prior.pkl数据类型：{type(prior_cnd)}")
        print(f"- prior.pkl形状：{prior_cnd.shape}（长度=GO术语总数）")
        print(f"- prior.pkl数值范围：{np.min(prior_cnd):.6f} ~ {np.max(prior_cnd):.6f}")
        
        # 适配一维数组打印
        if prior_cnd.ndim == 1:
            print(f"- prior.pkl数据示例（前10个GO术语的先验概率）：")
            go_terms = target.columns[:10]  # 取前10个GO术语
            for term, prob in zip(go_terms, prior_cnd[:10]):
                print(f"  {term}: {prob:.6f}")
            print(f"- prior.pkl作用：存储每个GO术语的先验概率（无特征时的注释概率），用于校正模型不平衡、初始化权重")
        else:
            print(f"- ❌ prior.pkl数组维度异常（{prior_cnd.ndim}维）")
        
        # 校验维度匹配性（移除根节点后）
        print(f"- 校验：先验概率长度 vs 移除根节点后GO术语列数 = {prior_cnd.shape[0]} vs {target.shape[1]}")
        print(f"- 维度是否匹配：{prior_cnd.shape[0] == target.shape[1]}（必须为True！）")
    else:
        print(f"- ❌ prior.pkl文件不存在！")
    
    print("\n" + "-"*60 + "\n")
    
    # --------------------------
    # 3. 检查nulls.pkl（掩码文件）
    # --------------------------
    print("【3. nulls.pkl（加权掩码文件）】")
    nulls_path = os.path.join(helpers_path, f'real_targets/{G.namespace}/nulls.pkl')
    if os.path.exists(nulls_path):
        nulls = joblib.load(nulls_path)
        print(f"- nulls.pkl数据类型：{type(nulls)}")
        print(f"- nulls.pkl形状：{nulls.shape}（长度=GO术语总数）")
        
        # 详细数值分析
        print(f"- nulls.pkl数值范围：{np.min(nulls):.6f} ~ {np.max(nulls):.6f}")
        print(f"- nulls.pkl均值：{np.mean(nulls):.6f}（越接近1说明数据可信度越高）")
        print(f"- nulls.pkl中位数：{np.median(nulls):.6f}")
        
        # 统计不同区间的数值占比（核心分析）
        high_conf = np.sum(nulls >= 0.99) / len(nulls) * 100
        mid_conf = np.sum((nulls >= 0.94) & (nulls < 0.99)) / len(nulls) * 100
        low_conf = np.sum(nulls < 0.94) / len(nulls) * 100
        print(f"- nulls.pkl数值分布占比：")
        print(f"  ✅ 高可信度（≥0.99）：{high_conf:.2f}%")
        print(f"  ⚠️ 中可信度（0.94~0.99）：{mid_conf:.2f}%")
        print(f"  ❌ 低可信度（<0.94）：{low_conf:.2f}%")
        
        # 打印示例（带含义）
        print(f"- nulls.pkl数据示例（前10个GO术语的掩码值）：")
        go_terms = target.columns[:10]
        for term, val in zip(go_terms, nulls[:10]):
            conf = "高可信度" if val >= 0.99 else "中可信度" if val >=0.94 else "低可信度"
            print(f"  {term}: {val:.6f}（{conf}）")
        
        # 核心作用说明
        print(f"- nulls.pkl核心作用：")
        print(f"  1. 标记GO术语注释的可信度（0=完全缺失，1=100%可信）；")
        print(f"  2. 模型训练时作为损失权重，高可信度注释权重更高；")
        print(f"  3. 区分'真实0注释'和'缺失注释'，避免模型误判。")
        
        # 维度校验
        print(f"- 校验：nulls长度 vs 移除根节点后GO术语列数 = {nulls.shape[0]} vs {target.shape[1]}")
        print(f"- 维度是否匹配：{nulls.shape[0] == target.shape[1]}（必须为True！）")
    else:
        print(f"- ❌ nulls.pkl文件不存在！")

# --------------------------
# 调用代码
# --------------------------
if __name__ == "__main__":
    helpers_path = "/home/dwb/workspace/kaggle/CAFA5-protein-function-prediction-2nd-place-main/helpers"
    
    # 替换为你的真实G对象
    class MockGO:
        def __init__(self):
            self.namespace = "biological_process"  # 替换为你的本体（bp/mf/cc）
            self.terms_list = [{'id': 'GO:0008150'}]
    
    G = MockGO()
    root_id = "GO:0008150"
    
    check_cafa5_data_detail(helpers_path, G, root_id)