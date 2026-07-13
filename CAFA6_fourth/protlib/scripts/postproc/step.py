import argparse
import os
import sys
import numpy as np  # 替换cupy为numpy
import pandas as pd  # 替换cudf为pandas
import tqdm
import yaml

sys.path.append(os.path.abspath(os.path.join(__file__, '../../../../')))

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config-path', type=str)
parser.add_argument('-d', '--device', type=str, default="1")  # 保留参数但无实际作用
parser.add_argument('-b', '--batch_size', type=int, default=30000)
parser.add_argument('-bi', '--batch_inner', type=int, default=5000)
parser.add_argument('-l', '--lr', type=float, default=0.1)
parser.add_argument('-dr', '--direction', type=str, default='max')

# ========== 核心修改1：移除cupy核函数，重写传播逻辑 ==========
def propagate_max(mat, G):
    """
    纯numpy实现max传播逻辑（替代原CUDA核函数）
    """
    # 遍历拓扑排序后的节点
    for f in G.order:
        adj = G.terms_list[f]['children']
        if len(adj) == 0:
            continue
        
        # 对每一行，将当前列f的值更新为f列和所有子节点列的最大值
        mat[:, f] = np.maximum(mat[:, f], mat[:, adj].max(axis=1))

def propagate_min(mat, G):
    """
    纯numpy实现min传播逻辑（替代原CUDA核函数）
    """
    from protlib.metric import get_depths  # 延迟导入，避免提前报错
    
    D = get_depths(G, True)
    for i in range(len(D)):
        for f in D[i]:
            adj = G.terms_list[f]['adj']
            if len(adj) == 0:
                continue
            
            # 对每一行，将当前列f的值更新为f列和所有邻接节点列的最小值
            mat[:, f] = np.minimum(mat[:, f], mat[:, adj].min(axis=1))

if __name__ == '__main__':
    args = parser.parse_args()

    # 保留环境变量设置，但无实际GPU作用
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    
    # ========== 核心修改2：移除cupy/cudf导入，替换为numpy/pandas ==========
    # import cupy as cp → 移除
    # import cudf → 移除

    try:
        from protlib.metric import get_funcs_mapper, get_ns_id, obo_parser, Graph, get_depths
    except Exception:
        get_funcs_mapper, get_ns_id, obo_parser, Graph, get_depths = [None] * 5

    with open(args.config_path) as f:
        config = yaml.safe_load(f)

    # ========== 核心修改3：移除CUDA核函数初始化（已用numpy重写） ==========
    # prop_max_kernel = get_kernel('max') → 移除
    # prop_min_kernel = get_kernel('min') → 移除

    graph_path = os.path.join(config['base_path'], 'Train/go-basic.obo')
    pp_path = os.path.join(config['base_path'], config['models_path'], 'postproc')
    input_path = os.path.join(pp_path, 'pred.tsv')
    output_path = os.path.join(pp_path, f'pred_{args.direction}.tsv')

    # ========== 核心修改4：cudf.read_csv → pd.read_csv ==========
    trainTerms = pd.read_csv(input_path, sep='\t', names=['EntryID', 'term', 'prob'], header=None)
    ontologies = []
    for ns, terms_dict in obo_parser(graph_path).items():
        ontologies.append(Graph(ns, terms_dict, None, True))

    # ========== 核心修改5：cudf去重/Series操作 → pandas操作 ==========
    back_prot_id = trainTerms['EntryID'].drop_duplicates().reset_index(drop=True)
    length = len(back_prot_id)
    # 生成蛋白ID到索引的映射
    prot_id = pd.Series(np.arange(length), index=back_prot_id)
    trainTerms['id'] = trainTerms['EntryID'].map(prot_id)

    flg = True

    # 批量处理
    for i in tqdm.tqdm(range(0, length, args.batch_size)):
        # ========== 核心修改6：cudf.query → pandas.query ==========
        sample = trainTerms.query(f'(id >= {i}) & (id < {i + args.batch_size})')
        batch_len = min(args.batch_size, length - i)

        for G in ontologies:
            # 获取GO term映射表
            mapper = pd.Series(get_funcs_mapper(G))
            # ========== 核心修改7：cudf.map/dropna/astype → pandas操作 ==========
            sample['term_id'] = sample['term'].map(mapper)
            sample_ont = sample.dropna().astype({'term_id': np.int32})

            # 调整id为批次内的相对索引
            sample_ont['id'] = sample_ont['id'] - i

            # ========== 核心修改8：cupy.zeros → numpy.zeros ==========
            mat = np.zeros((batch_len, G.idxs), dtype=np.float32)
            
            # ========== 核心修改9：cupy.scatter_add → numpy.add.at ==========
            # 将prob值填充到矩阵中（替代原scatter_add）
            rows = sample_ont['id'].values
            cols = sample_ont['term_id'].values
            vals = sample_ont['prob'].values
            np.add.at(mat, (rows, cols), vals)
            
            # 裁剪值到0-1范围
            mat = np.clip(mat, 0, 1)
            mat_old = mat.copy()

            # 执行传播逻辑
            if args.direction == 'max':
                propagate_max(mat, G)
            else:
                propagate_min(mat, G)

            # 融合传播结果和原始值
            mat = mat * args.lr + mat_old * (1 - args.lr)

            # 分批次处理非零值并保存
            for j in range(0, mat.shape[0], args.batch_inner):
                mat_batch = mat[j: j + args.batch_inner]
                # ========== 核心修改10：cupy.nonzero → numpy.nonzero ==========
                row, col = np.nonzero(mat_batch)

                # 构建结果DataFrame
                # ========== 核心修改11：cudf.DataFrame → pandas.DataFrame ==========
                sample_batch = pd.DataFrame({
                    'EntryID': back_prot_id[i + j + row].reset_index(drop=True),
                    'term': pd.Series(get_funcs_mapper(G, False))[col].reset_index(drop=True),
                    'prob': mat_batch[row, col]
                }).sort_values(['EntryID', 'term'], ascending=True)

                # 截断概率值到5位字符
                sample_batch['prob'] = sample_batch['prob'].astype(str).str.slice(0, 5)

                # 获取命名空间标识
                ns_id, ns_str = get_ns_id(G)
                asp = ns_str.upper() + 'O'

                # 写入文件（首次写覆盖，后续追加）
                with open(output_path, 'w' if flg else 'a') as f:
                    sample_batch.to_csv(f, index=False, sep='\t', header=None)
                    flg = False