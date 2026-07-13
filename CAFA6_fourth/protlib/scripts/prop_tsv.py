import argparse
import os
import sys
import time

import tqdm
import pandas as pd
import numpy as np

# 新增：替换cupy的kernel，实现numpy版本的术语传播逻辑
def propagate_col_numpy(indexer, adj, f, adj_len, mat_cols, mat_ravel):
    """
    numpy版本的GO术语向上传播核心逻辑
    功能：将子节点的数值传播到父节点（取最大值）
    """
    mat = mat_ravel.reshape(-1, mat_cols)
    # 父节点位置f，子节点位置adj
    mat[:, f] = np.maximum(mat[:, f], np.max(mat[:, adj], axis=1))
    return mat.ravel()

def propagate_max_numpy(mat, G):
    """
    numpy版本的propagate_max（替换原cupy版本）
    按GO术语的拓扑顺序向上传播
    """
    indexer = np.arange(mat.shape[0])

    for f in G.order:
        adj = G.terms_list[f]['children']

        if len(adj) == 0:
            continue

        adj = np.array(adj, dtype=np.int64)
        # 调用numpy版本的传播函数
        mat_ravel = propagate_col_numpy(
            indexer, adj, f, len(adj), mat.shape[1], mat.ravel()
        )
        mat = mat_ravel.reshape(mat.shape)

    return mat

sys.path.append(os.path.abspath(os.path.join(__file__, '../../../')))

parser = argparse.ArgumentParser()
parser.add_argument('-p', '--path', type=str)
parser.add_argument('-g', '--graph', type=str)
parser.add_argument('-o', '--output', type=str)

parser.add_argument('-d', '--device', type=str, default="1")  # 保留参数但无实际作用
parser.add_argument('-b', '--batch_size', type=int, default=30000)
parser.add_argument('-bi', '--batch_inner', type=int, default=5000)

if __name__ == '__main__':
    args = parser.parse_args()

    # 注释掉CUDA相关环境变量（CPU版本无需）
    # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    # 替换cupy/cudf为numpy/pandas
    try:
        from protlib.metric import get_funcs_mapper, get_ns_id, obo_parser, Graph
    except Exception as e:
        print(f"导入模块失败：{e}")
        sys.exit(1)

    print(f"读取输入文件：{args.path}")

    # 1. 替换cudf.read_csv为pandas.read_csv
    trainTerms = pd.read_csv(
        args.path, 
        sep='\t', 
        usecols=['EntryID', 'term'],
        dtype={'EntryID': str, 'term': str}  # 显式指定类型，避免自动转换
    )

    # 2. 解析GO本体图（逻辑不变）
    ontologies = []
    for ns, terms_dict in obo_parser(args.graph).items():
        ontologies.append(Graph(ns, terms_dict, None, True))

    # 3. 替换cudf的drop_duplicates/Series为pandas版本
    back_prot_id = trainTerms['EntryID'].drop_duplicates().reset_index(drop=True)
    length = len(back_prot_id)
    # 构建蛋白质ID到索引的映射（替换cudf.Series+cp.arange）
    prot_id = pd.Series(
        np.arange(length),  # numpy数组替换cupy数组
        index=back_prot_id.values
    )
    # 替换cudf的map为pandas的map
    trainTerms['id'] = trainTerms['EntryID'].map(prot_id)

    flg = True  # 标记是否为第一次写入（控制header）

    # 4. 分批次处理（逻辑不变，替换cudf.query为pandas条件筛选）
    for i in tqdm.tqdm(range(0, length, args.batch_size), desc="处理批次"):
        # 替换cudf.query为pandas布尔索引
        sample = trainTerms[
            (trainTerms['id'] >= i) & (trainTerms['id'] < i + args.batch_size)
        ].copy()
        batch_len = min(args.batch_size, length - i)

        for G in ontologies:
            # 构建术语到ID的映射（替换cudf.Series）
            mapper = pd.Series(get_funcs_mapper(G))
            # 替换cudf.map为pandas.map
            sample['term_id'] = sample['term'].map(mapper)
            # 替换dropna+astype为pandas版本
            sample_ont = sample.dropna().astype({'term_id': np.int32}).copy()

            # 调整id为批次内相对索引
            sample_ont['id'] = sample_ont['id'] - i

            # 5. 替换cupy数组为numpy数组，构建邻接矩阵
            mat = np.zeros((batch_len, G.idxs), dtype=np.float32)
            # 替换cp.scatter_add为numpy的索引赋值
            if not sample_ont.empty:
                rows = sample_ont['id'].values
                cols = sample_ont['term_id'].values
                mat[rows, cols] = 1.0  # 标记存在的术语
            mat = np.clip(mat, 0, 1)  # 替换cp.clip为np.clip

            # 6. 替换cupy版本的propagate_max为numpy版本
            mat = propagate_max_numpy(mat, G)

            # 7. 分小批次输出（替换cp.nonzero为np.nonzero）
            for j in range(0, mat.shape[0], args.batch_inner):
                batch_mat = mat[j: j + args.batch_inner]
                row, col = np.nonzero(batch_mat)  # numpy非零值索引

                if len(row) == 0:
                    continue  # 无数据则跳过

                # 构建输出DataFrame（替换cudf.DataFrame）
                sample_batch = pd.DataFrame({
                    'EntryID': back_prot_id[i + j + row].values,
                    'term': pd.Series(get_funcs_mapper(G, False))[col].values,
                }).sort_values(['EntryID', 'term'], ascending=True).reset_index(drop=True)

                # 添加aspect列
                ns_id, ns_str = get_ns_id(G)
                asp = ns_str.upper() + 'O'
                sample_batch['aspect'] = asp

                # 8. 写入文件（替换cudf.to_csv为pandas.to_csv）
                write_mode = 'w' if flg else 'a'
                sample_batch.to_csv(
                    args.output,
                    mode=write_mode,
                    index=False,
                    sep='\t',
                    header=flg
                )
                flg = False  # 第一次写入后关闭header

    print(f"处理完成！输出文件：{args.output}")