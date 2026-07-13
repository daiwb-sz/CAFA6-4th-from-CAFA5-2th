import glob
import os

import numpy as np
import pandas as pd

from ..metric import get_funcs_mapper


def get_tax(fasta):
    tax_list = [
        9606, 3702, 10090, 7955, 7227, 10116, 559292, 6239,
        284812, 83333, 83332, 44689, 237561, 39947, 9031, 36329,
        9913, 227321, 8355, 9823, 224308, 330879, 4577, 170187,
        9615, 99287, 85962, 243232, 287, 235443, 8364
    ]

    tax = fasta['taxonomyID'].map(
        {x: n for (n, x) in enumerate(tax_list, 1)}
    ).fillna(0).astype(np.int32).values[:, np.newaxis]
    tax = tax == np.arange(len(tax_list) + 1)[np.newaxis, :]
    tax = tax.astype(np.float32)

    return tax


def get_sergey_embeds(fasta, path, ):
    embed = np.load(path).astype(np.float32)

    dirname, basename = os.path.dirname(path), os.path.basename(path)
    id_name = os.path.join(dirname, basename.replace('_embeds', '_ids'))

    idx = np.load(id_name)

    if (len(idx) == len(fasta)) and (np.asarray(idx) == fasta['EntryID'].values).all():
        return embed

    idx = pd.Series(np.arange(idx.shape[0]), index=idx)
    idx = idx[fasta['EntryID'].values]

    return embed[idx]


def get_features_simple(fasta, embeds_list):
    fasta = pd.read_feather(fasta)
    tax = get_tax(fasta)
    embeds = np.concatenate([get_sergey_embeds(fasta, x) for x in embeds_list] + [tax], axis=1)

    return embeds, fasta['EntryID'].values


# def get_targets_from_parquet(path, ontologies, split, ids=None, names=None, fillna=False):
#     res = []

#     for i in range(3):

#         if split[i] == 0:
#             continue

#         G = ontologies[i]
#         start = sum(split[:i])
#         stop = start + split[i]

#         if ids is not None:
#             names_ = get_funcs_mapper(G, False)[ids[start: stop]].tolist()
#         elif names is not None:
#             names_ = list(names[start: stop])
#         else:
#             raise ValueError()

#         flist = sorted(glob.glob(os.path.join(path, G.namespace, 'part*')))
#         trg = pd.concat([
#             pd.read_parquet(x, columns=names_) for x in flist
#         ], ignore_index=True)  # pd.read_parquet(flist, columns=names_)

#         if fillna:
#             print('trg filled')
#             trg = trg.fillna(0)

#         res.append(trg.values)

#     return np.concatenate(res, axis=1)


def get_targets_from_parquet(path, ontologies, split, ids=None, names=None, fillna=False):
    res = []

    for i in range(3):
        if split[i] == 0:
            continue

        G = ontologies[i]
        start = sum(split[:i])
        stop = start + split[i]

        # ===== 测试1：打印切片和列名信息 =====
        print(f"\n=== 命名空间{i} ({G.namespace}) ===")
        if ids is not None:
            names_ = get_funcs_mapper(G, False)[ids[start: stop]].tolist()
            print(f"ids切片范围: [{start}:{stop}], 列名数量: {len(names_)}")
            print(f"列名前5个: {names_[:5]}")
        elif names is not None:
            names_ = list(names[start: stop])
        else:
            raise ValueError()

        # ===== 测试2：检查parquet文件 =====
        flist = sorted(glob.glob(os.path.join(path, G.namespace, 'part*')))
        print(f"找到parquet文件数: {len(flist)}")
        if len(flist) == 0:
            print(f"❌ 无文件！路径: {os.path.join(path, G.namespace)}")
            res.append(np.array([]))
            continue

        # ===== 测试3：读取文件并检查数据 =====
        try:
            trg = pd.concat([pd.read_parquet(x, columns=names_) for x in flist], ignore_index=True)
            print(f"读取后数据维度: {trg.shape}")
            print(f"NaN占比: {trg.isna().sum().sum() / trg.size:.4f}")
            print(f"正样本(1)数量: {np.sum(trg.values == 1)}")
        except Exception as e:
            print(f"❌ 读取报错: {e}")
            trg = pd.DataFrame()

        # 原有填充逻辑
        if fillna:
            print('trg filled')
            trg = trg.fillna(0)
            print(f"填充后0占比: {(trg == 0).sum().sum() / trg.size:.4f}")

        res.append(trg.values)
        print(f"添加到res的数组维度: {trg.values.shape}")

    # ===== 测试4：拼接前检查 =====
    print("\n=== 拼接前各数组维度 ===")
    for idx, arr in enumerate(res):
        print(f"数组{idx}: {arr.shape}")
    
    # 原有拼接逻辑
    return np.concatenate(res, axis=1)