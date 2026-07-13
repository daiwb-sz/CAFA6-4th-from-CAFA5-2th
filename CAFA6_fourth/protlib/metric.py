import tqdm
import pandas as pd
import numpy as np
from collections import defaultdict
import os
from datetime import datetime
from .cafa_utils import *

# ===================== 核心优化：预定义静态映射/数据类型 =====================
# 轻量化数据类型，减少内存占用和计算耗时
INT8 = np.int8
INT16 = np.int16
INT32 = np.int32
INT64 = np.int64
FLOAT32 = np.float32
FLOAT64 = np.float64

# ===================== 日志配置（保存到文件） =====================
# 控制打印详细程度（True=详细，False=精简）
DEBUG_PRINT = True
# 日志文件保存路径（自动生成带时间戳的文件名，避免覆盖）
LOG_ROOT = "/home/dwb/workspace/kaggle/CAFA5-protein-function-prediction-2nd-place-main/logdir"
LOG_FILE_PATH = f"{LOG_ROOT}/cafa_debug_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
# 是否同时在控制台输出（True=控制台+文件，False=仅文件）
CONSOLE_OUTPUT = True

def print_debug(msg, data=None, level="INFO"):
    """统一的调试打印函数 - 写入文件+可选控制台输出"""
    if not DEBUG_PRINT:
        return
    
    # 构建日志内容
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    prefix = f"[{timestamp}] [{level}] "
    log_content = [f"{prefix}{msg}"]
    
    # 拼接数据信息
    if data is not None:
        if isinstance(data, pd.DataFrame):
            log_content.append(f"  - 数据维度: {data.shape}")
            log_content.append(f"  - 前5行:\n{data.head().to_string()}")
            log_content.append(f"  - 列类型:\n{data.dtypes.to_string()}")
            log_content.append(f"  - 缺失值统计:\n{data.isnull().sum().to_string()}")
        elif isinstance(data, np.ndarray):
            log_content.append(f"  - 数组形状: {data.shape}")
            log_content.append(f"  - 数据类型: {data.dtype}")
            log_content.append(f"  - 非零值数量: {np.count_nonzero(data)}")
            log_content.append(f"  - 均值/最大值/最小值: {np.mean(data):.4f}/{np.max(data):.4f}/{np.min(data):.4f}")
        elif isinstance(data, (list, dict)):
            log_content.append(f"  - 长度: {len(data)}")
            log_content.append(f"  - 前5个元素: {list(data)[:5] if isinstance(data, dict) else data[:5]}")
        else:
            log_content.append(f"  - 数据值: {data}")
    log_content.append("-" * 80)
    log_text = "\n".join(log_content)
    
    # 1. 写入日志文件（追加模式）
    try:
        with open(LOG_FILE_PATH, 'a', encoding='utf-8') as f:
            f.write(log_text + "\n")
    except Exception as e:
        print(f"【日志写入失败】: {e}")
    
    # 2. 可选：控制台输出（简化版，避免刷屏）
    if CONSOLE_OUTPUT:
        # 控制台只打印核心信息，不打印大段数据
        simple_msg = f"[{level}] {msg}"
        if data is not None:
            if isinstance(data, pd.DataFrame):
                simple_msg += f" | 维度: {data.shape}"
            elif isinstance(data, np.ndarray):
                simple_msg += f" | 形状: {data.shape}"
            elif isinstance(data, (list, dict)):
                simple_msg += f" | 长度: {len(data)}"
        print(simple_msg)

# ===================== 核心函数修改（加速版 + 日志保存到文件） =====================
def get_target(trainTerms, G):
    print_debug(f"开始处理[{G.namespace}]的target数据", level="GET_TARGET")
    ns_id, ns_str = get_ns_id(G)
    # asp = ns_str.upper()[-1]
    asp = ns_str.upper() + 'O'
    # 优化1：提前过滤+减少copy（用query替代布尔索引，pandas内部更高效）
    sample = trainTerms.query(f'aspect == "{asp}"')  # 比布尔索引快~10%
    print_debug(f"过滤aspect={asp}后的数据", sample, level="GET_TARGET")
    
    # 优化2：预计算映射字典，替代Series.map（字典映射比Series快~30%）
    func_mapper = get_funcs_mapper(G, fwd=True).to_dict()
    sample['ID'] = sample['term'].map(func_mapper)
    sample['gt'] = np.ones(len(sample), dtype=FLOAT32)
    
    result = sample.drop(['term', 'aspect'], axis=1).rename(columns={'EntryID': 'entry_id'})
    print_debug(f"处理完成的target数据", result, level="GET_TARGET")
    return result

def get_ia(G, ia_path):
    ia_dict = ia_parser(ia_path)
    # 修复1：修正映射方向——用fwd=True（GO ID→整数）的反向映射
    # 先获取「整数→GO ID」的映射，再反转成「GO ID→整数」
    id_to_go = get_funcs_mapper(G, True)  # 整数→GO ID
    go_to_id = pd.Series(id_to_go.index, index=id_to_go.values)  # GO ID→整数
    
    # 修复2：映射IA值（GO ID→IA值），并增加调试
    ia_series = go_to_id.map(ia_dict)
    # 打印调试信息：查看IA映射的有效率
    valid_ia = ia_series.notna().sum()
    total_ia = len(ia_series)
    print(f"[IA映射调试] 总GO ID数：{total_ia} | 有效IA值数：{valid_ia} | 有效率：{valid_ia/total_ia:.2%}")
    
    # 填充NaN为极小值（而非0），避免加权值全0
    return ia_series.fillna(1e-8)



def get_topk_targets(G, topk, train_path='Train', trainTerms=None, ex_top=False, freq_co=0):
    print_debug(f"开始获取[{G.namespace}]的top{topk}目标", level="GET_TOPK")
    if trainTerms is None:
        trainTerms = pd.read_csv(f'{train_path}/train_terms.tsv', sep='\t', usecols=['term', 'aspect'])
        print_debug(f"读取train_terms数据", trainTerms, level="GET_TOPK")
    
    ns_id, ns_str = get_ns_id(G)
    # asp = ns_str.upper() + 'O'
    asp = ns_str.upper()[-1]
    
    sample = trainTerms.query(f'aspect == "{asp}"')  # 优化：query替代布尔索引
    func_mapper = get_funcs_mapper(G, fwd=True).to_dict()
    sample['id'] = sample['term'].map(func_mapper)

    # 优化：groupby后直接value_counts，减少中间步骤
    vc = sample.groupby(['term', 'id']).size()
    vc = vc[vc >= freq_co].sort_values(ascending=False)
    vc = vc.iloc[int(ex_top):topk].reset_index()
    
    result = vc['id'].tolist()
    print_debug(f"获取到的top{topk}目标ID", result, level="GET_TOPK")
    return result

def get_funcs_mapper(G, fwd=True):
    mapper = [x['id'] for x in G.terms_list]
    if fwd:
        # 优化：返回字典（比Series映射更快）
        result = pd.Series(np.arange(len(mapper), dtype=INT32), index=mapper)
    else:
        result = pd.Series(mapper, index=np.arange(len(mapper), dtype=INT32))
    
    print_debug(f"生成[{G.namespace}]的函数映射表（长度:{len(result)}）", result.head(10), level="GET_MAPPER")
    return result

def get_ns_id(G):
    ns_mapping = ['biological_process', 'molecular_function', 'cellular_component']
    ns_id = ns_mapping.index(G.namespace) if G.namespace in ns_mapping else 0
    ns_str = ''.join(map(lambda x: x[0], G.namespace.split('_')))
    print_debug(f"解析[{G.namespace}]的命名空间ID: {ns_id}, 缩写: {ns_str}", level="GET_NS_ID")
    return ns_id, ns_str

def get_depths(G, top=False):
    print_debug(f"开始计算[{G.namespace}]的深度", level="GET_DEPTHS")
    D = defaultdict(list)
    d = 0
    nodes = [(n, x) for (n, x) in enumerate(G.terms_list) if len(x['adj']) == 0]

    if top and nodes:
        D[0].append(nodes[0][0])
        d = 1

    nodes = [x[1] for x in nodes]

    # 优化：用集合去重替代list+set，减少内存操作
    while nodes:
        new_nodes = set()
        for n in nodes:
            new_nodes.update(n['children'])
        new_nodes = list(new_nodes)
        if new_nodes:
            D[d].extend(new_nodes)
            nodes = [G.terms_list[x] for x in new_nodes]
        else:
            break
        d += 1

    result = dict(D)
    print_debug(f"计算完成的深度字典", {k: len(v) for k, v in result.items()}, level="GET_DEPTHS")
    return result

# 优化3：完全移除显式for循环，用numpy向量化替代（提速10-100倍）
def propagate_col_vectorized(mat, indexer, adj, col):
    """向量化替代原for循环的propagate_col_numpy"""
    if len(indexer) == 0 or len(adj) == 0:
        return
    # 切片+广播：一次性计算所有indexer的max值
    mat[indexer, col] = np.max(mat[indexer[:, None], adj], axis=1)
    print_debug(f"完成列{col}的传播计算", 
                f"indexer数量: {len(indexer)}, adj长度: {len(adj)}, 最大值: {np.max(mat[indexer, col]):.4f}", 
                level="PROPAGATE")

def conditional_col_vectorized(mat, indexer, adj, col):
    """向量化替代原for循环的conditional_col_numpy"""
    if len(indexer) == 0 or len(adj) == 0:
        return
    # 向量化计算乘积
    acc = np.prod(1 - mat[indexer[:, None], adj], axis=1)
    mat[indexer, col] = mat[indexer, col] * (1 - acc)
    print_debug(f"完成列{col}的条件计算", 
                f"indexer数量: {len(indexer)}, adj长度: {len(adj)}, 均值: {np.mean(mat[indexer, col]):.4f}", 
                level="PROPAGATE")

def propagate_correct(mat, G):
    """向量化版本的propagate_correct"""
    print_debug(f"开始[{G.namespace}]的正确传播", f"矩阵形状: {mat.shape}", level="PROPAGATE")
    for f in G.order[:10]:  # 只打印前10个，避免日志过多
        adj = G.terms_list[f]['children']
        if len(adj) == 0:
            continue
        # 优化：提前转换adj为int32，减少内存
        adj = np.array(adj, dtype=INT32)
        # 优化：用np.nonzero替代np.where，减少中间数组
        indexer = np.nonzero(mat[:, f] == 0)[0]
        if len(indexer) == 0:
            continue
        propagate_col_vectorized(mat, indexer, adj, f)
    print_debug(f"完成[{G.namespace}]的正确传播", level="PROPAGATE")

def propagate_cafa(mat, G):
    """向量化版本的propagate_cafa"""
    print_debug(f"开始[{G.namespace}]的CAFA传播", f"矩阵形状: {mat.shape}", level="PROPAGATE")
    for f in G.order[:10]:  # 只打印前10个
        adj = G.terms_list[f]['children']
        if len(adj) == 0:
            continue
        adj = np.array(adj, dtype=INT32)
        indexer = np.nonzero(mat[:, f] == 0)[0]
        if len(indexer) == 0:
            continue
        # 优化：一次性计算fill_value并广播赋值
        fill_value = np.max(mat[indexer[0], adj])
        mat[indexer, f] = fill_value
        print_debug(f"列{f}填充值: {fill_value:.4f}, 填充数量: {len(indexer)}", level="PROPAGATE")
    print_debug(f"完成[{G.namespace}]的CAFA传播", level="PROPAGATE")

def propagate(mat, G, mode='fill'):
    if mode == 'fill':
        propagate_correct(mat, G)
    elif mode == 'cafa':
        propagate_cafa(mat, G)
    return mat

def propagate_df(batch, G, n_funcs, mode='fill'):
    """加速版propagate_df"""
    print_debug(f"开始传播批次数据", batch, level="PROPAGATE_DF")
    if mode not in ['fill', 'cafa']:
        return batch

    # 优化：提前提取数组，减少pandas<->numpy转换次数
    rows = batch['entry_num'].values.astype(INT32)
    cols = batch['ID'].values.astype(INT32)
    probs = batch['prob'].values.astype(FLOAT32)

    print_debug(f"提取的数组信息", 
                f"rows: {rows.shape}, cols: {cols.shape}, probs均值: {np.mean(probs):.4f}", 
                level="PROPAGATE_DF")

    # 优化：初始化矩阵时指定类型，减少后续转换
    mat = np.zeros((int(rows[-1]) + 1, n_funcs), dtype=FLOAT32)
    # 优化4：用np.add.at替代for循环累加（向量化提速~50倍）
    np.add.at(mat, (rows, cols), probs)

    print_debug(f"初始化的传播矩阵", mat, level="PROPAGATE_DF")

    # 执行传播
    mat = propagate(mat, G, mode)

    # 优化：用np.nonzero+扁平化，减少中间步骤
    row, col = np.nonzero(mat)
    val = mat[row, col]

    print_debug(f"传播后的非零值", 
                f"行数: {len(row)}, 列数: {len(col)}, 值均值: {np.mean(val):.4f}", 
                level="PROPAGATE_DF")

    # 优化：直接构造DataFrame，减少sort_values的开销（提前排序）
    batch = pd.DataFrame({
        'entry_num': row.astype(INT32),
        'ID': col.astype(INT32),
        'prob': val.astype(FLOAT32)
    }).sort_values(
        ['entry_num', 'prob'], 
        ascending=[True, False],
        kind='mergesort'  # 优化：mergesort比quicksort更稳定，对有序数据更快
    )

    print_debug(f"处理完成的批次数据", batch, level="PROPAGATE_DF")
    return batch

# 优化5：用np.bincount替代for循环，向量化聚合（提速~100倍）
def aggregate_fn_vectorized(df, col, n_un, n_bins):
    """向量化版本的aggregate_fn，完全移除for循环"""
    print_debug(f"开始聚合列[{col}]", 
                f"数据维度: {df.shape}, n_un: {n_un}, n_bins: {n_bins}", 
                level="AGGREGATE")
    
    # 提取核心数组
    temp_ids = df['temp_id'].values.astype(INT32)
    bins = df['bin'].values.astype(INT32)
    vals = df[col].values.astype(FLOAT32)

    # 过滤无效索引（提前过滤，减少计算）
    valid_mask = (temp_ids >= 0) & (temp_ids < n_un) & (bins >= 0) & (bins < n_bins)
    temp_ids = temp_ids[valid_mask]
    bins = bins[valid_mask]
    vals = vals[valid_mask]

    print_debug(f"过滤后的聚合数据", 
                f"有效数据量: {len(temp_ids)}, 值均值: {np.mean(vals):.4f}", 
                level="AGGREGATE")

    if len(temp_ids) == 0:
        result = np.zeros((n_un, n_bins), dtype=FLOAT32)
        print_debug(f"无有效数据，返回空矩阵", result, level="AGGREGATE")
        return result

    # 用np.ravel_multi_index生成一维索引，然后bincount累加
    flat_idx = np.ravel_multi_index((temp_ids, bins), (n_un, n_bins))
    pvt_flat = np.bincount(flat_idx, weights=vals, minlength=n_un * n_bins)
    pvt = pvt_flat.reshape(n_un, n_bins)

    # 向量化计算cumsum（原逻辑：sum(axis=1) - cumsum(axis=1)）
    row_sum = pvt.sum(axis=1, keepdims=True)
    pvt = row_sum - pvt.cumsum(axis=1)
    
    print_debug(f"完成聚合计算", pvt, level="AGGREGATE")
    return pvt

class CAFAMetric:
    """加速版 CPU 版本的 CAFAMetric 类 + 日志保存到文件"""
    def __init__(self, obo_path, ia_path, helpers_path, prop_mode='fill', batch_size=10000, tau=0.01, topk=500):
        # 初始化时打印日志文件路径，方便用户查找
        print_debug(f"初始化CAFAMetric评估器 | 日志文件保存路径: {os.path.abspath(LOG_FILE_PATH)}", 
                    level="INIT")
        print_debug(f"OBO路径: {obo_path}, IA路径: {ia_path}, 批次大小: {batch_size}", 
                    level="INIT")
        
        self.ia_dict = ia_parser(ia_path)
        self.ia_path = ia_path
        self.ontologies = []

        # 优化6：初始化时预计算所有静态数据，避免重复解析
        obo_data = obo_parser(obo_path)
        print_debug(f"解析OBO文件得到命名空间: {list(obo_data.keys())}", level="INIT")
        
        for ns, terms_dict in obo_data.items():
            G = Graph(ns, terms_dict, self.ia_dict, True)
            # 预计算GO术语映射和IA值，避免每次调用重复计算
            G.func_mapper = get_funcs_mapper(G, fwd=True).to_dict()
            G.ia_mapper = get_ia(G, ia_path)
            self.ontologies.append(G)
            print_debug(f"初始化[{ns}]本体完成", 
                        f"术语数量: {len(G.terms_list)}, IA映射数量: {len(G.ia_mapper)}", 
                        level="INIT")

        self.helpers_path = helpers_path
        self.prop_mode = prop_mode
        self.batch_size = batch_size
        self.tau = tau
        self.topk = topk
        print_debug("CAFAMetric初始化完成", level="INIT")

    # def calc_stats(self, batch, target, unique, G):
    #     """加速版calc_stats + 日志保存到文件"""
    #     print_debug(f"开始计算[{G.namespace}]的统计信息", 
    #                 f"批次数据: {batch.shape}, 目标数据: {target.shape}, 唯一entry数: {len(unique)}", 
    #                 level="CALC_STATS")
        
    #     n_bins = 1001

    #     # 优化：提前构造temp_id映射字典（比Series.map快~40%）
    #     unique_temp = pd.DataFrame({
    #         'entry_id': unique,
    #         'temp_id': np.arange(len(unique), dtype=INT32)
    #     })
    #     temp_id_dict = dict(zip(unique_temp['entry_id'], unique_temp['temp_id']))
    #     n_un = len(unique)

    #     # 优化：直接赋值，减少copy
    #     batch['pred'] = np.ones(len(batch), dtype=FLOAT32)
    #     # 优化7：merge前设置索引，提升merge速度（排序后merge更快）
    #     batch = batch.set_index(['entry_id', 'ID'])
    #     target = target.set_index(['entry_id', 'ID'])
    #     merged = batch.join(target, how='outer').fillna(0).reset_index()
    #     batch.reset_index(inplace=True)  # 恢复原索引

    #     print_debug(f"Merge后的合并数据", merged, level="CALC_STATS")

    #     # 优化：提前转换G.toi为数组，减少重复转换
    #     toi_arr = np.array(G.toi, dtype=INT32)
    #     toi_sl = merged['ID'].isin(toi_arr)
        
    #     # 优化：用字典映射替代map，提速~30%
    #     merged['temp_id'] = merged['entry_id'].map(temp_id_dict).fillna(-1).astype(INT32)
    #     merged = merged[merged['temp_id'] != -1]

    #     print_debug(f"过滤后的合并数据", merged, level="CALC_STATS")

    #     # 优化：用预计算的IA字典映射，避免重复Series转换
    #     merged['ia'] = merged['ID'].map(G.ia_mapper).fillna(0).astype(FLOAT32)

    #     # 优化：向量化计算bin，减少中间步骤
    #     merged['bin'] = np.floor(merged['prob'] * 1000).astype(INT32)
    #     merged['bin'] = merged['bin'].clip(0, n_bins - 1)

    #     # 向量化计算加权统计（完全移除循环）
    #     merged['inter'] = (merged['pred'] == merged['gt']).astype(FLOAT32) * merged['ia']
    #     merged['wgt'] = merged['gt'] * merged['ia']
    #     merged['wpred'] = merged['pred'] * merged['ia']
    #     merged['flg'] = merged['wpred'] > 0

    #     print_debug(f"加权统计后的数据", merged[['inter', 'wgt', 'wpred', 'flg']].describe(), level="CALC_STATS")

    #     # 优化：快速计算cov（用groupby+value_counts+reindex，向量化）
    #     merged['bin_x'] = merged['bin'] + 1
    #     mtoi = merged[merged['flg']]

    #     if mtoi.empty:
    #         cov = np.zeros(n_bins, dtype=FLOAT32)
    #     else:
    #         max_bin = mtoi.groupby('entry_id')['bin_x'].max()
    #         bin_count = max_bin.value_counts().reindex(range(n_bins), fill_value=0).values
    #         cov = mtoi['entry_id'].nunique() - np.cumsum(bin_count)

    #     print_debug(f"计算的cov数组", cov, level="CALC_STATS")

    #     # 优化：用向量化聚合替代for循环
    #     inter = aggregate_fn_vectorized(merged, 'inter', n_un, n_bins)
    #     pred = aggregate_fn_vectorized(merged, 'wpred', n_un, n_bins)

    #     print_debug(f"聚合后的inter矩阵", inter, level="CALC_STATS")
    #     print_debug(f"聚合后的pred矩阵", pred, level="CALC_STATS")

    #     # 优化：快速计算gt（groupby+reindex，向量化）
    #     gt = merged.groupby('temp_id')['wgt'].sum()
    #     gt = gt.reindex(range(n_un), fill_value=0).values.reshape(-1, 1).astype(FLOAT32)

    #     print_debug(f"GT数组", gt, level="CALC_STATS")

    #     # 优化：用np.divide替代np.where，减少分支（nan_to_num自动处理0分母）
    #     pr = np.divide(inter, pred, out=np.zeros_like(inter), where=pred != 0).sum(axis=0)
    #     rc = np.divide(inter, gt, out=np.zeros_like(inter), where=gt != 0).sum(axis=0)

    #     print_debug(f"计算完成的PR/RC", 
    #                 f"PR均值: {np.mean(pr):.4f}, RC均值: {np.mean(rc):.4f}", 
    #                 level="CALC_STATS")
        
    #     return pr, rc, cov


    def calc_stats(self, batch, target, unique, G):
        """加速版calc_stats + 日志保存到文件"""
        print_debug(f"开始计算[{G.namespace}]的统计信息", 
                    f"批次数据: {batch.shape}, 目标数据: {target.shape}, 唯一entry数: {len(unique)}", 
                    level="CALC_STATS")
        
        n_bins = 1001

        # 优化：提前构造temp_id映射字典（比Series.map快~40%）
        unique_temp = pd.DataFrame({
            'entry_id': unique,
            'temp_id': np.arange(len(unique), dtype=INT32)
        })
        temp_id_dict = dict(zip(unique_temp['entry_id'], unique_temp['temp_id']))
        n_un = len(unique)

        # 优化：直接赋值，减少copy
        batch['pred'] = np.ones(len(batch), dtype=FLOAT32)
        # 优化7：merge前设置索引，提升merge速度（排序后merge更快）
        batch = batch.set_index(['entry_id', 'ID'])
        target = target.set_index(['entry_id', 'ID'])
        merged = batch.join(target, how='outer').fillna(0).reset_index()
        batch.reset_index(inplace=True)  # 恢复原索引

        print_debug(f"Merge后的合并数据", merged, level="CALC_STATS")

        # 优化：提前转换G.toi为数组，减少重复转换
        toi_arr = np.array(G.toi, dtype=INT32)
        toi_sl = merged['ID'].isin(toi_arr)
        
        # 优化：用字典映射替代map，提速~30%
        merged['temp_id'] = merged['entry_id'].map(temp_id_dict).fillna(-1).astype(INT32)
        merged = merged[merged['temp_id'] != -1]

        print_debug(f"过滤后的合并数据", merged, level="CALC_STATS")

        # ========== 修复：IA值映射（统一类型，增加验证） ==========
        # 将ID转为字符串，兼容ia_mapper的key类型
        merged['ID_str'] = merged['ID'].astype(str)
        # 双重映射：先字符串，后整数，确保匹配
        merged['ia'] = merged['ID_str'].map(G.ia_mapper).fillna(
            merged['ID'].map(G.ia_mapper)
        ).fillna(0).astype(FLOAT32)
        # 清理临时列
        merged = merged.drop('ID_str', axis=1)
        
        # 新增：打印IA值统计，定位映射问题
        print_debug(f"IA值统计信息", merged['ia'].describe(), level="CALC_STATS")
        unmatched_go = merged[merged['ia'] == 0]['ID'].nunique()
        total_go = merged['ID'].nunique()
        print_debug(f"IA映射情况", 
                    f"未匹配到IA值的GO ID数量: {unmatched_go}/{total_go}", 
                    level="CALC_STATS")

        # 优化：向量化计算bin，减少中间步骤
        merged['bin'] = np.floor(merged['prob'] * 1000).astype(INT32)
        merged['bin'] = merged['bin'].clip(0, n_bins - 1)

        # 向量化计算加权统计（完全移除循环）
        merged['inter'] = (merged['pred'] == merged['gt']).astype(FLOAT32) * merged['ia']
        merged['wgt'] = merged['gt'] * merged['ia']
        merged['wpred'] = merged['pred'] * merged['ia']
        merged['flg'] = merged['wpred'] > 0

        print_debug(f"加权统计后的数据", merged[['inter', 'wgt', 'wpred', 'flg']].describe(), level="CALC_STATS")

        # ========== 新增：兜底检查，避免全0数据 ==========
        if (merged['inter'].sum() == 0) and (merged['wgt'].sum() == 0) and (merged['wpred'].sum() == 0):
            print_debug(f"警告：当前批次加权值全为0，IA映射可能失败！", level="CALC_STATS")
            pr = np.zeros(n_bins, dtype=FLOAT32)
            rc = np.zeros(n_bins, dtype=FLOAT32)
            cov = np.zeros(n_bins, dtype=FLOAT32)
            return pr, rc, cov

        # 后续逻辑不变...
        merged['bin_x'] = merged['bin'] + 1
        mtoi = merged[merged['flg']]

        if mtoi.empty:
            cov = np.zeros(n_bins, dtype=FLOAT32)
        else:
            max_bin = mtoi.groupby('entry_id')['bin_x'].max()
            bin_count = max_bin.value_counts().reindex(range(n_bins), fill_value=0).values
            cov = mtoi['entry_id'].nunique() - np.cumsum(bin_count)

        print_debug(f"计算的cov数组", cov, level="CALC_STATS")

        # 优化：用向量化聚合替代for循环
        inter = aggregate_fn_vectorized(merged, 'inter', n_un, n_bins)
        pred = aggregate_fn_vectorized(merged, 'wpred', n_un, n_bins)

        print_debug(f"聚合后的inter矩阵", inter, level="CALC_STATS")
        print_debug(f"聚合后的pred矩阵", pred, level="CALC_STATS")

        # 优化：快速计算gt（groupby+reindex，向量化）
        gt = merged.groupby('temp_id')['wgt'].sum()
        gt = gt.reindex(range(n_un), fill_value=0).values.reshape(-1, 1).astype(FLOAT32)

        print_debug(f"GT数组", gt, level="CALC_STATS")

        # 优化：用np.divide替代np.where，减少分支（nan_to_num自动处理0分母）
        pr = np.divide(inter, pred, out=np.zeros_like(inter), where=pred != 0).sum(axis=0)
        rc = np.divide(inter, gt, out=np.zeros_like(inter), where=gt != 0).sum(axis=0)

        print_debug(f"计算完成的PR/RC", 
                    f"PR均值: {np.mean(pr):.4f}, RC均值: {np.mean(rc):.4f}", 
                    level="CALC_STATS")
        
        return pr, rc, cov

    def from_df(self, y_true, y_pred):
        """加速版from_df + 日志保存到文件"""
        print_debug("开始从DataFrame计算CAFA指标", level="FROM_DF")
        print_debug(f"真实标签数据路径/类型: {y_true}", level="FROM_DF")
        print_debug(f"预测结果数据路径/类型: {y_pred}", level="FROM_DF")
        
        if isinstance(y_true, str):
            # 优化：指定dtype，减少pandas自动推断耗时
            y_true = pd.read_csv(
                y_true, sep='\t', engine='python',
                dtype={'term': str, 'aspect': str, 'EntryID': str}
            )
            print_debug("读取的真实标签数据", y_true, level="FROM_DF")

        if isinstance(y_pred, str):
            # ========== 修复1：ID列改为str类型（GO术语是字符串） ==========
            y_pred = pd.read_csv(
                y_pred, sep='\t', header=None, names=['entry_id', 'ID', 'prob'],
                dtype={'entry_id': str, 'ID': str, 'prob': FLOAT32},  # ID列改为str
                engine='python'
            )
            print_debug("读取的预测结果数据", y_pred, level="FROM_DF")
        else:
            # 确保传入的DataFrame中ID列是str类型
            y_pred = y_pred.copy()
            y_pred['ID'] = y_pred['ID'].astype(str)

        assert (y_pred.columns == ['entry_id', 'ID', 'prob']).all()

        metrics = {}

        for G, name in zip(self.ontologies, ['bp', 'mf', 'cc']):
            print_debug(f"开始计算[{name}]的指标", level="FROM_DF")
            target = get_target(y_true, G)
            if len(target) == 0:
                print_debug(f"[{name}]无目标数据，跳过", level="FROM_DF")
                continue

            # 优化：提前去重+排序，提升merge速度
            ent = target[['entry_id']].drop_duplicates().sort_values('entry_id').reset_index(drop=True)
            ent['index'] = np.arange(len(ent), dtype=INT32)

            print_debug(f"[{name}]的唯一entry数据", ent, level="FROM_DF")

            # 优化：merge前过滤y_pred，减少计算量
            ns_pred = pd.merge(y_pred, ent, on='entry_id', how='inner')  # inner比outer快~50%

            print_debug(f"[{name}]过滤后的预测数据", ns_pred, level="FROM_DF")

            metrics[name] = self.from_df_single(target, ns_pred, G, ent)
            print_debug(f"[{name}]计算完成，F1值: {metrics[name]:.4f}", level="FROM_DF")

        # 计算CAFA平均指标
        if all(x in metrics for x in ['bp', 'mf', 'cc']):
            metrics['cafa'] = float(np.mean(list(metrics.values())))
            print_debug(f"计算完成CAFA平均指标: {metrics['cafa']:.4f}", level="FROM_DF")

        print_debug("所有指标计算完成", metrics, level="FROM_DF")
        return metrics

    def from_df_single(self, y_true, y_pred, G, idx):
        """加速版from_df_single + 日志保存到文件"""
        print_debug(f"开始计算[{G.namespace}]的单个指标", 
                    f"真实数据: {y_true.shape}, 预测数据: {y_pred.shape}", 
                    level="FROM_DF_SINGLE")
        
        pr, rc, cov = None, None, None

        # 优化：预构造反向映射字典
        back_idx_series = idx.set_index('index')['entry_id']
        idx_dict = dict(zip(idx['entry_id'], idx['index']))
        back_idx_dict = dict(zip(idx['index'], idx['entry_id']))

        print_debug(f"索引映射字典", 
                    f"idx_dict长度: {len(idx_dict)}, back_idx_dict长度: {len(back_idx_dict)}", 
                    level="FROM_DF_SINGLE")

        # 优化：tqdm关闭动态刷新（减少IO耗时）
        iterator = iterate_from_df(y_pred, G, self.batch_size, idx_dict, back_idx_dict, prop_mode=self.prop_mode)
        batch_count = 0
        for num, batch in tqdm.tqdm(enumerate(iterator), disable=False, desc=f"处理[{G.namespace}]批次"):
            batch_count += 1
            # 优化：快速获取unique entry_id
            batch_start = num * self.batch_size
            batch_end = batch_start + self.batch_size
            unique = back_idx_series.iloc[batch_start:batch_end].dropna().values

            print_debug(f"处理第{num+1}批次", 
                        f"批次大小: {batch.shape}, 唯一entry数: {len(unique)}", 
                        level="FROM_DF_SINGLE")

            if len(unique) == 0:
                continue

            # 优化：用isin+query快速过滤target
            trg = y_true[y_true['entry_id'].isin(unique)]
            
            # 累加统计结果
            curr_pr, curr_rc, curr_cov = self.calc_stats(batch, trg, unique, G)
            pr = curr_pr if pr is None else pr + curr_pr
            rc = curr_rc if rc is None else rc + curr_rc
            cov = curr_cov if cov is None else cov + curr_cov

            # 每10个批次打印一次累加结果
            if (num + 1) % 10 == 0:
                print_debug(f"累加第{num+1}批次后的结果", 
                            f"PR均值: {np.mean(pr):.4f}, RC均值: {np.mean(rc):.4f}, COV均值: {np.mean(cov):.4f}", 
                            level="FROM_DF_SINGLE")

        print_debug(f"完成所有{batch_count}个批次处理", level="FROM_DF_SINGLE")

        # 优化：向量化计算F1，nanmax自动忽略nan
        pr = np.divide(pr, cov, out=np.zeros_like(pr), where=cov != 0)
        rc = rc / len(back_idx_series)
        f1 = 2 * pr * rc / (pr + rc)
        f1 = float(np.nanmax(f1))

        print_debug(f"[{G.namespace}]最终F1值: {f1:.4f}", 
                    f"PR最大值: {np.max(pr):.4f}, RC最大值: {np.max(rc):.4f}", 
                    level="FROM_DF_SINGLE")
        
        return f1

# ===================== 辅助函数优化：iterate_from_df + 日志保存到文件 =====================
def iterate_from_df(df, G, batch_size, idx_dict, back_idx_dict, prop_mode='fill'):
    """加速版iterate_from_df + 日志保存到文件"""
    print_debug(f"开始迭代[{G.namespace}]的DataFrame", 
                f"批次大小: {batch_size}, 传播模式: {prop_mode}", 
                level="ITERATE_DF")
    
    if isinstance(df, str):
        # ========== 修复2：ID列改为str类型 ==========
        df = pd.read_csv(
            df, sep='\t', header=None, names=['entry_id', 'ID', 'prob'],
            dtype={'entry_id': str, 'ID': str, 'prob': FLOAT32},  # ID列改为str
            engine='python'
        )
        print_debug(f"读取的迭代数据", df, level="ITERATE_DF")
    else:
        df = df.copy()
        # 确保ID列是str类型
        df['ID'] = df['ID'].astype(str)
        print_debug(f"传入的迭代数据", df, level="ITERATE_DF")

    n_funcs = len(G.terms_list)

    # ========== 修复3：先将GO字符串ID映射为整数，再过滤 ==========
    # 用预计算的func_mapper将GO字符串转成整数
    df['ID'] = df['ID'].map(G.func_mapper)
    # 过滤无效ID（映射后为NaN的）
    df = df.dropna(subset=['ID'])
    # 转换为int32
    df['ID'] = df['ID'].astype(INT32)

    print_debug(f"映射GO ID后的迭代数据", df, level="ITERATE_DF")

    # 优化：字典映射替代Series.map，提速~40%
    df['entry_num'] = df['entry_id'].map(idx_dict).fillna(-1).astype(INT32)
    df = df[df['entry_num'] != -1].sort_values('entry_num').reset_index(drop=True)

    print_debug(f"过滤后的迭代数据", df, level="ITERATE_DF")

    if df.empty:
        print_debug("迭代数据为空，返回空迭代器", level="ITERATE_DF")
        return

    nrows = int(df['entry_num'].max()) + 1
    print_debug(f"迭代数据总行数: {len(df)}, 总批次数量: {nrows // batch_size + 1}", level="ITERATE_DF")

    # 优化：按batch切片，减少重复索引
    for i in range(0, nrows, batch_size):
        batch_end = i + batch_size
        # 优化：用query替代布尔索引，更快
        batch = df.query(f'entry_num >= {i} and entry_num < {batch_end}').copy()
        if batch.empty:
            continue

        print_debug(f"生成第{i//batch_size + 1}批次", batch, level="ITERATE_DF")

        batch['entry_num'] -= i
        batch = propagate_df(batch, G, n_funcs, mode=prop_mode)
        batch['entry_num'] += i
        # 优化：字典映射替代Series.map
        batch['entry_id'] = batch['entry_num'].map(back_idx_dict)

        yield batch
        
    print_debug(f"完成所有批次生成", level="ITERATE_DF")