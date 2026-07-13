import argparse
import os
import sys

import yaml

sys.path.append(os.path.abspath(os.path.join(__file__, '../../../../')))

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config-path', type=str)
parser.add_argument('-d', '--device', type=str, default="1")

if __name__ == '__main__':
    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    # ========== 核心修改1：替换cudf为pandas ==========
    import pandas as pd  # 替换 import cudf

    try:
        from protlib.metric import CAFAMetric
    except Exception:
        CAFAMetric = None

    with open(args.config_path) as f:
        config = yaml.safe_load(f)

    graph_path = os.path.join(config['base_path'], 'Train/go-basic.obo')
    ia_path = os.path.join(config['base_path'], 'IA.txt')
    pp_path = os.path.join(config['base_path'], config['models_path'], 'postproc')
    os.makedirs(pp_path, exist_ok=True)

    metric = CAFAMetric(
        graph_path,
        ia_path,
        None,
        prop_mode='fill',
        batch_size=30000
    )

    tta_path = os.path.join(config['base_path'], config['models_path'], 'gcn/pred_tta_{0}.tsv')
    # ========== 核心修改2：cudf.read_csv → pd.read_csv ==========
    pred = pd.read_csv(tta_path.format(0), sep='\t', header=None, names=['EntryID', 'term', 'prob0'])

    for i in range(1, 4):
        # ========== 核心修改3：合并时同样使用pandas ==========
        pred = pd.merge(
            pred,
            pd.read_csv(tta_path.format(i), sep='\t', header=None, names=['EntryID', 'term', f'prob{i}']),
            how='outer', on=['EntryID', 'term']
        ).fillna(0)  # fillna用法和cudf一致

    # 计算概率平均值（API完全兼容）
    pred['prob'] = pred[['prob0', 'prob1', 'prob2', 'prob3']].mean(axis=1)
    # ========== 核心修改4：保存文件时保持参数不变 ==========
    pred[['EntryID', 'term', 'prob']].to_csv(
        os.path.join(pp_path, 'pred.tsv'), header=False, index=False, sep='\t'
    )
    label_path = os.path.join(config['base_path'], config['models_path'], 'gcn/bp/temp/labels.tsv')
    if os.path.exists(label_path):
        score = metric.from_df(label_path, os.path.join(pp_path, 'pred.tsv'))
        print('CAFA5 Scores')
        print(score)