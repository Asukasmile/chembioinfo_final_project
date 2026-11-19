#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
用 Vina 生成的 pose 重新喂给 OnionNet-2 打分。

核心逻辑:
  - 从 vina/out_<id>.pdbqt 提取每个分子的最佳 pose (MODEL 1, 最低结合能)
  - 转换成 OnionNet 需要的配体 PDB 格式 (最后一列=纯元素符号)
  - 调用 OnionNet generate_features 重打分
  - 输出 vina_pose_onionnet.csv (Vina 结合能 + OnionNet pKd 对比)

环境: cnn (Python 3.7 + TensorFlow, OnionNet 依赖)
用法:
  python run_onionnet_on_vina.py
"""
import os
import sys
import glob
import time
import numpy as np
import pandas as pd

PROJECT = os.path.dirname(os.path.abspath(__file__))
SCORING = os.path.join(PROJECT, 'OnionNet-2', 'scoring')
MODELS = os.path.join(PROJECT, 'OnionNet-2', 'models')
PROTEIN = os.path.join(PROJECT, '4HYI_protein.pdb')
VINA_DIR = os.path.join(PROJECT, 'vina')
OUT_DIR = os.path.join(PROJECT, 'onionnet_vina_results')
os.makedirs(OUT_DIR, exist_ok=True)

sys.path.insert(0, SCORING)
import predict as pred

class _Args:
    shells = 62

if not hasattr(pred, 'args'):
    pred.args = _Args()

import itertools
_rec_defined_res = ['GLY', 'ALA', 'VAL', 'LEU', 'ILE', 'PRO', 'PHE', 'TYR', 'TRP', 'SER',
                    'THR', 'CYS', 'MET', 'ASN', 'GLN', 'ASP', 'GLU', 'LYS', 'ARG', 'HIS', 'OTH']
_lig_defined_ele = ['H', 'C', 'O', 'N', 'P', 'S', 'Hal', 'DU']
pred.keys = ["_".join(x) for x in list(itertools.product(_rec_defined_res, _lig_defined_ele))]

import tensorflow as tf
import joblib

# ============ 1. 加载 OnionNet 模型 ============
shape = [int(x) for x in "84,124,1".split(",")]
scaler = joblib.load(os.path.join(MODELS, 'train_scaler.scaler'))
model_path = os.path.join(MODELS, '62shell_saved-model.h5')
print('[1/3] 加载 OnionNet 模型 ...')
model = tf.keras.models.load_model(
    model_path,
    custom_objects={'RMSE': pred.RMSE, 'PCC': pred.PCC, 'PCC_RMSE': pred.PCC_RMSE},
)
print('[1/3] 模型加载完成')


# ============ 2. 从 Vina PDBQT 提取最佳 pose -> 配体 PDB ============
# AutoDock 原子类型 -> 元素符号
AD_TYPE_TO_ELE = {
    'C': 'C', 'A': 'C', 'N': 'N', 'NA': 'N', 'NS': 'N', 'NR': 'N', 'N1': 'N', 'N2': 'N',
    'O': 'O', 'OA': 'O', 'OS': 'O', 'O1': 'O', 'O2': 'O', 'OW': 'O',
    'H': 'H', 'HD': 'H', 'HS': 'H', 'HO': 'H',
    'P': 'P', 'S': 'S', 'SA': 'S', 'F': 'F', 'Cl': 'Cl', 'Br': 'Br', 'I': 'I',
    'DU': 'C', 'Z': 'C', 'MG': 'Mg', 'CA': 'Ca',
}


def extract_best_pose_pdb(pdbqt_path, out_pdb):
    """从 Vina 输出 PDBQT 提取 MODEL 1 (最佳 pose) 的原子, 写成 OnionNet 配体 PDB"""
    atoms = []  # (serial, elem, x, y, z)
    in_model = False
    for line in open(pdbqt_path, encoding='utf-8', errors='ignore'):
        s = line.rstrip('\n')
        if s.startswith('MODEL'):
            in_model = True
            continue
        if s.startswith('ENDMDL'):
            break
        if not in_model:
            continue
        if s.startswith('ATOM') or s.startswith('HETATM'):
            # 解析: 元素符号从第 77-78 列(PDB标准), 若空则从 atomname
            atomname = s[12:16].strip()
            elem_field = s[76:78].strip() if len(s) > 78 else ''
            if not elem_field:
                # 从 atomname 提取 (如 'C01' -> 'C')
                elem_field = ''.join(ch for ch in atomname if ch.isalpha())[:2]

            # AutoDock 类型 -> 元素 (PDBQT 最后一列是 AutoDock 类型)
            ad_type = s.split()[-1] if s.split() else ''
            if elem_field in AD_TYPE_TO_ELE:
                elem = AD_TYPE_TO_ELE[elem_field]
            elif ad_type in AD_TYPE_TO_ELE:
                elem = AD_TYPE_TO_ELE[ad_type]
            else:
                # 从 atomname 兜底
                elem = ''.join(ch for ch in atomname if ch.isalpha())[:2].capitalize()

            x = float(s[30:38]); y = float(s[38:46]); z = float(s[46:54])
            atoms.append((elem, x, y, z))

    if not atoms:
        return None

    with open(out_pdb, 'w') as f:
        for i, (elem, x, y, z) in enumerate(atoms, 1):
            name = f'{elem.upper():>2}'
            line = (f'ATOM  {i:5d} {name:^4s} LIG B   1    '
                    f'{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          '
                    f'{elem.upper():>2s}  ')
            f.write(line + '\n')
        f.write('END\n')
    return out_pdb


# ============ 3. 逐分子重打分 ============
# 只处理 931 清单里的目标分子 (排除之前批次混入的 out_*.pdbqt)
TARGET_CSV = os.path.join(VINA_DIR, 'vina_931_list.csv')
_target_ids = None
if os.path.exists(TARGET_CSV):
    _tdf = pd.read_csv(TARGET_CSV)
    _tdf.columns = [c.strip('\ufeff') for c in _tdf.columns]
    _target_ids = set(int(x) for x in _tdf['ID'])

all_files = sorted(glob.glob(os.path.join(VINA_DIR, 'out_*.pdbqt')))
vina_files = []
for vf in all_files:
    cid = os.path.basename(vf).replace('out_', '').replace('.pdbqt', '')
    if _target_ids is None or int(cid) in _target_ids:
        vina_files.append(vf)
print(f'[2/3] 找到 {len(all_files)} 个 Vina 对接结果, 其中 931 清单目标 {len(vina_files)} 个')

results = []
t0 = time.time()
for i, vf in enumerate(vina_files):
    cid = os.path.basename(vf).replace('out_', '').replace('.pdbqt', '')

    # 提取 Vina 最佳结合能
    vina_score = None
    for line in open(vf, encoding='utf-8', errors='ignore'):
        if 'VINA RESULT' in line:
            parts = line.split()
            if len(parts) >= 4:
                vina_score = float(parts[3])
                break

    # 转换 pose -> 配体 PDB
    tmp_lig = os.path.join(OUT_DIR, f'lig_{cid}.pdb')
    if not extract_best_pose_pdb(vf, tmp_lig):
        print(f'  [{i+1}/{len(vina_files)}] {cid}: pose 提取失败')
        continue

    # OnionNet 打分
    try:
        feat = pred.generate_features(PROTEIN, tmp_lig)
        X_std = scaler.transform(feat.reshape(1, -1)).reshape([-1] + shape)
        pKd = float(model.predict(X_std).ravel()[0])
        results.append({'ID': int(cid), 'vina_score': vina_score, 'vina_pose_pKd': pKd})
    except Exception as e:
        print(f'  [{i+1}/{len(vina_files)}] {cid}: OnionNet 打分失败 {e}')
        continue

    if (i + 1) % 20 == 0 or (i + 1) == len(vina_files):
        elapsed = time.time() - t0
        print(f'  进度 {i+1}/{len(vina_files)} ({elapsed:.0f}s)', flush=True)

# ============ 4. 保存 + 相关性 ============
res_df = pd.DataFrame(results)
out_csv = os.path.join(PROJECT, 'vina', 'vina_pose_onionnet.csv')
res_df.to_csv(out_csv, index=False, encoding='utf-8-sig')

print(f'\n[3/3] 完成! {len(res_df)} 个分子 -> {out_csv}')

if len(res_df) > 2:
    r = np.corrcoef(res_df['vina_score'], res_df['vina_pose_pKd'])[0, 1]
    print(f'\n=== Vina 结合能 vs OnionNet pKd (Vina pose) ===')
    print(f'Pearson r = {r:+.3f}')
    print(f'\nVina 结合能范围: {res_df["vina_score"].min():.2f} ~ {res_df["vina_score"].max():.2f}')
    print(f'OnionNet pKd 范围: {res_df["vina_pose_pKd"].min():.3f} ~ {res_df["vina_pose_pKd"].max():.3f}')
