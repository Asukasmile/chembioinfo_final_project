# -*- coding: utf-8 -*-
"""
CHEK1 抑制剂配体基粗筛 —— 对 Topscience (ChemDiv) 库做多策略评分
==================================================================
策略（配体基，两种互补信号）：
  1. ML 活性预测：用训练好的 RF + SVM 模型预测每个化合物的活性概率
     (score_ml = RF 概率，SVM 概率作为辅助)
  2. 相似性搜索：以 AZD7762 / Prexasertib 为 query，计算 Morgan 指纹
     Tanimoto 相似度，取最大值 (score_sim)
  综合打分后保留高分候选，把 168 万大幅富集缩小。

设计要点：
  - 分块读取 5 个 CSV，逐块处理，内存友好
  - 断点续跑：每块结果即时追加写入，中途中断可从断点继续
  - 同时计算类药性参数 (MW, LogP, HBD, HBA, TPSA) 供后续过滤

运行环境：conda env chem_rdkit
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import joblib

from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs, Descriptors, Crippen
from rdkit.Chem import Lipinski
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

# ----------------------------- 参数 -----------------------------
LIB_FILES = [
    'chemdiv_CleanD001-1.csv',
    'chemdiv_CleanD001-2.csv',
    'chemdiv_CleanD001-3.csv',
    'chemdiv_CleanD001-4.csv',
    'chemdiv_cleanD001-5.csv',
]
MODEL_DIR = 'models'
OUT_DIR = 'screening'
CHUNK_SIZE = 20000  # 每块化合物数

QUERY_SMILES = {
    'AZD7762': 'NC(=O)Nc1cc(-c2cccc(F)c2)sc1C(=O)N[C@H]1CCCNC1',
    'Prexasertib': 'COc1cccc(OCCCN)c1-c1cc(Nc2cnc(C#N)cn2)n[nH]1',
}

N_BITS = 2048
RADIUS = 2

os.makedirs(OUT_DIR, exist_ok=True)

# ----------------------------- 1. 加载模型 & query 指纹 -----------------------------
print('[1] 加载模型 ...')
rf = joblib.load(os.path.join(MODEL_DIR, 'rf_chk1.joblib'))
svm = joblib.load(os.path.join(MODEL_DIR, 'svm_chk1.joblib'))

print('[2] 生成 query 分子指纹 ...')
query_fps = {}
for name, smi in QUERY_SMILES.items():
    mol = Chem.MolFromSmiles(smi)
    query_fps[name] = AllChem.GetMorganFingerprintAsBitVect(mol, RADIUS, nBits=N_BITS)
    print(f'  {name}: {smi}')

def mol_to_fp(mol):
    return AllChem.GetMorganFingerprintAsBitVect(mol, RADIUS, nBits=N_BITS)

# ----------------------------- 2. 逐文件逐块处理 -----------------------------
# 输出文件（断点续跑用 append 模式，但表头只写一次）
out_csv = os.path.join(OUT_DIR, 'screening_scores.csv')

# 记录已处理的行数（用于断点续跑标记，这里简单用文件存在与否判断是否已写表头）
resume_from = 0  # 若中断，可手动设置从某文件某行继续

header_written = os.path.exists(out_csv)

def write_batch(records, mode):
    df = pd.DataFrame(records)
    df.to_csv(out_csv, mode=mode, header=not header_written, index=False)

total_kept = 0
total_processed = 0

for fi, fname in enumerate(LIB_FILES):
    if not os.path.exists(fname):
        print(f'  [跳过] 文件不存在: {fname}')
        continue
    print(f'\n[3] 处理文件 {fi+1}/{len(LIB_FILES)}: {fname} ...')
    reader = pd.read_csv(fname, chunksize=CHUNK_SIZE)
    for chunk in reader:
        total_processed += len(chunk)
        records = []
        ids = chunk['ID'].tolist() if 'ID' in chunk.columns else chunk.iloc[:, 0].tolist()
        smiles_list = chunk['SMILES'].tolist() if 'SMILES' in chunk.columns else chunk.iloc[:, 1].tolist()

        # 批量解析 + 指纹
        mols = [Chem.MolFromSmiles(s) for s in smiles_list]
        valid = [(i, m) for i, m in enumerate(mols) if m is not None]
        if not valid:
            continue

        fps = [mol_to_fp(m) for i, m in valid]
        fp_arr = np.array([np.array(fp, dtype=np.float32) for fp in fps])

        # ML 预测概率
        prob_rf = rf.predict_proba(fp_arr)[:, 1]
        prob_svm = svm.predict_proba(fp_arr)[:, 1]

        # 相似性（与两个 query 的最大 Tanimoto）
        sim_max = np.zeros(len(valid), dtype=np.float32)
        for i, fp in enumerate(fps):
            s_azd = DataStructs.TanimotoSimilarity(fp, query_fps['AZD7762'])
            s_prex = DataStructs.TanimotoSimilarity(fp, query_fps['Prexasertib'])
            sim_max[i] = max(s_azd, s_prex)

        # 综合打分 & 类药性
        for k, (orig_i, mol) in enumerate(valid):
            mw = Descriptors.MolWt(mol)
            logp = Crippen.MolLogP(mol)
            hbd = Lipinski.NumHDonors(mol)
            hba = Lipinski.NumHAcceptors(mol)
            tpsa = Descriptors.TPSA(mol)
            records.append({
                'ID': ids[orig_i],
                'SMILES': smiles_list[orig_i],
                'prob_rf': round(float(prob_rf[k]), 6),
                'prob_svm': round(float(prob_svm[k]), 6),
                'sim_max': round(float(sim_max[k]), 6),
                'MW': round(mw, 2),
                'LogP': round(logp, 3),
                'HBD': hbd,
                'HBA': hba,
                'TPSA': round(tpsa, 2),
            })

        # 即时写出（断点续跑友好）
        write_batch(records, 'a')
        header_written = True

        # 简单进度
        n_high = sum(1 for r in records if r['prob_rf'] > 0.5 or r['sim_max'] > 0.4)
        total_kept += n_high
        if total_processed % 100000 < CHUNK_SIZE:
            print(f'    已处理 {total_processed:,} 个化合物 ...')

print(f'\n[完成] 共处理 {total_processed:,} 个化合物')
print(f'        筛选结果已写入 {out_csv}')
