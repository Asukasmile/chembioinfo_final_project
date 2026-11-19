# -*- coding: utf-8 -*-
"""
CHEK1 抑制剂配体基粗筛 —— 多进程并行版（v2，仅 RF，提速）
==========================================================
对 Topscience (ChemDiv) 库（168 万）做配体基评分：
  1. ML 活性预测（Random Forest 概率，训练时性能最优，AUC 0.994）
  2. 相似性搜索（AZD7762 / Prexasertib 的 Morgan Tanimoto 最大值）

优化：
  - 去掉慢速的 SVM predict_proba（基准：SVM 占 60% 时间，且与 RF 信息冗余）
  - multiprocessing.Pool 32 进程并行
  - RF 内部 n_jobs=1，避免 Loky 冲突

运行环境：conda env chem_rdkit（32 核 CPU）
"""

import os
import numpy as np
import pandas as pd
import joblib
from multiprocessing import Pool

from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs, Descriptors, Crippen, Lipinski
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
N_PROC = 32

QUERY_SMILES = {
    'AZD7762': 'NC(=O)Nc1cc(-c2cccc(F)c2)sc1C(=O)N[C@H]1CCCNC1',
    'Prexasertib': 'COc1cccc(OCCCN)c1-c1cc(Nc2cnc(C#N)cn2)n[nH]1',
}
N_BITS = 2048
RADIUS = 2

os.makedirs(OUT_DIR, exist_ok=True)

# ----------------------------- 全局对象 -----------------------------
_G = {}

def _init_worker():
    _G['rf'] = joblib.load(os.path.join(MODEL_DIR, 'rf_chk1.joblib'))
    _G['rf'].n_jobs = 1
    _G['query_fps'] = {}
    for name, smi in QUERY_SMILES.items():
        m = Chem.MolFromSmiles(smi)
        _G['query_fps'][name] = AllChem.GetMorganFingerprintAsBitVect(m, RADIUS, nBits=N_BITS)

def _process_chunk(args):
    ids, smiles_list = args
    rf = _G['rf']
    query_fps = _G['query_fps']

    records = []
    mols = [Chem.MolFromSmiles(s) for s in smiles_list]
    valid = [(i, m) for i, m in enumerate(mols) if m is not None]
    if not valid:
        return records

    fps = [AllChem.GetMorganFingerprintAsBitVect(m, RADIUS, nBits=N_BITS) for i, m in valid]
    fp_arr = np.array([np.array(fp, dtype=np.float32) for fp in fps])

    prob_rf = rf.predict_proba(fp_arr)[:, 1]

    for k, (orig_i, mol) in enumerate(valid):
        fp = fps[k]
        s_azd = DataStructs.TanimotoSimilarity(fp, query_fps['AZD7762'])
        s_prex = DataStructs.TanimotoSimilarity(fp, query_fps['Prexasertib'])
        sim_max = max(s_azd, s_prex)

        records.append({
            'ID': ids[orig_i],
            'SMILES': smiles_list[orig_i],
            'prob_rf': round(float(prob_rf[k]), 6),
            'sim_max': round(float(sim_max), 6),
            'MW': round(Descriptors.MolWt(mol), 2),
            'LogP': round(Crippen.MolLogP(mol), 3),
            'HBD': Lipinski.NumHDonors(mol),
            'HBA': Lipinski.NumHAcceptors(mol),
            'TPSA': round(Descriptors.TPSA(mol), 2),
        })
    return records

# ----------------------------- 主流程 -----------------------------
def main():
    print('[1] 初始化 32 进程池 ...', flush=True)
    pool = Pool(processes=N_PROC, initializer=_init_worker)

    out_csv = os.path.join(OUT_DIR, 'screening_scores.csv')
    if os.path.exists(out_csv):
        os.remove(out_csv)
    header_written = False

    total_processed = 0
    CHUNK = 20000

    for fname in LIB_FILES:
        if not os.path.exists(fname):
            print(f'  [跳过] {fname}')
            continue
        print(f'[2] 处理 {fname} ...', flush=True)
        df = pd.read_csv(fname)
        ids = df['ID'].tolist() if 'ID' in df.columns else df.iloc[:, 0].tolist()
        smiles = df['SMILES'].tolist() if 'SMILES' in df.columns else df.iloc[:, 1].tolist()

        tasks = []
        for i in range(0, len(ids), CHUNK):
            tasks.append((ids[i:i+CHUNK], smiles[i:i+CHUNK]))

        for result in pool.imap_unordered(_process_chunk, tasks, chunksize=1):
            total_processed += len(result)
            if result:
                pdf = pd.DataFrame(result)
                pdf.to_csv(out_csv, mode='a', header=not header_written, index=False)
                header_written = True

    pool.close()
    pool.join()
    print(f'\n[完成] 共处理 {total_processed:,} 个化合物')
    print(f'        结果: {out_csv}')

if __name__ == '__main__':
    main()
