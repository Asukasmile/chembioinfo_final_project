# -*- coding: utf-8 -*-
"""
CHEK1 候选化合物 —— 骨架聚类 & 分子多样性分析
================================================
对配体基筛出的 top 候选 (screening/top_candidates.csv) 做：
  1. Bemis-Murcko 骨架 (scaffold) 提取与统计
  2. 骨架聚类（按骨架分组，统计每类的化合物数）
  3. 分子多样性分析（Tanimoto 内部相似度、骨架多样性）
  4. 输出：骨架分布表、多样性指标、Top 骨架结构

输出：
  - screening/scaffold_analysis.csv  (每个候选的骨架 SMILES + 类别)
  - screening/scaffold_summary.csv   (骨架频率统计)
  - screening/diversity_metrics.json (多样性指标)

运行环境：conda env chem_rdkit
"""

import os
import json
import numpy as np
import pandas as pd
from collections import Counter

from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

# ----------------------------- 参数 -----------------------------
IN_CSV = 'screening/top_candidates.csv'
OUT_DIR = 'screening'
N_BITS = 2048

df = pd.read_csv(IN_CSV)
print(f'[1] 读取候选: {len(df)} 个化合物')

# ----------------------------- 2. 提取骨架 -----------------------------
print('[2] 提取 Bemis-Murcko 骨架 ...')
scaffolds = []
scaffold_generic = []  # 泛化骨架（去掉侧链，仅保留环系统+连接子）
valid_mols = []
for smi in df['SMILES']:
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        scaffolds.append(None)
        scaffold_generic.append(None)
        continue
    try:
        scaff = MurckoScaffold.GetScaffoldForMol(mol)
        scaff_smi = Chem.MolToSmiles(scaff)
        # 泛化：把骨架中的所有原子变成碳，保留环拓扑（用于更粗的聚类）
        gen = MurckoScaffold.MakeScaffoldGeneric(scaff)
        gen_smi = Chem.MolToSmiles(gen)
    except Exception:
        scaff_smi = None
        gen_smi = None
    scaffolds.append(scaff_smi)
    scaffold_generic.append(gen_smi)
    valid_mols.append(mol)

df['scaffold'] = scaffolds
df['scaffold_generic'] = scaffold_generic

print(f'  有效骨架: {sum(1 for s in scaffolds if s)}')

# ----------------------------- 3. 骨架频率统计 -----------------------------
print('[3] 骨架聚类统计 ...')
scaff_counts = Counter(s for s in scaffolds if s)
gen_counts = Counter(s for s in scaffold_generic if s)

scaff_summary = pd.DataFrame(scaff_counts.most_common(), columns=['scaffold', 'count'])
scaff_summary['freq_pct'] = (scaff_summary['count'] / len(df) * 100).round(2)

print(f'  唯一骨架数: {len(scaff_counts)}')
print(f'  唯一泛化骨架数: {len(gen_counts)}')

# ----------------------------- 4. 多样性指标 -----------------------------
print('[4] 计算分子多样性指标 ...')

# 4.1 骨架多样性 = 唯一骨架数 / 化合物数
scaffold_diversity = len(scaff_counts) / len(df)

# 4.2 内部 Tanimoto 相似度（采样计算，避免 O(n^2) 太大）
# 对 top 候选采样 500 个计算平均成对相似度
sample_n = min(500, len(valid_mols))
rng = np.random.RandomState(42)
sample_idx = rng.choice(len(valid_mols), sample_n, replace=False)
sample_fps = []
for i in sample_idx:
    fp = AllChem.GetMorganFingerprintAsBitVect(valid_mols[i], 2, nBits=N_BITS)
    sample_fps.append(fp)

pair_sims = []
for i in range(sample_n):
    for j in range(i+1, sample_n):
        pair_sims.append(DataStructs.TanimotoSimilarity(sample_fps[i], sample_fps[j]))

pair_sims = np.array(pair_sims)
mean_internal_sim = float(pair_sims.mean())
median_internal_sim = float(np.median(pair_sims))

# 4.3 前 10 骨架的集中度（top10 骨架覆盖多少化合物）
top10_frac = scaff_summary.head(10)['count'].sum() / len(df)

metrics = {
    'n_compounds': int(len(df)),
    'n_unique_scaffolds': int(len(scaff_counts)),
    'n_unique_generic_scaffolds': int(len(gen_counts)),
    'scaffold_diversity': round(scaffold_diversity, 4),
    'mean_internal_tanimoto': round(mean_internal_sim, 4),
    'median_internal_tanimoto': round(median_internal_sim, 4),
    'top10_scaffold_coverage_frac': round(top10_frac, 4),
    'singleton_scaffold_count': int(sum(1 for c in scaff_counts.values() if c == 1)),
}

print()
print('=== 多样性指标 ===')
for k, v in metrics.items():
    print(f'  {k}: {v}')

# ----------------------------- 5. 保存结果 -----------------------------
df.to_csv(os.path.join(OUT_DIR, 'scaffold_analysis.csv'), index=False, encoding='utf-8-sig')
scaff_summary.to_csv(os.path.join(OUT_DIR, 'scaffold_summary.csv'), index=False, encoding='utf-8-sig')
with open(os.path.join(OUT_DIR, 'diversity_metrics.json'), 'w', encoding='utf-8') as f:
    json.dump(metrics, f, indent=2, ensure_ascii=False)

print()
print('=== 保存完成 ===')
print(f'  {OUT_DIR}/scaffold_analysis.csv')
print(f'  {OUT_DIR}/scaffold_summary.csv')
print(f'  {OUT_DIR}/diversity_metrics.json')

print()
print('=== Top 15 高频骨架 ===')
print(scaff_summary.head(15).to_string(index=False))
