#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
步骤5-6 最终选择：骨架多样性 + 类药性(Lipinski) + 综合评分 → 5-10 个 lead

对共识排序前列（STRONG 相互作用分子优先）做：
  1. Bemis-Murcko 骨架提取 + 骨架聚类（多样性去重）
  2. 类药性五规则 (Lipinski Ro5) 统计
  3. 综合评分：共识排名 + 相互作用(hinge) + 多样性
  4. 输出最终 5-10 个候选清单 + 生成 PyMOL 查看的复合物 PDB

输出:
  vina/final_leads.csv
  vina/final_leads_analysis.csv (含骨架/类药性)
"""
import os
import numpy as np
import pandas as pd
from collections import Counter

from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs, Descriptors, Lipinski, Crippen
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

PROJECT = os.path.dirname(os.path.abspath(__file__))
VINA_DIR = os.path.join(PROJECT, 'vina')

# 读取含相互作用的共识排序
df = pd.read_csv(os.path.join(VINA_DIR, 'consensus_ranking_interactions.csv'))
df.columns = [c.strip('\ufeff') for c in df.columns]

print(f'共识排序分子数: {len(df)}')


def calc_props(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mw = Descriptors.MolWt(mol)
    logp = Crippen.MolLogP(mol)
    hbd = Lipinski.NumHDonors(mol)
    hba = Lipinski.NumHAcceptors(mol)
    tpsa = Descriptors.TPSA(mol)
    rotb = Lipinski.NumRotatableBonds(mol)
    rings = Descriptors.RingCount(mol)
    # Ro5 违例数
    violations = 0
    if mw > 500: violations += 1
    if logp > 5: violations += 1
    if hbd > 5: violations += 1
    if hba > 10: violations += 1
    return dict(mw=mw, logp=logp, hbd=hbd, hba=hba, tpsa=tpsa,
                rotb=rotb, rings=rings, ro5_violations=violations)


def get_scaffold(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        scaf = MurckoScaffold.GetScaffoldForMol(mol)
        return Chem.MolToSmiles(scaf)
    except Exception:
        return None


# 计算性质
props = []
scaffolds = []
for smi in df['SMILES']:
    p = calc_props(smi)
    props.append(p)
    scaffolds.append(get_scaffold(smi))

df['mw'] = [p['mw'] if p else np.nan for p in props]
df['logp'] = [p['logp'] if p else np.nan for p in props]
df['hbd'] = [p['hbd'] if p else np.nan for p in props]
df['hba'] = [p['hba'] if p else np.nan for p in props]
df['tpsa'] = [p['tpsa'] if p else np.nan for p in props]
df['rotb'] = [p['rotb'] if p else np.nan for p in props]
df['rings'] = [p['rings'] if p else np.nan for p in props]
df['ro5_violations'] = [p['ro5_violations'] if p else np.nan for p in props]
df['scaffold'] = scaffolds

# 骨架聚类编号
scaf_to_id = {}
scaf_counter = Counter()
for s in scaffolds:
    if s is not None:
        scaf_counter[s] += 1

df['scaffold_id'] = [scaf_counter[s] if s else np.nan for s in scaffolds]

print(f'\n=== 类药性统计（全部 {len(df)} 个） ===')
print(f'Ro5 全部满足(0违例): {sum(df["ro5_violations"]==0)} 个')
print(f'Ro5 1违例: {sum(df["ro5_violations"]==1)} 个')
print(f'Ro5 ≥2违例: {sum(df["ro5_violations"]>=2)} 个')

# 只看 STRONG 相互作用的分子
strong = df[df['verdict'] == 'STRONG'].copy()
print(f'\n=== STRONG 相互作用分子: {len(strong)} 个 ===')
print(f'  其中 Ro5 0违例: {sum(strong["ro5_violations"]==0)} 个')

# 在 STRONG 里做骨架多样性：优先每个骨架选共识排名最高的
strong_sorted = strong.sort_values('consensus_rank')

# 最终选择逻辑：
# 1. 优先 STRONG 分子
# 2. 骨架多样性：每个骨架最多选 1-2 个
# 3. 类药性：Ro5 违例 ≤ 1
# 4. 共识排名靠前
selected = []
seen_scaffolds = Counter()
for _, row in strong_sorted.iterrows():
    scaf = row['scaffold']
    if scaf is None:
        continue
    if row['ro5_violations'] > 1:
        continue  # 排除类药性差
    if seen_scaffolds[scaf] >= 1:
        continue  # 每个骨架最多 1 个（保证多样性）
    selected.append(row)
    seen_scaffolds[scaf] += 1
    if len(selected) >= 10:
        break

print(f'\n=== 最终 lead 选择 ===')
print(f'筛选条件: STRONG + Ro5≤1违例 + 骨架去重')
print(f'最终 lead 数量: {len(selected)}')

final_df = pd.DataFrame(selected).reset_index(drop=True)
if len(final_df) > 0:
    final_df['final_rank'] = final_df.index + 1

    # 输出最终清单
    out_cols = ['final_rank', 'consensus_rank', 'ID', 'SMILES',
                'vina_score_kcal_mol', 'onionnet_pKd', 'rank_avg',
                'prob_rf', 'sim_max', 'n_hinge_hbond',
                'dist_Glu85_O', 'dist_Cys87_N', 'dist_gatekeeper_Leu15',
                'n_interior_contacts', 'scaffold', 'mw', 'logp', 'hbd', 'hba',
                'tpsa', 'ro5_violations']
    final_out = final_df[out_cols]
    final_out.to_csv(os.path.join(VINA_DIR, 'final_leads.csv'), index=False, encoding='utf-8-sig')

    print(f'\n已保存: vina/final_leads.csv\n')
    print('=== 最终 lead 候选 ===')
    show_cols = ['final_rank', 'ID', 'vina_score_kcal_mol', 'onionnet_pKd',
                 'rank_avg', 'n_hinge_hbond', 'ro5_violations', 'scaffold']
    print(final_out[show_cols].to_string(index=False))

# 保存完整分析
df.to_csv(os.path.join(VINA_DIR, 'consensus_ranking_full_analysis.csv'),
          index=False, encoding='utf-8-sig')
print(f'\n完整分析: vina/consensus_ranking_full_analysis.csv')
