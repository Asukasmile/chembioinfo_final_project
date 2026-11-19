#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
步骤4-5 共识排序：Vina score + OnionNet-2 pKd 平均排名（主）+ RF 辅助证据（辅）

核心逻辑（用户确认的方案）：
  1. Vina score（越低越好）与 OnionNet pKd（越高越好）各算排名
  2. 取平均排名 rank_avg 作为共识分（主排序依据）
  3. RF prob_rf + sim_max 作为辅助证据（标注，不参与主排序，仅在并列时决胜）
  4. 排除 pose 质检 FAIL 的分子（299801, 43034）
  5. 输出共识排序表，并标注 top-N

输出: vina/consensus_ranking.csv
"""
import os
import numpy as np
import pandas as pd

PROJECT = os.path.dirname(os.path.abspath(__file__))
VINA_DIR = os.path.join(PROJECT, 'vina')

# 读取 931 清单（含 prob_rf, sim_max）
lst = pd.read_csv(os.path.join(VINA_DIR, 'vina_931_list.csv'))
lst.columns = [c.strip('\ufeff') for c in lst.columns]

# 读取 Vina 对接结果
docking = pd.read_csv(os.path.join(VINA_DIR, 'vina_docking_931.csv'))
docking.columns = [c.strip('\ufeff') for c in docking.columns]
docking = docking[['ID', 'vina_score']].dropna(subset=['vina_score'])

# 读取 OnionNet 重打分
onion = pd.read_csv(os.path.join(VINA_DIR, 'vina_pose_onionnet.csv'))
onion.columns = [c.strip('\ufeff') for c in onion.columns]
onion = onion[['ID', 'vina_pose_pKd']].dropna(subset=['vina_pose_pKd'])

# 读取 pose 质检结果
quality = pd.read_csv(os.path.join(VINA_DIR, 'pose_quality_vina.csv'))
quality.columns = [c.strip('\ufeff') for c in quality.columns]
fail_ids = set(quality[quality['verdict'] == 'FAIL']['ID'])

# 合并
df = lst.merge(docking, on='ID', how='inner').merge(onion, on='ID', how='inner')

print(f'清单: {len(lst)}, Vina: {len(docking)}, OnionNet: {len(onion)}, 合并: {len(df)}')
print(f'pose 质检 FAIL (排除): {sorted(fail_ids)}')

# 排除 FAIL
df = df[~df['ID'].isin(fail_ids)]
print(f'排除 FAIL 后: {len(df)} 个分子进入共识排序\n')

# 排名：Vina 越低越好（rank 1 = 最低结合能）；OnionNet pKd 越高越好（rank 1 = 最高 pKd）
df['rank_vina'] = df['vina_score'].rank(method='min')          # 越小越靠前
df['rank_onion'] = df['vina_pose_pKd'].rank(method='min', ascending=False)  # 越大越靠前

# 共识分：平均排名（越小越好）
df['rank_avg'] = (df['rank_vina'] + df['rank_onion']) / 2.0

# 排序
df = df.sort_values('rank_avg').reset_index(drop=True)
df['consensus_rank'] = df.index + 1

# 整理列
out = df[['consensus_rank', 'ID', 'SMILES', 'vina_score', 'vina_pose_pKd',
          'rank_vina', 'rank_onion', 'rank_avg', 'prob_rf', 'sim_max', 'rank_score']].copy()
out = out.rename(columns={
    'vina_score': 'vina_score_kcal_mol',
    'vina_pose_pKd': 'onionnet_pKd',
    'rank_score': 'ligand_rank_score',
})

out_csv = os.path.join(VINA_DIR, 'consensus_ranking.csv')
out.to_csv(out_csv, index=False, encoding='utf-8-sig')

print('=== 共识排序完成 ===')
print(f'已保存: {out_csv}\n')

# 统计
print('=== Top 20 共识排名 ===')
print(out.head(20)[['consensus_rank', 'ID', 'vina_score_kcal_mol', 'onionnet_pKd',
                    'rank_vina', 'rank_onion', 'rank_avg', 'prob_rf', 'sim_max']].to_string(index=False))

print('\n=== 相关性 ===')
r = np.corrcoef(df['vina_score'], df['vina_pose_pKd'])[0, 1]
print(f'Vina vs OnionNet pKd: r = {r:+.3f}')

# 三个榜单 top20 重叠情况
top20_cons = set(out.head(20)['ID'])
top20_vina = set(df.nsmallest(20, 'vina_score')['ID'])
top20_onion = set(df.nlargest(20, 'vina_pose_pKd')['ID'])
print(f'\nTop20 共识 ∩ Top20 Vina: {len(top20_cons & top20_vina)} 个')
print(f'Top20 共识 ∩ Top20 OnionNet: {len(top20_cons & top20_onion)} 个')
print(f'Top20 Vina ∩ Top20 OnionNet: {len(top20_vina & top20_onion)} 个')
