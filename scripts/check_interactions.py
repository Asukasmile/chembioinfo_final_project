#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
步骤5 关键相互作用判定：对共识前列的 Vina pose 做 CHEK1 关键相互作用分析。

判据（从 4HYI 共晶实测 + prexasertib 文献验证）：
  三段式口袋:
    hinge 区:      Glu85(主链O,氢键供体), Cys87(主链N-H,氢键受体), Tyr86
    gatekeeper:    Leu15
    核糖/极性区:   Glu91, Glu134, Asn135, Leu137
    内部疏水口袋:  Lys38, Glu55, Val68, Leu84, Ser147, Asp148

自动判定（供 PyMOL 人工质检前的机器预判）:
  1. hinge 氢键: 配体极性原子(HBD的H 或 HBA的N/O)与 Glu85 主链 O 或 Cys87 主链 N 距离 < 3.5 Å
  2. gatekeeper 接触: 配体任意原子与 Leu15 < 4.5 Å（进入深口袋）
  3. 深口袋占据: 配体与内部疏水口袋残基(Lys38/Glu55/Val68/Leu84/Ser147/Asp148) < 4.5 Å 的接触数

输出: vina/interaction_analysis.csv
"""
import os
import glob
import numpy as np
import pandas as pd

PROJECT = os.path.dirname(os.path.abspath(__file__))
VINA_DIR = os.path.join(PROJECT, 'vina')
PROTEIN_PDB = os.path.join(PROJECT, '4HYI_protein.pdb')

# 关键残基定义
HINGE_DONOR = [('GLU', 85)]   # 主链 O 作氢键受体（配体供 H）
HINGE_ACCEPTOR = [('CYS', 87)]  # 主链 N-H 作氢键供体（配体接受 H）
GATEKEEPER = [('LEU', 15)]
INTERIOR_POCKET = [('LYS', 38), ('GLU', 55), ('VAL', 68), ('LEU', 84),
                   ('SER', 147), ('ASP', 148)]
RIBOSE_POCKET = [('GLU', 91), ('GLU', 134), ('ASN', 135), ('LEU', 137)]

H_BOND_DIST = 3.5
CONTACT_DIST = 4.5

VDW = {'C': 1.70, 'N': 1.55, 'O': 1.52, 'F': 1.47, 'S': 1.80,
       'Cl': 1.75, 'Br': 1.85, 'P': 1.80}


def read_protein_atoms():
    """返回 {('GLU',85): [(elem,x,y,z,atomname), ...]}"""
    atoms = {}
    for line in open(PROTEIN_PDB, encoding='utf-8', errors='ignore'):
        if not line.startswith('ATOM'):
            continue
        resn = line[17:20].strip()
        resi = int(line[22:26])
        atomname = line[12:16].strip()
        elem = line[76:78].strip() if len(line) > 78 else line[12:16].strip()
        elem = elem[:1].upper()
        if elem == 'H':
            continue
        x = float(line[30:38]); y = float(line[38:46]); z = float(line[46:54])
        atoms.setdefault((resn, resi), []).append((elem, x, y, z, atomname))
    return atoms


def extract_best_pose(pdbqt_path):
    """提取 MODEL 1 的配体原子: [(elem, x, y, z, atomname)]"""
    atoms = []
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
            atomname = s[12:16].strip()
            elem_field = s[76:78].strip() if len(s) > 78 else ''
            if not elem_field:
                elem_field = ''.join(ch for ch in atomname if ch.isalpha())[:2]
            e = elem_field.upper()
            if e.startswith('CL'):
                elem = 'Cl'
            elif e.startswith('BR'):
                elem = 'Br'
            elif e.startswith('F'):
                elem = 'F'
            elif e.startswith('A'):
                elem = 'C'
            else:
                elem = e[0]
            if elem == 'H':
                continue
            if elem not in VDW:
                continue
            x = float(s[30:38]); y = float(s[38:46]); z = float(s[46:54])
            atoms.append((elem, x, y, z, atomname))
    return atoms


def min_dist(lig_atoms, prot_atoms):
    """配体原子 vs 某残基原子的最小距离"""
    best = 999.0
    for (le, lx, ly, lz, _) in lig_atoms:
        for (pe, px, py, pz, _) in prot_atoms:
            d = ((lx-px)**2 + (ly-py)**2 + (lz-pz)**2) ** 0.5
            if d < best:
                best = d
    return best


def check_hinge_hbond(lig_atoms, prot):
    """检查 hinge 氢键: 返回 (hbond_count, 距离 Glu85主链O, 距离 Cys87主链N)"""
    # Glu85 主链 O 原子
    glu85_O = [a for a in prot.get(('GLU', 85), []) if a[4] == 'O']
    # Cys87 主链 N 原子
    cys87_N = [a for a in prot.get(('CYS', 87), []) if a[4] == 'N']

    d_glu85 = min_dist(lig_atoms, glu85_O) if glu85_O else 999
    d_cys87 = min_dist(lig_atoms, cys87_N) if cys87_N else 999

    # 氢键判定：配体的 N/O 与 hinge 主链 O/N 距离 < 3.5 Å
    hbond = 0
    for (le, lx, ly, lz, _) in lig_atoms:
        if le not in ('N', 'O'):
            continue
        for (pe, px, py, pz, pn) in glu85_O:
            d = ((lx-px)**2 + (ly-py)**2 + (lz-pz)**2) ** 0.5
            if d < H_BOND_DIST:
                hbond += 1
        for (pe, px, py, pz, pn) in cys87_N:
            d = ((lx-px)**2 + (ly-py)**2 + (lz-pz)**2) ** 0.5
            if d < H_BOND_DIST:
                hbond += 1
    return hbond, d_glu85, d_cys87


def main():
    prot = read_protein_atoms()
    print(f'蛋白残基解析: {len(prot)} 个')

    # 读取共识排序
    cons = pd.read_csv(os.path.join(VINA_DIR, 'consensus_ranking.csv'))
    cons.columns = [c.strip('\ufeff') for c in cons.columns]

    # 对共识前列（默认 top 40）做相互作用分析
    N = 40
    top_ids = cons.head(N)['ID'].tolist()

    results = []
    for i, cid in enumerate(top_ids):
        vf = os.path.join(VINA_DIR, f'out_{cid}.pdbqt')
        if not os.path.exists(vf):
            continue
        lig_atoms = extract_best_pose(vf)
        if not lig_atoms:
            continue

        # 1. hinge 氢键
        n_hbond, d_glu85, d_cys87 = check_hinge_hbond(lig_atoms, prot)

        # 2. gatekeeper 接触
        d_gk = min_dist(lig_atoms, prot.get(('LEU', 15), []))

        # 3. 内部疏水口袋接触数
        interior_contacts = 0
        for res in INTERIOR_POCKET:
            if res in prot and min_dist(lig_atoms, prot[res]) < CONTACT_DIST:
                interior_contacts += 1

        # 4. 核糖口袋接触数
        ribose_contacts = 0
        for res in RIBOSE_POCKET:
            if res in prot and min_dist(lig_atoms, prot[res]) < CONTACT_DIST:
                ribose_contacts += 1

        # 判定
        has_hinge = n_hbond >= 1
        in_deep_pocket = d_gk < CONTACT_DIST
        if has_hinge and interior_contacts >= 3:
            verdict = 'STRONG'   # hinge + 深口袋
        elif has_hinge:
            verdict = 'HINGE_ONLY'  # 只有 hinge
        else:
            verdict = 'NO_HINGE'   # 无 hinge 氢键

        results.append({
            'ID': cid,
            'n_hinge_hbond': n_hbond,
            'dist_Glu85_O': round(d_glu85, 2),
            'dist_Cys87_N': round(d_cys87, 2),
            'dist_gatekeeper_Leu15': round(d_gk, 2),
            'n_interior_contacts': interior_contacts,
            'n_ribose_contacts': ribose_contacts,
            'verdict': verdict,
        })

    df_int = pd.DataFrame(results)
    out_csv = os.path.join(VINA_DIR, 'interaction_analysis.csv')
    df_int.to_csv(out_csv, index=False, encoding='utf-8-sig')

    print(f'\n=== 相互作用判定统计（共识 top {N}） ===')
    print(df_int['verdict'].value_counts().to_string())
    print(f'\n已保存: {out_csv}')

    # 合并回共识表
    merged = cons.merge(df_int, on='ID', how='left')
    merged.to_csv(os.path.join(VINA_DIR, 'consensus_ranking_interactions.csv'),
                  index=False, encoding='utf-8-sig')
    print(f'合并结果: vina/consensus_ranking_interactions.csv')

    print('\n=== Top 20 相互作用详情 ===')
    top20 = merged.head(20)
    cols = ['consensus_rank', 'ID', 'vina_score_kcal_mol', 'onionnet_pKd',
            'n_hinge_hbond', 'dist_Glu85_O', 'dist_Cys87_N',
            'dist_gatekeeper_Leu15', 'n_interior_contacts', 'verdict']
    print(top20[cols].to_string(index=False))


if __name__ == '__main__':
    main()
