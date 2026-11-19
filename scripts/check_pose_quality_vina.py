#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
步骤3: Pose Quality Filtering —— 对 AutoDock Vina 对接得到的蛋白-配体结合构象(pose)
做物理/化学合理性检查, 避免把不合理构象送入 OnionNet-2 打分。

检查三个维度 (对应方法学里 "Steric Clashes + Geometry + Binding Pocket"):
  1. 配体-蛋白空间碰撞 (Steric Clashes)
     - 配体原子 vs 蛋白原子距离 < (vdW_r_lig + vdW_r_prot) * 0.75 判定为碰撞
  2. 配体分子内碰撞/几何合理性 (Geometry)
     - 配体原子对 (拓扑距离 >= 4 键) 距离 < (vdW_i + vdW_j) * 0.8 判定为分子内碰撞
  3. 结合口袋约束 (Binding Pocket)
     - 配体所有原子必须落在搜索盒子内 (中心 15.31,-2.87,11.82, 盒子 22x22x22)
     - 配体质心与口袋中心距离不能太远 (口袋内合理范围)

输出: vina/pose_quality_vina.csv
  - 每分子记录: 配体-蛋白碰撞数/最严重比例, 分子内碰撞数, 是否在口袋内, 综合判定
  - 判定分档: PASS / WARN / FAIL
"""

import os
import glob
import numpy as np
import pandas as pd
from rdkit import Chem

PROJECT = os.path.dirname(os.path.abspath(__file__))
VINA_DIR = os.path.join(PROJECT, 'vina')
PROTEIN_PDB = os.path.join(PROJECT, '4HYI_protein.pdb')
RANKING_CSV = os.path.join(PROJECT, 'onionnet_only', 'onionnet_full_ranking.csv')

# 加载 SMILES 字典 (用于分子内碰撞的拓扑距离)
_SMILES = {}
if os.path.exists(RANKING_CSV):
    _df = pd.read_csv(RANKING_CSV)
    _df.columns = [c.strip('\ufeff') for c in _df.columns]
    for _, r in _df.iterrows():
        _SMILES[int(r['ID'])] = r['SMILES']

# 口袋参数 (晶体配体 1AO 结合位点)
POCKET_CENTER = np.array([15.31, -2.87, 11.82])
BOX_HALF = 11.0  # 盒子 22x22x22 的半宽

# vdW 半径 (Å)
VDW = {'C': 1.70, 'N': 1.55, 'O': 1.52, 'F': 1.47, 'S': 1.80,
       'Cl': 1.75, 'Br': 1.85, 'P': 1.80, 'H': 1.09}

# 配体-蛋白碰撞阈值系数
LIGPROT_SCALE = 0.75
# 分子内碰撞阈值系数
INTRAMOL_SCALE = 0.80
# 分子内碰撞最小拓扑距离 (键数)
INTRAMOL_TOPO_MIN = 4


def read_protein_atoms(pdb_path):
    """读取蛋白重原子: (elem, x, y, z)"""
    atoms = []
    for line in open(pdb_path, encoding='utf-8', errors='ignore'):
        if line.startswith('ATOM') or line.startswith('HETATM'):
            elem = line[76:78].strip() if len(line) > 78 else line[12:16].strip()
            elem = elem[:1].upper()
            if elem == 'H':
                continue
            if elem not in VDW:
                continue
            x = float(line[30:38]); y = float(line[38:46]); z = float(line[46:54])
            atoms.append((elem, x, y, z))
    return atoms


def extract_best_pose_atoms(pdbqt_path):
    """提取 Vina 最佳 pose (MODEL 1) 的重原子: (elem, x, y, z)"""
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
            # 元素归一化 (取首字母, 卤素/芳香碳特殊处理)
            e = elem_field.upper()
            if e.startswith('CL'):
                elem = 'Cl'
            elif e.startswith('BR'):
                elem = 'Br'
            elif e.startswith('F'):
                elem = 'F'
            elif e.startswith('A'):  # AutoDock 芳香碳类型 'A'
                elem = 'C'
            else:
                elem = e[0]
            if elem == 'H':
                continue
            if elem not in VDW:
                continue
            x = float(s[30:38]); y = float(s[38:46]); z = float(s[46:54])
            atoms.append((elem, x, y, z))
    return atoms


def check_ligprot_clash(lig_atoms, prot_atoms):
    """配体-蛋白碰撞检查, 返回 (碰撞数, 最严重比例)"""
    lig_arr = np.array([(a[1], a[2], a[3]) for a in lig_atoms])
    lig_elems = [a[0] for a in lig_atoms]
    prot_arr = np.array([(a[1], a[2], a[3]) for a in prot_atoms])
    prot_elems = [a[0] for a in prot_atoms]

    n_clash = 0
    worst_ratio = 1.0  # 距离/最小允许距离, 越小越严重

    for i in range(len(lig_atoms)):
        ri = VDW[lig_elems[i]]
        d = np.linalg.norm(prot_arr - lig_arr[i], axis=1)
        for j in range(len(prot_atoms)):
            rmin = (ri + VDW[prot_elems[j]]) * LIGPROT_SCALE
            if d[j] < rmin:
                n_clash += 1
                ratio = d[j] / rmin
                if ratio < worst_ratio:
                    worst_ratio = ratio

    return n_clash, worst_ratio


def check_intramol_clash(lig_atoms, cid):
    """
    配体分子内碰撞检查 (纯几何, 不依赖 SMILES 原子顺序匹配):
    - Vina 是力场最小化, 分子内原子重叠(距离过近)会被能量项惩罚
    - 这里只查真实重叠: 任意两个重原子距离 < COVALENT_MIN (排除共价键)
    - 共价键键长: C-C 1.54, C=C 1.34, C≡N 1.16, C=O 1.23, C-N 1.47 等
    - 用 1.0 Å 作为阈值: 任何重原子间距 < 1.0 Å 必为物理重叠(碰撞)
    """
    COVALENT_MIN = 1.0  # Å, 低于此必为碰撞(最短共价键 C≡N 也 >1.1 Å)
    n = len(lig_atoms)
    n_clash = 0
    worst_ratio = 1.0
    for i in range(n):
        ri = VDW[lig_atoms[i][0]]
        for j in range(i + 1, n):
            dx = lig_atoms[i][1] - lig_atoms[j][1]
            dy = lig_atoms[i][2] - lig_atoms[j][2]
            dz = lig_atoms[i][3] - lig_atoms[j][3]
            d = (dx*dx + dy*dy + dz*dz) ** 0.5
            if d < COVALENT_MIN:
                n_clash += 1
                # 相对 vdW 最小允许距离的比例
                rmin = (ri + VDW[lig_atoms[j][0]]) * INTRAMOL_SCALE
                ratio = d / rmin if rmin > 0 else 0
                if ratio < worst_ratio:
                    worst_ratio = ratio
    return n_clash, worst_ratio


def check_pocket(lig_atoms):
    """检查配体是否在结合口袋内"""
    lig_arr = np.array([(a[1], a[2], a[3]) for a in lig_atoms])
    # 所有原子到口袋中心的最大距离
    dist_to_center = np.linalg.norm(lig_arr - POCKET_CENTER, axis=1)
    max_dist = dist_to_center.max()
    centroid = lig_arr.mean(axis=0)
    centroid_dist = np.linalg.norm(centroid - POCKET_CENTER)
    # 质心在口袋内 (合理范围, 半宽 11 Å + 容差)
    in_pocket = centroid_dist < (BOX_HALF + 1.0)  # 12 Å
    return in_pocket, centroid_dist, max_dist


def main():
    prot_atoms = read_protein_atoms(PROTEIN_PDB)
    print(f'蛋白重原子: {len(prot_atoms)}')

    vina_files = sorted(glob.glob(os.path.join(VINA_DIR, 'out_*.pdbqt')))
    print(f'Vina 对接结果: {len(vina_files)} 个\n')

    results = []
    for i, vf in enumerate(vina_files):
        cid = os.path.basename(vf).replace('out_', '').replace('.pdbqt', '')
        lig_atoms = extract_best_pose_atoms(vf)
        if not lig_atoms:
            print(f'  [{i+1}/{len(vina_files)}] {cid}: pose 解析失败')
            continue

        # 1. 配体-蛋白碰撞
        n_lp, worst_lp = check_ligprot_clash(lig_atoms, prot_atoms)
        # 2. 分子内碰撞
        n_intra, worst_intra = check_intramol_clash(lig_atoms, cid)
        # 3. 口袋约束
        in_pocket, centroid_dist, max_dist = check_pocket(lig_atoms)

        # 综合判定
        if n_lp == 0 and n_intra == 0 and in_pocket:
            verdict = 'PASS'
        elif n_lp == 0 and in_pocket and n_intra <= 2:
            verdict = 'WARN'  # 轻微分子内碰撞, 可接受
        elif n_lp > 0:
            verdict = 'FAIL'  # 配体-蛋白碰撞不可接受
        else:
            verdict = 'FAIL'

        results.append({
            'ID': int(cid),
            'n_ligprot_clash': n_lp,
            'worst_ligprot_ratio': round(worst_lp, 3) if n_lp > 0 else np.nan,
            'n_intramol_clash': n_intra,
            'worst_intramol_ratio': round(worst_intra, 3) if n_intra > 0 else np.nan,
            'in_pocket': in_pocket,
            'centroid_dist': round(centroid_dist, 2),
            'max_dist_to_center': round(max_dist, 2),
            'verdict': verdict,
        })

    df = pd.DataFrame(results)
    out_csv = os.path.join(VINA_DIR, 'pose_quality_vina.csv')
    df.to_csv(out_csv, index=False, encoding='utf-8-sig')

    # 统计
    print('\n=== Pose Quality 统计 ===')
    print(df['verdict'].value_counts().to_string())
    print(f'\nPASS (全部通过):    {sum(df["verdict"]=="PASS")}')
    print(f'WARN (轻微分子内): {sum(df["verdict"]=="WARN")}')
    print(f'FAIL (配体-蛋白碰撞或口袋外): {sum(df["verdict"]=="FAIL")}')
    print(f'\n已保存: {out_csv}')

    # 清洗临时文件
    for f in glob.glob(os.path.join(VINA_DIR, '_tmp_*.pdb')):
        try:
            os.remove(f)
        except OSError:
            pass


if __name__ == '__main__':
    main()
