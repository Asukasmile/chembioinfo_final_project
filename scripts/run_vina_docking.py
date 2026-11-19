#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Vina 对接流水线 —— 受体与配体准备 + 对接。

使用 chem_rdkit 环境 (Python 3.11 + RDKit 2026.03.5)。

依赖:
  - vina_1.2.5_win.exe (已下载到 vina/)
  - meeko (配体 PDBQT 准备, SMILES->3D->PDBQT)
  - RDKit (结构处理 + Gasteiger 电荷)

流程:
  1. 受体准备: 4HYI_protein.pdb -> vina/4HYI_receptor.pdbqt (加 Gasteiger 电荷)
  2. 配体准备: 对每个分子 SMILES -> 3D 构象 -> meeko PDBQT
  3. 对接: 以晶体配体 1AO 结合位点为中心设定搜索盒子
  4. 输出每个分子的最佳 pose 结合能 (kcal/mol) 与 PDBQT

结合位点 (1AO 几何中心): (15.31, -2.87, 11.82)
盒子尺寸: 22 x 22 x 22 Å (覆盖晶体配体 10x6.4x6 并留余量)
"""
import os
import sys
import subprocess
import glob
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem

PROJECT = os.path.dirname(os.path.abspath(__file__))
VINA = os.path.join(PROJECT, 'vina', 'vina_1.2.5_win.exe')
VINA_DIR = os.path.join(PROJECT, 'vina')
PROTEIN = os.path.join(PROJECT, '4HYI_protein.pdb')

# 结合位点参数
CENTER = (15.31, -2.87, 11.82)   # 1AO 几何中心
BOX_SIZE = (22.0, 22.0, 22.0)    # 搜索盒子尺寸
EXHAUSTIVENESS = 16              # 搜索精度 (默认8, 16更细致)
N_POSES = 9                      # 输出 pose 数
N_CPU = 4                        # CPU 线程数


def prepare_receptor():
    """4HYI_protein.pdb -> PDBQT (Gasteiger 电荷)"""
    out = os.path.join(VINA_DIR, '4HYI_receptor.pdbqt')
    if os.path.exists(out):
        print(f'[已存在] 受体 PDBQT: {out}')
        return out

    # 用 RDKit 读取蛋白, 加 Gasteiger 电荷
    mol = Chem.MolFromPDBFile(PROTEIN, removeHs=True)
    if mol is None:
        raise RuntimeError('无法解析受体 PDB')

    # 加氢 + Gasteiger 电荷
    mol = Chem.AddHs(mol)
    AllChem.ComputeGasteigerCharges(mol)

    # 写 PDBQT (PDB 格式 + 每行末尾加电荷)
    with open(out, 'w') as f:
        f.write('REMARK  Receptor: 4HYI (CHEK1)\n')
        conf = mol.GetConformer()
        for atom in mol.GetAtoms():
            idx = atom.GetIdx()
            if atom.GetSymbol() == 'H':
                continue  # 对接通常去掉非极性氢, 这里先保留但meeko受体已去氢
            pos = conf.GetAtomPosition(idx)
            elem = atom.GetSymbol()
            chg = float(atom.GetProp('_GasteigerCharge')) if atom.HasProp('_GasteigerCharge') else 0.0
            # PDBQT 原子行: 与 PDB ATOM 相同, 但在末尾加电荷/类型
            line = (
                f'ATOM  {idx+1:5d} {elem.upper():>2s}   UNK A   1    '
                f'{pos.x:8.3f}{pos.y:8.3f}{pos.z:8.3f}'
                f'  1.00  0.00    {chg:6.3f} {elem.upper():>2s}  \n'
            )
            f.write(line)
        f.write('END\n')
    print(f'[OK] 受体 PDBQT: {out} ({mol.GetNumAtoms()} 原子)')
    return out


def prepare_ligand(cid, smiles):
    """SMILES -> 3D -> PDBQT (用 meeko)"""
    out = os.path.join(VINA_DIR, f'ligand_{cid}.pdbqt')
    if os.path.exists(out):
        return out

    # RDKit 生成 3D
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = Chem.AddHs(mol)
    if AllChem.EmbedMolecule(mol, randomSeed=42) != 0:
        # 重新尝试
        if AllChem.EmbedMolecule(mol, randomSeed=123) != 0:
            return None
    AllChem.MMFFOptimizeMolecule(mol)

    # meeko 准备 (直接用 RDKit mol 对象, 无需临时文件)
    try:
        from meeko import MoleculePreparation
        from meeko import PDBQTWriterLegacy
        preparator = MoleculePreparation()
        mol_setups = preparator.prepare(mol)
        for setup in mol_setups:
            pdbqt_string, is_ok, err = PDBQTWriterLegacy.write_string(setup)
            if is_ok:
                with open(out, 'w') as f:
                    f.write(pdbqt_string)
                return out
    except Exception as e:
        print(f'  [meeko 失败] {cid}: {e}')

    return None


def dock_ligand(cid, receptor_pdbqt, ligand_pdbqt, out_pdbqt, out_log):
    """对单个配体执行 Vina 对接"""
    cmd = [
        VINA,
        '--receptor', receptor_pdbqt,
        '--ligand', ligand_pdbqt,
        '--center_x', str(CENTER[0]),
        '--center_y', str(CENTER[1]),
        '--center_z', str(CENTER[2]),
        '--size_x', str(BOX_SIZE[0]),
        '--size_y', str(BOX_SIZE[1]),
        '--size_z', str(BOX_SIZE[2]),
        '--exhaustiveness', str(EXHAUSTIVENESS),
        '--num_modes', str(N_POSES),
        '--cpu', str(N_CPU),
        '--out', out_pdbqt,
    ]
    with open(out_log, 'w') as logf:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        logf.write(r.stdout)
        logf.write(r.stderr)
    return r.returncode


def parse_score(out_pdbqt):
    """从 Vina 输出 PDBQT 提取最佳结合能"""
    best = None
    if not os.path.exists(out_pdbqt):
        return None
    with open(out_pdbqt) as f:
        for line in f:
            if line.startswith('MODEL'):
                pass
            elif 'VINA RESULT' in line:
                # 格式: VINA RESULT:  -9.8  0.0  0.0
                parts = line.split()
                if len(parts) >= 4:
                    best = float(parts[3])
                    break
    return best


def main():
    os.makedirs(VINA_DIR, exist_ok=True)
    receptor_pdbqt = prepare_receptor()

    # 读取候选分子 (默认 top 100, 可传参)
    n_top = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    df = pd.read_csv(os.path.join(PROJECT, 'onionnet_only', 'onionnet_full_ranking.csv'))
    df.columns = [c.strip('\ufeff') for c in df.columns]
    df = df.sort_values('onionnet_pKd', ascending=False).head(n_top)

    print(f'\n=== 对接 top {n_top} 个分子 ===')

    results = []
    for i, (_, r) in enumerate(df.iterrows(), 1):
        cid = int(r['ID'])
        smiles = r['SMILES']
        pkd = r['onionnet_pKd']

        lig_pdbqt = prepare_ligand(cid, smiles)
        if lig_pdbqt is None:
            print(f'[{i}/{n_top}] {cid}: 配体准备失败')
            results.append({'ID': cid, 'onionnet_pKd': pkd,
                            'vina_score': None, 'status': 'prep_fail'})
            continue

        out_pdbqt = os.path.join(VINA_DIR, f'out_{cid}.pdbqt')
        out_log = os.path.join(VINA_DIR, f'log_{cid}.txt')

        # 断点续跑: 已有结果则跳过
        if os.path.exists(out_pdbqt):
            existing = parse_score(out_pdbqt)
            if existing is not None:
                print(f'[{i}/{n_top}] {cid}: 已存在 Vina {existing:.2f} kcal/mol (跳过)')
                results.append({'ID': cid, 'onionnet_pKd': pkd,
                                'vina_score': existing, 'status': 'ok'})
                continue

        rc = dock_ligand(cid, receptor_pdbqt, lig_pdbqt, out_pdbqt, out_log)
        score = parse_score(out_pdbqt)

        if rc == 0 and score is not None:
            print(f'[{i}/{n_top}] {cid}: pKd={pkd:.3f} -> Vina {score:.2f} kcal/mol')
            results.append({'ID': cid, 'onionnet_pKd': pkd,
                            'vina_score': score, 'status': 'ok'})
        else:
            print(f'[{i}/{n_top}] {cid}: 对接失败 (rc={rc})')
            results.append({'ID': cid, 'onionnet_pKd': pkd,
                            'vina_score': None, 'status': 'dock_fail'})

    # 保存结果
    res_df = pd.DataFrame(results)
    out_csv = os.path.join(VINA_DIR, f'vina_docking_top{n_top}.csv')
    res_df.to_csv(out_csv, index=False, encoding='utf-8-sig')
    print(f'\n=== 完成! 结果: {out_csv} ===')
    ok = res_df[res_df['status'] == 'ok']
    print(f'成功对接: {len(ok)}/{len(res_df)}')
    if len(ok) > 0:
        print(f'Vina 结合能范围: {ok["vina_score"].min():.2f} ~ {ok["vina_score"].max():.2f} kcal/mol')


if __name__ == '__main__':
    main()
