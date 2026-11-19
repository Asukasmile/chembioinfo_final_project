#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Vina 单分子 worker —— 供并行主脚本通过 subprocess 调用。

用法:
  python vina_worker.py <cid> <smiles>

独立完成: 配体准备 (meeko) + Vina 对接, 输出 out_<cid>.pdbqt
通过 stdout 打印结果 (JSON 一行)。
"""
import os
import sys
import json
import subprocess

PROJECT = os.path.dirname(os.path.abspath(__file__))
VINA = os.path.join(PROJECT, 'vina', 'vina_1.2.5_win.exe')
VINA_DIR = os.path.join(PROJECT, 'vina')

CENTER = (15.31, -2.87, 11.82)
BOX_SIZE = (22.0, 22.0, 22.0)
EXHAUSTIVENESS = 8   # 实测: 8 与 16 结合能几乎相同, 但快一倍
N_POSES = 9


def parse_score(out_pdbqt):
    if not os.path.exists(out_pdbqt):
        return None
    try:
        with open(out_pdbqt, encoding='utf-8', errors='ignore') as f:
            for line in f:
                if 'VINA RESULT' in line:
                    parts = line.split()
                    if len(parts) >= 4:
                        return float(parts[3])
    except Exception:
        return None
    return None


def prepare_ligand(cid, smiles):
    out = os.path.join(VINA_DIR, f'ligand_{cid}.pdbqt')
    if os.path.exists(out) and os.path.getsize(out) > 0:
        return out

    from rdkit import Chem
    from rdkit.Chem import AllChem
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = Chem.AddHs(mol)
    if AllChem.EmbedMolecule(mol, randomSeed=42) != 0:
        if AllChem.EmbedMolecule(mol, randomSeed=123) != 0:
            return None
    AllChem.MMFFOptimizeMolecule(mol)

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
        sys.stderr.write(f'[meeko 失败] {cid}: {e}\n')
    return None


def main():
    if len(sys.argv) < 3:
        sys.stderr.write('用法: python vina_worker.py <cid> <smiles>\n')
        sys.exit(1)

    cid = int(sys.argv[1])
    smiles = sys.argv[2]
    receptor_pdbqt = os.path.join(VINA_DIR, '4HYI_receptor.pdbqt')
    out_pdbqt = os.path.join(VINA_DIR, f'out_{cid}.pdbqt')
    out_log = os.path.join(VINA_DIR, f'log_{cid}.txt')

    # 断点续跑
    existing = parse_score(out_pdbqt)
    if existing is not None:
        print(json.dumps({'ID': cid, 'vina_score': existing, 'status': 'ok', 'note': 'cached'}))
        return

    lig_pdbqt = prepare_ligand(cid, smiles)
    if lig_pdbqt is None:
        print(json.dumps({'ID': cid, 'vina_score': None, 'status': 'prep_fail', 'note': ''}))
        return

    cmd = [
        VINA,
        '--receptor', receptor_pdbqt,
        '--ligand', lig_pdbqt,
        '--center_x', str(CENTER[0]),
        '--center_y', str(CENTER[1]),
        '--center_z', str(CENTER[2]),
        '--size_x', str(BOX_SIZE[0]),
        '--size_y', str(BOX_SIZE[1]),
        '--size_z', str(BOX_SIZE[2]),
        '--exhaustiveness', str(EXHAUSTIVENESS),
        '--num_modes', str(N_POSES),
        '--cpu', '2',
        '--out', out_pdbqt,
    ]
    try:
        with open(out_log, 'w') as logf:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
            logf.write(r.stdout)
            logf.write(r.stderr)
    except subprocess.TimeoutExpired:
        print(json.dumps({'ID': cid, 'vina_score': None, 'status': 'timeout', 'note': ''}))
        return
    except Exception as e:
        print(json.dumps({'ID': cid, 'vina_score': None, 'status': 'error', 'note': str(e)}))
        return

    score = parse_score(out_pdbqt)
    if r.returncode == 0 and score is not None:
        print(json.dumps({'ID': cid, 'vina_score': score, 'status': 'ok', 'note': 'docked'}))
    else:
        print(json.dumps({'ID': cid, 'vina_score': None, 'status': 'dock_fail', 'note': f'rc={r.returncode}'}))


if __name__ == '__main__':
    main()
