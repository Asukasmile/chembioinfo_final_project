#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
生成最终 lead 的复合物 PDB（Vina pose + 4HYI 蛋白）+ PyMOL 一键高亮脚本。

输出:
  complexes_vina/CHEK1_<id>_complex.pdb   复合物（蛋白链A + 配体链B）
  complexes_vina/highlight_leads.pml       PyMOL 脚本：高亮 hinge/gatekeeper/深口袋残基
"""
import os
import pandas as pd

PROJECT = os.path.dirname(os.path.abspath(__file__))
VINA_DIR = os.path.join(PROJECT, 'vina')
PROTEIN = os.path.join(PROJECT, '4HYI_protein.pdb')
OUTDIR = os.path.join(PROJECT, 'complexes_vina')
os.makedirs(OUTDIR, exist_ok=True)

# 关键残基
HINGE = [85, 87]          # Glu85, Cys87
GATEKEEPER = [15]          # Leu15
INTERIOR = [38, 55, 68, 84, 147, 148]  # 内部深口袋
RIBOSE = [91, 134, 135, 137]

# 读最终 lead
leads = pd.read_csv(os.path.join(VINA_DIR, 'final_leads.csv'))
leads.columns = [c.strip('\ufeff') for c in leads.columns]
lead_ids = leads['ID'].tolist()

# 读蛋白 ATOM 行
protein_lines = []
for line in open(PROTEIN, encoding='utf-8', errors='ignore'):
    if line.startswith('ATOM'):
        protein_lines.append(line.rstrip('\n'))

print(f'蛋白原子: {len(protein_lines)}')


def extract_ligand_from_pdbqt(pdbqt_path):
    """从 Vina 输出提取 MODEL 1 配体原子，转成 PDB ATOM 行"""
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
            x = float(s[30:38]); y = float(s[38:46]); z = float(s[46:54])
            atoms.append((elem, x, y, z))
    return atoms


pdb_lines = {}
for cid in lead_ids:
    vf = os.path.join(VINA_DIR, f'out_{cid}.pdbqt')
    if not os.path.exists(vf):
        print(f'[跳过] {cid}: 无 out_{cid}.pdbqt')
        continue
    lig_atoms = extract_ligand_from_pdbqt(vf)
    if not lig_atoms:
        print(f'[跳过] {cid}: pose 为空')
        continue

    out = os.path.join(OUTDIR, f'CHEK1_{cid}_complex.pdb')
    with open(out, 'w') as f:
        f.write(f'REMARK  Complex: CHEK1 4HYI + ligand {cid} (Vina pose)\n')
        f.write(f'REMARK  Protein chain A, Ligand chain B (resname LIG)\n')
        for l in protein_lines:
            f.write(l + '\n')
        f.write('TER\n')
        for i, (elem, x, y, z) in enumerate(lig_atoms, 1):
            name = f'{elem.upper():>2}'
            f.write(f'HETATM{i:5d} {name:^4s} LIG B   1    '
                    f'{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          '
                    f'{elem.upper():>2s}  \n')
        f.write('END\n')
    pdb_lines[cid] = out
    print(f'[OK] {cid}: {len(lig_atoms)} 配体原子 -> {os.path.basename(out)}')

# 生成 PyMOL 高亮脚本
pml_path = os.path.join(OUTDIR, 'highlight_leads.pml')
with open(pml_path, 'w') as f:
    f.write('# CHEK1 lead 复合物 PyMOL 一键高亮脚本\n')
    f.write('# 用法: PyMOL -> File -> Run Script -> 选择本文件\n')
    f.write('# 或命令行: @highlight_leads.pml\n\n')
    f.write('reinitialize\n')
    f.write('bg_color white\n')

    for cid in lead_ids:
        f.write(f'\n# ===== Lead {cid} =====\n')
        f.write(f'load {cid}_complex.pdb\n')
        f.write(f'remove solvent\n')
        f.write(f'hide everything\n')
        f.write(f'show cartoon\n')
        f.write(f'color grey80, all\n')
        # 配体
        f.write(f'show sticks, resn LIG\n')
        f.write(f'color yellow, resn LIG\n')
        f.write(f'util.cbac resn LIG\n')
        # hinge 残基
        f.write(f'# hinge: Glu85, Cys87\n')
        f.write(f'select hinge, (chain A and resi 85+87)\n')
        f.write(f'show sticks, hinge\n')
        f.write(f'color red, hinge\n')
        # gatekeeper
        f.write(f'# gatekeeper: Leu15\n')
        f.write(f'select gatekeeper, (chain A and resi 15)\n')
        f.write(f'show sticks, gatekeeper\n')
        f.write(f'color orange, gatekeeper\n')
        # 内部深口袋
        f.write(f'# interior pocket\n')
        f.write(f'select interior, (chain A and resi 38+55+68+84+147+148)\n')
        f.write(f'show sticks, interior\n')
        f.write(f'color cyan, interior\n')
        # 核糖口袋
        f.write(f'# ribose pocket\n')
        f.write(f'select ribose, (chain A and resi 91+134+135+137)\n')
        f.write(f'show sticks, ribose\n')
        f.write(f'color magenta, ribose\n')
        # 显示氢键
        f.write(f'distance hbond_hinge, resn LIG, (chain A and resi 85+87), 3.5\n')
        f.write(f'hide labels\n')
        f.write(f'zoom resn LIG, 8\n')

    f.write('\n# 图例:\n')
    f.write('# 黄色 = 配体; 红色 = hinge(Glu85/Cys87); 橙色 = gatekeeper(Leu15)\n')
    f.write('# 青色 = 内部深口袋; 品红 = 核糖口袋; 虚线 = 氢键(<3.5A)\n')

print(f'\nPyMOL 脚本: {pml_path}')
print(f'共生成 {len(pdb_lines)} 个复合物 PDB')
print(f'\n提示: 将 complexes_vina 目录下所有 .pdb 和 .pml 复制到同一目录，在 PyMOL 里 run highlight_leads.pml')
