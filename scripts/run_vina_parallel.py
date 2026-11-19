#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Vina 并行对接主脚本 —— 24 并发 subprocess 对接 931 个分子。

方案: 用 ThreadPoolExecutor 并发拉起多个 vina_worker.py 子进程。
每个子进程独立做「配体准备 + Vina 对接」, 天然并行, 无 multiprocessing pickle 问题。

断点续跑: 已有 out_<id>.pdbqt 且能解析分数则跳过。
"""
import os
import sys
import json
import subprocess
import concurrent.futures as cf
import pandas as pd

PROJECT = os.path.dirname(os.path.abspath(__file__))
VINA_DIR = os.path.join(PROJECT, 'vina')
LIST_CSV = os.path.join(VINA_DIR, 'vina_931_list.csv')
WORKER = os.path.join(PROJECT, 'vina_worker.py')
PYTHON = sys.executable  # 当前环境的 python

N_WORKERS = 12  # 并发数 (每 worker Vina --cpu 2, 共约 24 核)


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


def run_one(cid, smiles):
    """启动一个 vina_worker 子进程, 返回结果字典"""
    out_pdbqt = os.path.join(VINA_DIR, f'out_{cid}.pdbqt')
    existing = parse_score(out_pdbqt)
    if existing is not None:
        return {'ID': cid, 'vina_score': existing, 'status': 'ok', 'note': 'cached'}

    try:
        r = subprocess.run(
            [PYTHON, WORKER, str(cid), smiles],
            capture_output=True, text=True, timeout=1000,
        )
        # worker 最后一行是 JSON 结果
        lines = [l for l in r.stdout.strip().split('\n') if l.strip()]
        if lines:
            return json.loads(lines[-1])
        return {'ID': cid, 'vina_score': None, 'status': 'error', 'note': 'no output'}
    except subprocess.TimeoutExpired:
        return {'ID': cid, 'vina_score': None, 'status': 'timeout', 'note': ''}
    except Exception as e:
        return {'ID': cid, 'vina_score': None, 'status': 'error', 'note': str(e)}


def main():
    os.makedirs(VINA_DIR, exist_ok=True)

    receptor_pdbqt = os.path.join(VINA_DIR, '4HYI_receptor.pdbqt')
    if not os.path.exists(receptor_pdbqt):
        print('错误: 受体 PDBQT 不存在')
        sys.exit(1)

    df = pd.read_csv(LIST_CSV)
    df.columns = [c.strip('\ufeff') for c in df.columns]
    df['ID'] = df['ID'].astype(int)

    tasks = [(int(r['ID']), r['SMILES']) for _, r in df.iterrows()]
    total = len(tasks)
    done = sum(1 for cid, _ in tasks if parse_score(os.path.join(VINA_DIR, f'out_{cid}.pdbqt')) is not None)

    print(f'=== {N_WORKERS} 并发 subprocess 对接 {total} 个分子 ===', flush=True)
    print(f'已有结果(跳过): {done}, 待对接: {total - done}', flush=True)
    print(flush=True)

    results = []
    n_done = 0
    with cf.ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
        futs = {ex.submit(run_one, cid, smi): cid for cid, smi in tasks}
        for fut in cf.as_completed(futs):
            cid = futs[fut]
            try:
                res = fut.result()
            except Exception as e:
                res = {'ID': cid, 'vina_score': None, 'status': 'error', 'note': str(e)}
            results.append(res)
            n_done += 1
            if n_done % 20 == 0 or n_done == total:
                n_ok = sum(1 for r in results if r['status'] == 'ok')
                print(f'  进度 {n_done}/{total} (成功 {n_ok})', flush=True)

    res_df = pd.DataFrame(results)
    out_csv = os.path.join(VINA_DIR, 'vina_docking_931.csv')
    res_df.to_csv(out_csv, index=False, encoding='utf-8-sig')

    print(f'\n=== 完成! 结果: {out_csv} ===', flush=True)
    print(res_df['status'].value_counts().to_string(), flush=True)
    ok = res_df[res_df['status'] == 'ok']
    if len(ok) > 0:
        print(f'\n成功对接: {len(ok)}/{len(res_df)}', flush=True)
        print(f'Vina 结合能范围: {ok["vina_score"].min():.2f} ~ {ok["vina_score"].max():.2f} kcal/mol', flush=True)


if __name__ == '__main__':
    main()
