# -*- coding: utf-8 -*-
"""
从 ChEMBL 下载 CHEK1 (CHEMBL4630) 的活性数据
使用标准库 urllib 直接访问 ChEMBL REST API，绕开可能卡死的第三方库。

用法（任意有网络的环境）：
    python download_chembl.py
"""
import json
import csv
import time
import urllib.request
import urllib.parse

TARGET_ID = "CHEMBL4630"
OUT_CSV = "data/chembl4630_activities_raw.csv"
API_BASE = "https://www.ebi.ac.uk/chembl/api/data/activity"


def fetch_page(target_id, limit, offset):
    params = urllib.parse.urlencode({
        "target_chembl_id": target_id,
        "limit": limit,
        "offset": offset,
        "format": "json",
    })
    url = f"{API_BASE}?{params}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read().decode("utf-8"))


def fetch_all(target_id, page_size=100, sleep=0.3):
    rows = []
    offset = 0
    total = None
    while True:
        for attempt in range(6):
            try:
                data = fetch_page(target_id, page_size, offset)
                break
            except Exception as e:
                wait = 5 * (attempt + 1)
                print(f"  第{attempt+1}次重试(offset={offset})，等待{wait}s：{e}")
                time.sleep(wait)
        else:
            print(f"  offset={offset} 连续失败，终止。")
            break

        if total is None:
            total = data.get("page_meta", {}).get("total_count")
            print(f"  总记录数：{total}")

        batch = data.get("activities", [])
        if not batch:
            break
        rows.extend(batch)
        offset += page_size
        print(f"  已获取 {len(rows)} 条 ...")
        if len(batch) < page_size:
            break
        time.sleep(sleep)
    return rows


def main():
    print(f"正在下载 {TARGET_ID} 的活性数据 ...")
    rows = fetch_all(TARGET_ID)
    print(f"完成，共 {len(rows)} 条记录")

    fields = [
        "activity_id", "molecule_chembl_id", "canonical_smiles",
        "standard_type", "standard_value", "standard_units",
        "standard_relation", "pchembl_value", "target_chembl_id",
        "target_organism", "assay_type", "bao_label",
        "document_chembl_id", "document_year",
    ]
    with open(OUT_CSV, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"已保存到 {OUT_CSV}")


if __name__ == "__main__":
    main()
