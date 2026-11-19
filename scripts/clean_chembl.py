# -*- coding: utf-8 -*-
"""
清洗 ChEMBL CHEK1 (CHEMBL4630) 原始活性数据 -> 标准 active/inactive 训练集（增强版）

在 v1 基础上，额外利用「单点筛选」定性数据补充 inactive 样本，
以缓解 ChEMBL 天然阳性偏倚导致的类别严重不平衡。

标签来源：
  A) 浓度型活性 (IC50/Ki/Kd/EC50, relation='=')：
       pIC50 >= 6  -> active
       pIC50 <= 5  -> inactive
  B) 单点抑制率 (Inhibition %, 单位=%)：
       value <= 10  -> inactive（在测试浓度下几乎不抑制）
  C) 残留活性 (Residual Activity %, 单位=%)：
       value >= 90  -> inactive（几乎没被抑制）

规则：
  - 同一化合物若同时有 A 类和 B/C 类信号，优先采用 A 类（浓度型更可靠）
  - 同一化合物多条同类型记录取中位数
  - 输出最终训练集

用法：
    python clean_chembl.py
"""
import math
import pandas as pd

RAW = "data/chembl4630_activities_raw.csv"
OUT = "data/chembl4630_train.csv"
OUT_ALL = "data/chembl4630_labeled_all.csv"

CONC_TYPES = {"IC50", "Ki", "Kd", "EC50"}
ACTIVE_PIC50 = 6.0
INACTIVE_PIC50 = 5.0


def to_nm(v, u):
    if u == "uM":
        return v * 1000.0
    if u == "mM":
        return v * 1e6
    if u == "M":
        return v * 1e9
    return v


def main():
    df = pd.read_csv(RAW, encoding="utf-8-sig")
    print(f"原始记录数: {len(df)}")

    # ---------- A 类：浓度型活性 ----------
    dA = df[df["standard_type"].isin(CONC_TYPES)].copy()
    dA = dA[dA["standard_relation"] == "="]
    dA = dA[dA["standard_value"].notna()]
    dA = dA[dA["standard_units"].isin(["nM", "uM", "mM", "M"])]
    dA["value_nM"] = dA.apply(lambda r: to_nm(r["standard_value"], r["standard_units"]), axis=1)
    dA["pIC50"] = dA["value_nM"].apply(lambda x: 9 - math.log10(x) if x > 0 else None)
    dA = dA[dA["pIC50"].notna()]

    grpA = dA.groupby("molecule_chembl_id").agg(
        canonical_smiles=("canonical_smiles", "first"),
        pIC50_median=("pIC50", "median"),
        n_A=("activity_id", "count"),
    ).reset_index()
    print(f"A类浓度型活性 -> 唯一化合物: {len(grpA)}")

    # ---------- B 类：Inhibition % ----------
    dB = df[(df["standard_type"] == "Inhibition") & (df["standard_units"] == "%")].copy()
    dB = dB[dB["standard_value"].notna()]
    dB["inh_pct"] = dB["standard_value"].astype(float)
    dB = dB[dB["inh_pct"] <= 10]  # 抑制率 <= 10% 视为 inactive
    grpB = dB.groupby("molecule_chembl_id").agg(
        canonical_smiles=("canonical_smiles", "first"),
        inh_pct_min=("inh_pct", "min"),
        n_B=("activity_id", "count"),
    ).reset_index()
    print(f"B类 Inhibition<=10% -> 唯一化合物: {len(grpB)}")

    # ---------- C 类：Residual Activity % ----------
    dC = df[df["standard_type"].isin(["Residual Activity", "Residual activity"]) &
            (df["standard_units"] == "%")].copy()
    dC = dC[dC["standard_value"].notna()]
    dC["res_pct"] = dC["standard_value"].astype(float)
    dC = dC[dC["res_pct"] >= 90]  # 残留活性 >= 90% 视为 inactive
    grpC = dC.groupby("molecule_chembl_id").agg(
        canonical_smiles=("canonical_smiles", "first"),
        res_pct_max=("res_pct", "max"),
        n_C=("activity_id", "count"),
    ).reset_index()
    print(f"C类 ResidualActivity>=90% -> 唯一化合物: {len(grpC)}")

    # ---------- 合并 ----------
    # 以 A 类为准，B/C 类补充 inactive
    merged = grpA[["molecule_chembl_id", "canonical_smiles", "pIC50_median"]].copy()

    def label_from_A(p):
        if p >= ACTIVE_PIC50:
            return 1
        if p <= INACTIVE_PIC50:
            return 0
        return None

    merged["label"] = merged["pIC50_median"].apply(label_from_A)
    merged["evidence"] = "conc"

    # B/C 类补充：仅当该化合物不在 A 类中时
    a_ids = set(merged["molecule_chembl_id"])
    extra_rows = []
    for _, r in grpB.iterrows():
        if r["molecule_chembl_id"] not in a_ids:
            extra_rows.append({
                "molecule_chembl_id": r["molecule_chembl_id"],
                "canonical_smiles": r["canonical_smiles"],
                "pIC50_median": None,
                "label": 0,
                "evidence": "inh<=10%",
            })
    for _, r in grpC.iterrows():
        if r["molecule_chembl_id"] not in a_ids:
            extra_rows.append({
                "molecule_chembl_id": r["molecule_chembl_id"],
                "canonical_smiles": r["canonical_smiles"],
                "pIC50_median": None,
                "label": 0,
                "evidence": "res>=90%",
            })
    if extra_rows:
        merged = pd.concat([merged, pd.DataFrame(extra_rows)], ignore_index=True)

    # 去重（同一 id 可能同时被 B 和 C 命中）
    merged = merged.drop_duplicates(subset="molecule_chembl_id", keep="first")

    # 剔除灰色区（label 为 None）
    merged = merged[merged["label"].notna()].copy()
    merged["label"] = merged["label"].astype(int)
    merged = merged.dropna(subset=["canonical_smiles"])

    n_active = (merged["label"] == 1).sum()
    n_inactive = (merged["label"] == 0).sum()
    print(f"\n=== 最终训练集 ===")
    print(f"总化合物数: {len(merged)}")
    print(f"  active:   {n_active}")
    print(f"  inactive: {n_inactive}")
    print(f"  比例 active:inactive = {n_active}:{n_inactive}")
    print(f"\n标签证据来源分布:")
    print(merged["evidence"].value_counts())

    cols = ["molecule_chembl_id", "canonical_smiles", "pIC50_median", "evidence", "label"]
    merged[cols].to_csv(OUT, index=False, encoding="utf-8-sig")
    print(f"\n训练集已保存: {OUT}")


if __name__ == "__main__":
    main()
