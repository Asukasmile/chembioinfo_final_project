# -*- coding: utf-8 -*-
"""
CHEK1 (CHK1) 抑制剂活性预测 —— 配体基机器学习模型训练
==========================================================
流程：
  1. 读取清洗后的训练集 (data/chembl4630_train.csv)
  2. RDKit 生成分子指纹 (Morgan/ECFP4 + MACCS)
  3. 划分训练/测试集（分层，避免数据泄漏）
  4. 训练 Random Forest + SVM 分类器（5 折交叉验证 + 独立测试集评估）
  5. 输出评估指标（准确率、AUC、混淆矩阵、F1 等）并保存模型

运行环境：conda env chem_rdkit (RDKit + scikit-learn)
"""

import os
import json
import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs, MACCSkeys
from rdkit import RDLogger

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, roc_auc_score, f1_score,
                             matthews_corrcoef, confusion_matrix,
                             classification_report, roc_curve)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

RDLogger.DisableLog('rdApp.*')

# ----------------------------- 参数 -----------------------------
DATA_PATH = 'data/chembl4630_train.csv'
OUT_DIR = 'models'
os.makedirs(OUT_DIR, exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.2
N_CV = 5

# ----------------------------- 1. 读取数据 -----------------------------
df = pd.read_csv(DATA_PATH, encoding='utf-8-sig')
print(f'[1] 读取训练集: {len(df)} 个化合物')
print(df['label'].value_counts().to_string())

smiles_list = df['canonical_smiles'].tolist()
y = df['label'].values.astype(int)

# ----------------------------- 2. 生成分子指纹 -----------------------------
def smiles_to_fp(smiles, radius=2, nbits=2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=nbits)
    return np.array(fp, dtype=np.float32)

def smiles_to_maccs(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return np.array(MACCSkeys.GenMACCSKeys(mol), dtype=np.float32)

print('\n[2] 生成分子指纹 (ECFP4 / Morgan r=2, 2048 bits) ...')
X_morgan = []
valid_idx = []
for i, s in enumerate(smiles_list):
    fp = smiles_to_fp(s)
    if fp is not None:
        X_morgan.append(fp)
        valid_idx.append(i)
    else:
        print(f'  [警告] 无法解析 SMILES (第 {i} 行): {s[:40]}')

X_morgan = np.array(X_morgan)
y_valid = y[valid_idx]
print(f'  有效化合物: {len(valid_idx)} / {len(smiles_list)}')
print(f'  特征矩阵形状: {X_morgan.shape}')

# ----------------------------- 3. 划分训练/测试集 -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_morgan, y_valid, test_size=TEST_SIZE,
    stratify=y_valid, random_state=RANDOM_STATE
)
print(f'\n[3] 划分数据集: 训练 {len(X_train)} / 测试 {len(X_test)} (分层)')
print(f'  训练集标签分布: {dict(zip(*np.unique(y_train, return_counts=True)))}')
print(f'  测试集标签分布: {dict(zip(*np.unique(y_test, return_counts=True)))}')

# ----------------------------- 4. 训练 Random Forest -----------------------------
print('\n[4] 训练 Random Forest ...')
rf = RandomForestClassifier(
    n_estimators=500,
    max_features='sqrt',
    min_samples_split=2,
    class_weight='balanced',   # 处理类别不平衡
    n_jobs=-1,
    random_state=RANDOM_STATE
)

# 5 折交叉验证
cv = StratifiedKFold(n_splits=N_CV, shuffle=True, random_state=RANDOM_STATE)
cv_auc_rf = cross_val_score(rf, X_train, y_train, cv=cv, scoring='roc_auc', n_jobs=-1)
cv_acc_rf = cross_val_score(rf, X_train, y_train, cv=cv, scoring='accuracy', n_jobs=-1)
print(f'  CV AUC (均值±std): {cv_auc_rf.mean():.4f} ± {cv_auc_rf.std():.4f}')
print(f'  CV ACC (均值±std): {cv_acc_rf.mean():.4f} ± {cv_acc_rf.std():.4f}')

rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
y_prob_rf = rf.predict_proba(X_test)[:, 1]

# ----------------------------- 5. 训练 SVM (RBF) -----------------------------
print('\n[5] 训练 SVM (RBF 核) ...')
svm = Pipeline([
    ('scaler', StandardScaler(with_mean=False)),
    ('svc', SVC(kernel='rbf', C=1.0, gamma='scale', probability=True,
                class_weight='balanced', random_state=RANDOM_STATE))
])

cv_auc_svm = cross_val_score(svm, X_train, y_train, cv=cv, scoring='roc_auc', n_jobs=-1)
cv_acc_svm = cross_val_score(svm, X_train, y_train, cv=cv, scoring='accuracy', n_jobs=-1)
print(f'  CV AUC (均值±std): {cv_auc_svm.mean():.4f} ± {cv_auc_svm.std():.4f}')
print(f'  CV ACC (均值±std): {cv_acc_svm.mean():.4f} ± {cv_acc_svm.std():.4f}')

svm.fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)
y_prob_svm = svm.predict_proba(X_test)[:, 1]

# ----------------------------- 6. 评估与汇总 -----------------------------
def evaluate(name, y_true, y_pred, y_prob):
    auc = roc_auc_score(y_true, y_prob)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    print(f'\n===== {name} (独立测试集) =====')
    print(f'  Accuracy: {acc:.4f}')
    print(f'  AUC:      {auc:.4f}')
    print(f'  F1:       {f1:.4f}')
    print(f'  MCC:      {mcc:.4f}')
    print(f'  混淆矩阵 (行=真实, 列=预测):\n{cm}')
    return {'accuracy': acc, 'auc': auc, 'f1': f1, 'mcc': mcc,
            'confusion_matrix': cm.tolist()}

results = {}
results['RandomForest'] = evaluate('Random Forest', y_test, y_pred_rf, y_prob_rf)
results['SVM'] = evaluate('SVM (RBF)', y_test, y_pred_svm, y_prob_svm)

# ----------------------------- 7. 特征重要性 (RF) -----------------------------
importances = rf.feature_importances_
top_idx = np.argsort(importances)[::-1][:20]
print('\n[7] Random Forest Top 20 重要指纹位 (Morgan bit index):')
for rank, idx in enumerate(top_idx, 1):
    print(f'  {rank:2d}. bit {idx:4d}  importance={importances[idx]:.4f}')

# ----------------------------- 8. 保存模型与评估结果 -----------------------------
import joblib
joblib.dump(rf, os.path.join(OUT_DIR, 'rf_chk1.joblib'))
joblib.dump(svm, os.path.join(OUT_DIR, 'svm_chk1.joblib'))

summary = {
    'n_compounds': int(len(valid_idx)),
    'n_train': int(len(X_train)),
    'n_test': int(len(X_test)),
    'n_active': int((y_valid == 1).sum()),
    'n_inactive': int((y_valid == 0).sum()),
    'fingerprint': 'Morgan_r2_2048bit',
    'random_forest': results['RandomForest'],
    'svm': results['SVM'],
}
with open(os.path.join(OUT_DIR, 'ml_summary.json'), 'w', encoding='utf-8') as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

print(f'\n✅ 完成。模型与评估结果已保存到 {OUT_DIR}/')
print(f'   - rf_chk1.joblib  (Random Forest)')
print(f'   - svm_chk1.joblib (SVM)')
print(f'   - ml_summary.json (评估指标)')
