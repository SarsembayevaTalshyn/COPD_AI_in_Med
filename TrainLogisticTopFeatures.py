#!/usr/bin/env python3
# TrainLogisticTopFeatures.py — v2
# Сравнивает три набора фич: t-test(FDR), L1-logistic, RFE.
# Логистическая регрессия (balanced), отчёт по тесту, графики весов.

import os, json
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, balanced_accuracy_score

OUT_DIR = "./results_final/logreg_topfeatures"
os.makedirs(OUT_DIR, exist_ok=True)

# данные
meta = pd.read_csv("./embeddings/probe/embeddings_meta.csv")[["patient_id","split","label"]]
emb  = pd.read_csv("./embeddings/probe/embeddings.csv")
df   = emb.merge(meta, on="patient_id", how="inner")

# загрузим списки фич
lists = {}
ttest_path = "./results_final/top_features/top_by_ttest.csv"
l1_path    = "./results_final/top_features/top_by_l1logit.csv"
rfe_path   = "./results_final/top_features/top_by_rfe.csv"

if os.path.exists(ttest_path):
    lists["ttest_fdr"] = pd.read_csv(ttest_path)["feature"].head(20).tolist()
if os.path.exists(l1_path):
    lists["l1_logit"]  = pd.read_csv(l1_path, header=None).iloc[:,0].head(20).tolist() if "feature" not in pd.read_csv(l1_path, nrows=1).columns \
                         else pd.read_csv(l1_path)["feature"].head(20).tolist()
if os.path.exists(rfe_path):
    lists["rfe"]       = pd.read_csv(rfe_path)["feature"].head(20).tolist()

reports = {}

for name, feats in lists.items():
    feats = [f for f in feats if f in df.columns]  # защита
    X = df[feats].values.astype(np.float32)
    y = df["label"].astype(int).values
    split = df["split"].values

    tr = split == "train"; te = split == "test"
    sc = StandardScaler().fit(X[tr])
    Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
    ytr, yte = y[tr], y[te]

    clf = LogisticRegression(max_iter=5000, class_weight="balanced", solver="liblinear")
    clf.fit(Xtr, ytr)
    p_te = clf.predict_proba(Xte)[:,1]

    auc  = roc_auc_score(yte, p_te)
    ap   = average_precision_score(yte, p_te)
    yhat = (p_te >= 0.5).astype(int)
    f1   = f1_score(yte, yhat)
    bacc = balanced_accuracy_score(yte, yhat)

    # веса
    w = pd.Series(clf.coef_[0], index=feats).sort_values()
    plt.figure(figsize=(6,6)); w.plot(kind="barh"); plt.title(f"Weights — {name}")
    plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, f"weights_{name}.png"), dpi=150); plt.close()

    reports[name] = {"auc": float(auc), "ap": float(ap), "f1": float(f1), "bacc": float(bacc), "n_test": int(te.sum())}

json.dump(reports, open(os.path.join(OUT_DIR, "metrics.json"), "w"), indent=2)
print(json.dumps(reports, indent=2))
print("✓ Saved →", OUT_DIR)
