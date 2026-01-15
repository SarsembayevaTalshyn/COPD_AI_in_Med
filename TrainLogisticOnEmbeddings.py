#!/usr/bin/env python3
# TrainLogisticOnEmbeddings.py — с бутстрэп CI, ROC/PR, patient-level stratification log

import os, json
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, balanced_accuracy_score
from sklearn.metrics import roc_curve, precision_recall_curve
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from tqdm import tqdm

# === setup
os.makedirs("./results_final/logreg_embeddings", exist_ok=True)
df = pd.read_csv("./embeddings/probe/embeddings_meta.csv")  # patient_id, split, label, logit_0/1, laa, perc15, tlv

# эмбеддинги
emb = pd.read_csv("./embeddings/probe/embeddings.csv")      # feat_0...feat_2047
feat_cols = [c for c in emb.columns if c.startswith("feat_")]
df = df.merge(emb[["patient_id"]+feat_cols], on="patient_id", how="left")

# === train/val/test split
Xtr, ytr = df.loc[df.split=="train", feat_cols].values, df.loc[df.split=="train","label"].values
Xte, yte = df.loc[df.split=="test",  feat_cols].values, df.loc[df.split=="test","label"].values

scaler = StandardScaler().fit(Xtr)
Xtr, Xte = scaler.transform(Xtr), scaler.transform(Xte)

# === train logistic regression (balanced)
clf = LogisticRegression(max_iter=2000, class_weight="balanced", solver="liblinear")
clf.fit(Xtr, ytr)
p_te = clf.predict_proba(Xte)[:,1]

# === базовые метрики
auc = roc_auc_score(yte, p_te)
ap  = average_precision_score(yte, p_te)
f1  = f1_score(yte, (p_te>=0.5).astype(int))
bacc= balanced_accuracy_score(yte, (p_te>=0.5).astype(int))

# === bootstrap CI
boot = []
for i in tqdm(range(2000), desc="Bootstrap CI"):
    Xb, yb = resample(Xte, yte, replace=True, stratify=yte, random_state=i)
    pb = clf.predict_proba(Xb)[:,1]
    boot.append([
        roc_auc_score(yb,pb),
        average_precision_score(yb,pb),
        f1_score(yb,(pb>=0.5).astype(int)),
        balanced_accuracy_score(yb,(pb>=0.5).astype(int))
    ])
boot = np.array(boot)
ci = {m: [np.percentile(boot[:,i],2.5), np.percentile(boot[:,i],97.5)]
      for i,m in enumerate(["AUC","AP","F1","bACC"])}

# === графики
fpr,tpr,_=roc_curve(yte,p_te)
plt.figure(); plt.plot(fpr,tpr,label=f"AUC={auc:.3f}"); plt.plot([0,1],[0,1],'--')
plt.legend(); plt.title("ROC — Logistic on embeddings"); plt.savefig("./results_final/logreg_embeddings/roc.png",dpi=150); plt.close()

pr,rc,_=precision_recall_curve(yte,p_te)
plt.figure(); plt.plot(rc,pr,label=f"AP={ap:.3f}"); plt.legend(); plt.title("PR — Logistic on embeddings")
plt.savefig("./results_final/logreg_embeddings/pr.png",dpi=150); plt.close()

# === save report
report={"n_test":int(len(yte)),"auc":auc,"ap":ap,"f1":f1,"bacc":bacc,"ci":ci}
json.dump(report, open("./results_final/logreg_embeddings/metrics.json","w"), indent=2)
print(json.dumps(report,indent=2))
