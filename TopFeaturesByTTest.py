#!/usr/bin/env python3
# TopFeaturesByTTest.py — v2
# t-test + FDR/Bonferroni, и КОРРЕКТНЫЙ "lasso" для классификации: LogisticRegressionCV (L1)
# также RFE на логистике. Масштабируем признаки → нет предупреждений о сходимости.

import os, json
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.feature_selection import RFE

OUT_DIR = "./results_final/top_features"
os.makedirs(OUT_DIR, exist_ok=True)

# === данные
emb = pd.read_csv("./embeddings/probe/embeddings.csv")               # feat_*
meta = pd.read_csv("./embeddings/probe/embeddings_meta.csv")[["patient_id","label"]]
df = emb.merge(meta, on="patient_id", how="left").dropna(subset=["label"])
feat_cols = [c for c in df.columns if c.startswith("feat_")]
X = df[feat_cols].values.astype(np.float32)
y = df["label"].astype(int).values

# Стандартизация (очень важна для L1!)
sc = StandardScaler().fit(X)
Xz = sc.transform(X)

# === 1) Однофакторный t-test + FDR/Bonferroni
pvals = np.zeros(len(feat_cols), dtype=float)
for i, f in enumerate(feat_cols):
    x0 = Xz[y==0, i]
    x1 = Xz[y==1, i]
    _, p = ttest_ind(x0, x1, equal_var=False)
    pvals[i] = p

res = pd.DataFrame({"feature": feat_cols, "p": pvals})
# Коррекции кратных сравнений
res["p_fdr"] = multipletests(res["p"].values, alpha=0.05, method="fdr_bh")[1]
res["p_bonf"] = np.minimum(res["p"].values * len(res), 1.0)
res = res.sort_values("p_fdr").reset_index(drop=True)
res.to_csv(os.path.join(OUT_DIR, "all_features_ttest.csv"), index=False)

top_ttest = res.head(30)
top_ttest.to_csv(os.path.join(OUT_DIR, "top_by_ttest.csv"), index=False)

# Визуализация распределения значимостей
plt.figure(figsize=(6,4))
plt.hist(-np.log10(res["p"].clip(lower=1e-300)), bins=60)
plt.xlabel("-log10(p)"); plt.ylabel("Count")
plt.title("Distribution of p-values across features")
plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, "pvals_hist.png"), dpi=150); plt.close()

# === 2) "LASSO" для классификации: LogisticRegressionCV с L1
# Cs — сетка по регуляризации, class_weight='balanced' из-за дисбаланса, метрика — ROC AUC
cs = np.logspace(-3, 3, 15)
lrcv = LogisticRegressionCV(
    Cs=cs, cv=5, penalty="l1", solver="liblinear",
    scoring="roc_auc", class_weight="balanced", max_iter=5000, n_jobs=-1, refit=True
).fit(Xz, y)

coef = pd.Series(lrcv.coef_.ravel(), index=feat_cols)
sel_l1 = (coef != 0)
top_l1 = coef.abs().sort_values(ascending=False).head(30)
top_l1.to_csv(os.path.join(OUT_DIR, "top_by_l1logit.csv"))

# Сохраним также полный вектор коэфов и лучший C
pd.DataFrame({"feature": feat_cols, "coef": coef.values}).to_csv(os.path.join(OUT_DIR, "l1logit_all_coefs.csv"), index=False)
json.dump({"best_C": float(lrcv.C_[0])}, open(os.path.join(OUT_DIR, "l1logit_info.json"), "w"), indent=2)

# Картинка: веса L1-логистики по топ-фичам
plt.figure(figsize=(6,6))
top_l1.sort_values().plot(kind="barh")
plt.title("L1-logistic coefficients (top)")
plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, "l1logit_top_weights.png"), dpi=150); plt.close()

# === 3) Альтернатива: RFE на логистике (balanced)
base = LogisticRegression(max_iter=5000, class_weight="balanced", solver="liblinear")
rfe = RFE(base, n_features_to_select=30).fit(Xz, y)
mask = rfe.support_
top_rfe = [f for f, m in zip(feat_cols, mask) if m]
pd.Series(top_rfe, name="feature").to_csv(os.path.join(OUT_DIR, "top_by_rfe.csv"), index=False)

print("✓ Top features saved →", OUT_DIR)
