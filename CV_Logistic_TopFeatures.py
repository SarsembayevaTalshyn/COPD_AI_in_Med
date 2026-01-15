import numpy as np, pandas as pd, json
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, balanced_accuracy_score

meta = pd.read_csv("./embeddings/probe/embeddings_meta.csv")[["patient_id","label","split"]]
emb  = pd.read_csv("./embeddings/probe/embeddings.csv")
df   = emb.merge(meta, on="patient_id", how="inner")
top  = pd.read_csv("./results_final/top_features/top_by_ttest.csv")["feature"].head(20).tolist()
X, y = df[top].values.astype("float32"), df["label"].astype(int).values

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
rows=[]
for k,(tr,te) in enumerate(skf.split(X,y),1):
    sc = StandardScaler().fit(X[tr])
    Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
    ytr, yte = y[tr], y[te]
    clf = LogisticRegression(max_iter=5000, class_weight="balanced", solver="liblinear").fit(Xtr, ytr)
    p   = clf.predict_proba(Xte)[:,1]
    yhat= (p>=0.5).astype(int)
    rows.append({
        "fold":k,
        "AUC": roc_auc_score(yte,p),
        "AP":  average_precision_score(yte,p),
        "F1":  f1_score(yte,yhat),
        "bACC":balanced_accuracy_score(yte,yhat)
    })
res = pd.DataFrame(rows)
summary = res.agg(["mean","std"]).to_dict()
print(json.dumps(summary, indent=2))
res.to_csv("./results_final/logreg_topfeatures/cv5_metrics.csv", index=False)
