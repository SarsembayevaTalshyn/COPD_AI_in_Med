#!/usr/bin/env python3
# AnalyzeEmbeddings.py — v3 (TTA + linear probe + calibrated fusion + lung-masked Grad-CAM)
# - архитектура неизменна (ResNet152), загружаем указанные пользователем веса
# - никаких дообучений сети: логистические модели и калибровка только поверх эмбеддингов/логитов (train/val), отчёт на test
# - анти-утечка: разбиение берём из dataset_manifest.csv, test не участвует ни в чём, кроме финальной оценки

import os, json, argparse, numpy as np, pandas as pd, matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve, confusion_matrix
from sklearn.metrics import f1_score, balanced_accuracy_score
from sklearn.decomposition import PCA
from tensorflow.keras.applications import ResNet152
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Softmax, Input
from tensorflow.keras.models import Model
import tensorflow as tf

# ---------- utils ----------
def ensure_dir(p): os.makedirs(p, exist_ok=True); return p
def load_labels_and_manifest(labels_csv, manifest_csv):
    lab = pd.read_csv(labels_csv)
    ren = {c: c[:-2] for c in lab.columns if c.endswith("_y")}
    lab = lab.rename(columns=ren)
    lab = lab.drop_duplicates(subset=["patient_id"])
    man = pd.read_csv(manifest_csv)[["patient_id","split"]].drop_duplicates("patient_id")
    return lab.merge(man, on="patient_id", how="inner")

def load_patches(patch_dir):
    rows = []
    for f in os.listdir(patch_dir):
        if f.endswith("_B.npy"):
            pid = f.replace("_B.npy","")
            rows.append({"patient_id": pid, "patch_path": os.path.join(patch_dir,f)})
    return pd.DataFrame(rows)

def zscore(x): return (x - np.nanmean(x)) / (np.nanstd(x) + 1e-8)

def normalize_patch(arr):
    arr = arr.astype(np.float32)
    # патчи уже в [0,1]; дополнительно стабилизируем min-max
    mn, mx = arr.min(), arr.max()
    if mx > mn: arr = (arr - mn) / (mx - mn)
    return arr

def build_model(weights_path, input_shape=(224,224,3)):
    inp = Input(shape=input_shape)
    base = ResNet152(weights=None, include_top=False, input_tensor=inp)
    x = GlobalAveragePooling2D(name="avg_pool")(base.output)
    fc = Dense(2, name="fc2")(x)
    sm = Softmax(name="softmax")(fc)
    model = Model(inp, sm)
    # грузим ИМЕННО ваши веса
    model.load_weights(weights_path, by_name=True, skip_mismatch=False)
    model_avg = Model(model.input, model.get_layer("avg_pool").output)
    model_fc2 = Model(model.input, model.get_layer("fc2").output)
    # последний conv для CAM
    try:
        last_conv = model.get_layer("conv5_block3_out")
    except Exception:
        last_conv = [l for l in model.layers if len(getattr(l,"output",None).shape)==4][-1]
    grad_model = Model(model.input, [last_conv.output, model.get_layer("fc2").output])
    return model, model_avg, model_fc2, grad_model

def grad_cam_image(grad_model, img, class_index=1):
    with tf.GradientTape() as tape:
        x = tf.cast(img[None,...], tf.float32)
        conv_out, logits = grad_model(x, training=False)
        score = logits[:, class_index]
    grads = tape.gradient(score, conv_out)
    if grads is None: return None
    w = tf.reduce_mean(grads, axis=(1,2))
    cam = tf.reduce_sum(w[:,None,None,:] * conv_out, axis=-1)[0].numpy()
    cam = np.maximum(cam, 0); cam = cam / (cam.max() + 1e-8)
    cam = tf.image.resize(cam[...,None], (img.shape[0], img.shape[1])).numpy()[...,0]
    return cam

def apply_lung_mask_to_cam(cam, patch_img):
    # порог по яркости как прокси маски (лёгкие тёмнее): оставляем 70% самых тёмных
    thr = np.quantile(patch_img.mean(axis=-1), 0.7)
    lung_mask = (patch_img.mean(axis=-1) < thr).astype(np.float32)
    cam_masked = cam * lung_mask
    if cam_masked.max() > 0: cam_masked = cam_masked / cam_masked.max()
    return cam_masked

def tta_logits(model_fc2, img):
    # 2-кратный TTA: оригинал + горизонтальный флип, усредняем логиты
    im1 = img[None,...].astype(np.float32)
    im2 = im1[:, :, ::-1, :]
    l1 = model_fc2.predict(im1, verbose=0)[0]
    l2 = model_fc2.predict(im2, verbose=0)[0]
    return ((l1 + l2) / 2.0).astype(np.float32)

def fit_logreg_trainval(X_tr, y_tr, X_val, y_val, class_weight="balanced", C=1.0):
    clf = LogisticRegression(max_iter=2000, class_weight=class_weight, C=C, solver="liblinear")
    clf.fit(X_tr, y_tr)
    p_val = clf.predict_proba(X_val)[:,1]
    # изотоническая калибровка по валу
    ir = IsotonicRegression(out_of_bounds="clip").fit(p_val, y_val)
    return clf, ir

def choose_threshold_on_val(p_val_cal, y_val):
    fpr, tpr, thr = roc_curve(y_val, p_val_cal)
    youden = tpr - fpr
    t_idx = int(np.argmax(youden))
    return float(thr[t_idx])

def evaluate_block(name, y_true, p_score, thr=None, out_dir=".", prefix=""):
    auc = roc_auc_score(y_true, p_score)
    ap  = average_precision_score(y_true, p_score)
    if thr is None:
        thr = choose_threshold_on_val(p_score, y_true)
    y_hat = (p_score >= thr).astype(int)
    f1 = f1_score(y_true, y_hat)
    bacc = balanced_accuracy_score(y_true, y_hat)
    cm = confusion_matrix(y_true, y_hat)
    # ROC
    fpr, tpr, _ = roc_curve(y_true, p_score)
    plt.figure(figsize=(5,4))
    plt.plot(fpr, tpr); plt.plot([0,1],[0,1], linestyle="--")
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title(f"ROC — {name} (AUC={auc:.3f})")
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, f"{prefix}roc_{name}.png"), dpi=150); plt.close()
    # PR
    pr, rc, _ = precision_recall_curve(y_true, p_score)
    plt.figure(figsize=(5,4))
    plt.plot(rc, pr)
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title(f"PR — {name} (AP={ap:.3f})")
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, f"{prefix}pr_{name}.png"), dpi=150); plt.close()
    # CM
    plt.figure(figsize=(3.6,3.2))
    plt.imshow(cm, cmap="Blues"); plt.title(f"CM @ {thr:.3f}")
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(cm[i,j]), ha="center", va="center")
    plt.xticks([0,1],["Pred 0","Pred 1"]); plt.yticks([0,1],["True 0","True 1"])
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, f"{prefix}cm_{name}.png"), dpi=150); plt.close()
    return {"AUC": float(auc), "AP": float(ap), "F1": float(f1), "bACC": float(bacc), "thr": float(thr),
            "cm": cm.astype(int).tolist()}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-weights", required=True)
    ap.add_argument("--patch-dir", default="./patches")
    ap.add_argument("--labels-csv", default="./balanced_volumes/labels_selected.csv")
    ap.add_argument("--manifest-csv", default="./balanced_volumes/dataset_manifest.csv")
    ap.add_argument("--out-root", default="./embeddings")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--gradcam-topk", type=int, default=12)
    ap.add_argument("--mode-tag", default="probe",
                help="Имя подпапки; заодно используем для выбора варианта fusion: "
                     "probe / fusion_full / fusion_no_laa / fusion_no_qct")
    args = ap.parse_args()

    np.random.seed(args.seed); tf.random.set_seed(args.seed)
    out_dir = ensure_dir(os.path.join(args.out_root, args.mode_tag))
    figs = ensure_dir(os.path.join(out_dir, "figs"))
    cams = ensure_dir(os.path.join(out_dir, "gradcam"))

    # --- таблицы и сплиты
    lab = load_labels_and_manifest(args.labels_csv, args.manifest_csv)
    pat = load_patches(args.patch_dir)
    df = pat.merge(lab[["patient_id","split","label","laa_percent","perc15_hu","tlv_liters"]], on="patient_id", how="inner")
    df = df.dropna(subset=["split","label"]).reset_index(drop=True)
    df[["patient_id","split","label"]].to_csv(os.path.join(out_dir,"id_log.csv"), index=False)

    # --- модель
    model, model_avg, model_fc2, grad_model = build_model(args.model_weights)

    # --- прогон c TTA: логиты и эмбеддинги
    recs = []
    for _, r in tqdm(df.iterrows(), total=len(df), desc="Infer"):
        img = normalize_patch(np.load(r.patch_path))
        logits = tta_logits(model_fc2, img)   # 2-way TTA
        emb = model_avg.predict(img[None,...], verbose=0)[0]
        recs.append({"patient_id": r.patient_id, "split": r.split, "label": int(r.label),
                     "logit_0": float(logits[0]), "logit_1": float(logits[1]),
                     "laa": float(r.laa_percent), "perc15": float(r.perc15_hu), "tlv": float(r.tlv_liters)})
        # сохраним пару CAM на test-топах позже
    df_e = pd.DataFrame(recs)
    df_e.to_csv(os.path.join(out_dir,"embeddings_meta.csv"), index=False)

    # для probes нужен матрица эмбеддингов
    feats = []
    for _, r in tqdm(df.iterrows(), total=len(df), desc="Embeddings"):
        img = normalize_patch(np.load(r.patch_path))
        emb = model_avg.predict(img[None,...], verbose=0)[0]
        feats.append(emb)
    X = np.vstack(feats).astype(np.float32)
    y = df["label"].astype(int).values

    # --- train/val/test индексы
    idx_tr = df["split"]=="train"; idx_val = df["split"]=="val"; idx_te = df["split"]=="test"
    # стандартизация по train
    scaler = StandardScaler().fit(X[idx_tr])
    Xtr, Xval, Xte = scaler.transform(X[idx_tr]), scaler.transform(X[idx_val]), scaler.transform(X[idx_te])
    ytr, yval, yte = y[idx_tr], y[idx_val], y[idx_te]



    emb_df = pd.DataFrame(X, columns=[f"feat_{i}" for i in range(X.shape[1])])
    emb_df.insert(0, "patient_id", df["patient_id"].values)
    emb_df.to_csv(os.path.join(out_dir, "embeddings.csv"), index=False)



    # ===== 1) logits-only (с калибровкой) =====
    s_tr  = df_e.loc[idx_tr, "logit_1"].values
    s_val = df_e.loc[idx_val,"logit_1"].values
    s_te  = df_e.loc[idx_te, "logit_1"].values
    # переводим логиты -> sigmoid как базовую вероятность
    p_tr  = 1/(1+np.exp(-s_tr)); p_val = 1/(1+np.exp(-s_val)); p_te = 1/(1+np.exp(-s_te))
    # калибруем по валу
    ir = IsotonicRegression(out_of_bounds="clip").fit(p_val, yval)
    p_tr_cal  = ir.predict(p_tr)
    p_val_cal = ir.predict(p_val)
    p_te_cal  = ir.predict(p_te)
    thr_val = choose_threshold_on_val(p_val_cal, yval)
    metrics_logits_val = evaluate_block("logits_val", yval, p_val_cal, thr=thr_val, out_dir=figs, prefix="val_")
    metrics_logits_te  = evaluate_block("logits_test", yte,  p_te_cal,  thr=thr_val, out_dir=figs, prefix="te_")

    # ===== 2) embeddings linear probe =====
    clf_emb, ir_emb = fit_logreg_trainval(Xtr, ytr, Xval, yval, class_weight="balanced", C=1.0)
    pv_tr  = ir_emb.predict(clf_emb.predict_proba(Xtr)[:,1])
    pv_val = ir_emb.predict(clf_emb.predict_proba(Xval)[:,1])
    pv_te  = ir_emb.predict(clf_emb.predict_proba(Xte)[:,1])
    thr_emb = choose_threshold_on_val(pv_val, yval)
    metrics_emb_val = evaluate_block("emb_probe_val", yval, pv_val, thr=thr_emb, out_dir=figs, prefix="val_")
    metrics_emb_te  = evaluate_block("emb_probe_test", yte,  pv_te,  thr=thr_emb, out_dir=figs, prefix="te_")

    # ===== 3) fusion: logits + LAA + Perc15 + TLV =====
    # ===== 3) fusion: варианты =====
    mode = args.mode_tag

    if mode == "fusion_no_qct":
        # Эксперимент 3: fusion без QCT — только калиброванный logit CNN
        Ftr  = p_tr_cal.reshape(-1, 1)
        Fval = p_val_cal.reshape(-1, 1)
        Fte  = p_te_cal.reshape(-1, 1)
        fusion_name = "fusion_no_qct"

    elif mode == "fusion_no_laa":
        # Эксперимент 2: fusion без LAA — используем только Perc15 и TLV
        Ftr  = np.stack([
            p_tr_cal,
            zscore(df_e.loc[idx_tr, "perc15"].values),
            zscore(df_e.loc[idx_tr, "tlv"].values)
        ], axis=1)
        Fval = np.stack([
            p_val_cal,
            zscore(df_e.loc[idx_val, "perc15"].values),
            zscore(df_e.loc[idx_val, "tlv"].values)
        ], axis=1)
        Fte  = np.stack([
            p_te_cal,
            zscore(df_e.loc[idx_te, "perc15"].values),
            zscore(df_e.loc[idx_te, "tlv"].values)
        ], axis=1)
        fusion_name = "fusion_no_laa"

    else:
        # Оригинальный fusion: logits + %LAA + Perc15 + TLV
        Ftr  = np.stack([
            p_tr_cal,
            zscore(df_e.loc[idx_tr, "laa"].values),
            zscore(df_e.loc[idx_tr, "perc15"].values),
            zscore(df_e.loc[idx_tr, "tlv"].values)
        ], axis=1)
        Fval = np.stack([
            p_val_cal,
            zscore(df_e.loc[idx_val, "laa"].values),
            zscore(df_e.loc[idx_val, "perc15"].values),
            zscore(df_e.loc[idx_val, "tlv"].values)
        ], axis=1)
        Fte  = np.stack([
            p_te_cal,
            zscore(df_e.loc[idx_te, "laa"].values),
            zscore(df_e.loc[idx_te, "perc15"].values),
            zscore(df_e.loc[idx_te, "tlv"].values)
        ], axis=1)
        fusion_name = "fusion_full"

    clf_f, ir_f = fit_logreg_trainval(Ftr, ytr, Fval, yval, class_weight="balanced", C=1.0)
    pf_val = ir_f.predict(clf_f.predict_proba(Fval)[:, 1])
    pf_te  = ir_f.predict(clf_f.predict_proba(Fte)[:, 1])
    thr_f = choose_threshold_on_val(pf_val, yval)

    metrics_f_val = evaluate_block(f"{fusion_name}_val",  yval, pf_val, thr=thr_f, out_dir=figs, prefix="val_")
    metrics_f_te  = evaluate_block(f"{fusion_name}_test", yte,  pf_te,  thr=thr_f, out_dir=figs, prefix="te_")


    # ===== Grad-CAM (легко-маскированный) на top-K TEST =====
    dtest = df.loc[idx_te].copy()
    # берём топ по калиброванной вероятности fusion (как самый содержательный скор)
    dtest["score"] = pf_te
    topK = dtest.sort_values("score", ascending=False).head(args.gradcam_topk)
    for _, r in topK.iterrows():
        pid = r.patient_id; ppath = os.path.join(args.patch_dir, f"{pid}_B.npy")
        if not os.path.exists(ppath): continue
        img = normalize_patch(np.load(ppath))
        cam = grad_cam_image(grad_model, img, class_index=1)
        if cam is None: continue
        cam_m = apply_lung_mask_to_cam(cam, img)  # ограничиваем лёгкими
        fig, ax = plt.subplots(2,3, figsize=(9,6))
        for i in range(3):
            ax[0,i].imshow(img[:,:,i], cmap="gray", vmin=0, vmax=1); ax[0,i].axis("off")
            ax[1,i].imshow(img[:,:,i], cmap="gray", vmin=0, vmax=1)
            ax[1,i].imshow(cam_m, cmap="jet", alpha=0.45); ax[1,i].axis("off")
        fig.suptitle(f"{pid} | fusion p={r['score']:.3f}")
        plt.tight_layout(); plt.savefig(os.path.join(cams, f"{pid}_gradcam.png"), dpi=150); plt.close()


    # ===== Итоговый отчёт =====
    report = {
        "n_total": int(len(df)),
        "class_counts": df["label"].value_counts().to_dict(),
        "val": {
            "logits": metrics_logits_val,
            "emb_probe": metrics_emb_val,
            "fusion": metrics_f_val
        },
        "test": {
            "logits": metrics_logits_te,
            "emb_probe": metrics_emb_te,
            "fusion": metrics_f_te
        },
        "notes": {
            "weights": os.path.basename(args.model_weights),
            "features": "avg_pool 2048, ResNet152",
            "tta": "orig + horizontal flip",
            "fusion_mode": mode,
            "fusion_features": {
                "fusion_full":   ["calibrated_logit1", "%LAA", "Perc15", "TLV"],
                "fusion_no_laa": ["calibrated_logit1", "Perc15", "TLV"],
                "fusion_no_qct": ["calibrated_logit1"]
            },
            "leakage_control": "fit on train, tune on val, report on test"
        }
    }
    with open(os.path.join(out_dir,"metrics.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
    print(f"\n✓ Outputs: {out_dir}")

if __name__ == "__main__":
    main()
