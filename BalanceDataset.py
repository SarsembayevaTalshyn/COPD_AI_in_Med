#!/usr/bin/env python3
# BalanceDataset.py — v2
# Режимы набора: all / balanced / weighted
# Стратегии баланса: downsample / upsample / smote (если imblearn нет — fallback на kNN-oversample)
# Стратифицированный train/val/test split, детальный манифест с patient_id и путями
# Совместим с пайплайном: SegmentAndMark v2 (./resampled_volumes, labels.csv) и PreparePatches v2

import os
import json
import shutil
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    from sklearn.model_selection import train_test_split
    SK_SPLIT = True
except Exception:
    SK_SPLIT = False

def stratified_split(df, seed, ratios=(0.7, 0.15, 0.15)):
    assert abs(sum(ratios) - 1.0) < 1e-6, "Ratios must sum to 1"
    if SK_SPLIT:
        train_df, tmp = train_test_split(
            df, test_size=(1 - ratios[0]), stratify=df["label"], random_state=seed
        )
        rel = ratios[1] / (ratios[1] + ratios[2])
        val_df, test_df = train_test_split(
            tmp, test_size=(1 - rel), stratify=tmp["label"], random_state=seed
        )
    else:
        # простой детерминированный стратифицированный сплит без sklearn
        rng = np.random.default_rng(seed)
        train_idx, val_idx, test_idx = [], [], []
        for y, group in df.groupby("label"):
            idx = list(group.index)
            rng.shuffle(idx)
            n = len(idx)
            n_train = int(round(ratios[0] * n))
            n_val = int(round(ratios[1] * n))
            train_idx += idx[:n_train]
            val_idx += idx[n_train:n_train+n_val]
            test_idx += idx[n_train+n_val:]
        train_df = df.loc[train_idx]
        val_df = df.loc[val_idx]
        test_df = df.loc[test_idx]
    train_df = train_df.copy(); train_df["split"] = "train"
    val_df   = val_df.copy();   val_df["split"] = "val"
    test_df  = test_df.copy();  test_df["split"] = "test"
    return pd.concat([train_df, val_df, test_df], axis=0)

def compute_class_weights(df):
    # N/(2*N_c): суммарная сумма весов по классам ≈ 1
    counts = df["label"].value_counts().to_dict()
    N = len(df)
    weights = {int(k): float(N / (2.0 * v)) for k, v in counts.items()}
    return weights, counts

def knn_like_oversample(minority_df, need_extra, seed):
    """Простой оверсэмплинг: случайно повторяем minority до нужного размера (fallback вместо SMOTE)."""
    rng = np.random.default_rng(seed)
    if need_extra <= 0 or len(minority_df) == 0:
        return minority_df.copy()
    reps = minority_df.sample(n=need_extra, replace=True, random_state=seed)
    return pd.concat([minority_df, reps], axis=0)

def try_smote_oversample(df, seed, features_cols):
    """Пытаемся использовать imblearn.SMOTE для определения «каких минорити» чаще повторять.
       Если нет imblearn — fallback на knn_like_oversample.
       Возвращаем DataFrame с оверсэмплингом МИНОРИТИ до размера мажорити."""
    y = df["label"].values
    # minor / major
    counts = df["label"].value_counts().to_dict()
    maj = max(counts, key=counts.get)
    minc = min(counts, key=counts.get)
    n_target = counts[maj]

    if counts[minc] >= n_target:
        return df.copy()  # уже сбалансировано

    try:
        from imblearn.over_sampling import SMOTE
        X = df[features_cols].fillna(df[features_cols].median()).values
        sm = SMOTE(random_state=seed)
        X_res, y_res = sm.fit_resample(X, y)
        # SMOTE возвращает "синтетические" точки — нам не к чему; мы используем это только
        # чтобы посчитать сколько ДОБАВИТЬ минорити и затем продублируем реальные пациенты.
        # Поэтому просто делаем равновесный upsample минорити по случайному выбору:
        need_extra = n_target - counts[minc]
        minor_df = df[df["label"] == minc]
        up_df = minor_df.sample(n=need_extra, replace=True, random_state=seed)
        return pd.concat([df[df["label"] == maj], minor_df, up_df], axis=0)
    except Exception:
        # fallback — равновесный оверсэмплинг минорити
        need_extra = n_target - counts[minc]
        minor_df = df[df["label"] == minc]
        maj_df = df[df["label"] == maj]
        up_df = knn_like_oversample(minor_df, need_extra, seed)
        return pd.concat([maj_df, up_df], axis=0)

def copy_case_files(pids, source_dir, spacing_dir, dest_dir, no_copy=False, link=False):
    os.makedirs(dest_dir, exist_ok=True)
    copied = []

    for pid in tqdm(pids, desc="Prepare volumes"):
        npy_src = os.path.join(source_dir, f"{pid}.npy")
        json_src = os.path.join(spacing_dir, f"{pid}_spacing.json")

        if not (os.path.exists(npy_src) and os.path.exists(json_src)):
            print(f"[!] Missing files for {pid}")
            continue

        if no_copy:
            copied.append(pid)
            continue

        try:
            npy_dst = os.path.join(dest_dir, f"{pid}.npy")
            json_dst = os.path.join(dest_dir, f"{pid}_spacing.json")

            if link:
                if not os.path.exists(npy_dst):
                    os.symlink(os.path.abspath(npy_src), npy_dst)
                if not os.path.exists(json_dst):
                    os.symlink(os.path.abspath(json_src), json_dst)
            else:
                shutil.copy2(npy_src, npy_dst)
                shutil.copy2(json_src, json_dst)

            copied.append(pid)

        except Exception as e:
            print(f"[!] Failed for {pid}: {e}")

    return copied


def build_manifest(df_sel, source_dir, spacing_dir, dest_dir,
                   mode, balance_strategy, class_weights=None, no_copy=False):

    manifest = df_sel[["patient_id", "label"]].copy()

    if no_copy:
        manifest["npy_path"] = manifest["patient_id"].apply(
            lambda x: os.path.join(source_dir, f"{x}.npy")
        )
        manifest["spacing_json"] = manifest["patient_id"].apply(
            lambda x: os.path.join(spacing_dir, f"{x}_spacing.json")
        )
    else:
        manifest["npy_path"] = manifest["patient_id"].apply(
            lambda x: os.path.join(dest_dir, f"{x}.npy")
        )
        manifest["spacing_json"] = manifest["patient_id"].apply(
            lambda x: os.path.join(dest_dir, f"{x}_spacing.json")
        )

    if "split" in df_sel.columns:
        manifest["split"] = df_sel["split"].values

    if class_weights is not None:
        manifest["weight"] = manifest["label"].map(class_weights).astype(float)

    manifest["mode"] = mode
    manifest["balance_strategy"] = balance_strategy
    return manifest

def main():
    ap = argparse.ArgumentParser(description="Balance dataset and create split manifests")
    ap.add_argument("--labels-csv", default="./laa_analysis/labels.csv")
    ap.add_argument("--source-dir", default="./resampled_volumes")
    ap.add_argument("--spacing-dir", default="./npy_volumes")
    ap.add_argument("--dest-dir", default="./balanced_volumes")
    ap.add_argument("--mode", choices=["all", "balanced", "weighted"], default="balanced")
    ap.add_argument("--balance-strategy", choices=["downsample", "upsample", "smote"], default="downsample")
    ap.add_argument("--split", type=float, nargs=3, default=[0.7, 0.15, 0.15], metavar=("TRAIN","VAL","TEST"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-copy", action="store_true",
                    help="Do not copy .npy/.json files, only build CSV manifests with original paths")
    ap.add_argument("--link", action="store_true",
                    help="Create symlinks instead of copying files")
    args = ap.parse_args()

    os.makedirs(args.dest_dir, exist_ok=True)

    df = pd.read_csv(args.labels_csv)
    base_cols = ["patient_id","label","laa_percent","mean_lung_hu","perc15_hu","tlv_liters"]
    for c in base_cols:
        if c not in df.columns:
            raise ValueError(f"Column missing in labels: {c}")

    # фильтруем на валидные случаи (без NaN по ключевым полям)
    df = df.dropna(subset=["patient_id","label"]).copy()
    df["label"] = df["label"].astype(int)

    # Выбор подрежима
    sel = None
    strat_text = "none"
    if args.mode == "all":
        sel = df.copy()
        strat_text = "no balancing"

    elif args.mode == "balanced":
        counts = df["label"].value_counts().to_dict()
        if len(counts) < 2:
            raise ValueError("Need both classes for balancing")
        maj = max(counts, key=counts.get)
        minc = min(counts, key=counts.get)

        if args.balance_strategy == "downsample":
            pos = df[df["label"] == minc] if minc == 1 else df[df["label"] == 1]
            neg = df[df["label"] == maj] if maj == 0 else df[df["label"] == 0]
            neg_bal = neg.sample(n=len(pos), random_state=args.seed, replace=False)
            sel = pd.concat([pos, neg_bal], axis=0)

        elif args.balance_strategy == "upsample":
            pos = df[df["label"] == 1]
            neg = df[df["label"] == 0]
            if len(pos) == 0:
                raise ValueError("No positives to upsample")
            pos_up = pos.sample(n=len(neg), replace=True, random_state=args.seed)
            sel = pd.concat([neg, pos_up], axis=0)

        elif args.balance_strategy == "smote":
            # Используем табличные признаки как прокси-пространство для SMOTE, физически дублируем реальные минорити.
            features_cols = ["laa_percent","mean_lung_hu","perc15_hu","tlv_liters"]
            sel = try_smote_oversample(df[["patient_id","label"]+features_cols], args.seed, features_cols)

        strat_text = args.balance_strategy

    elif args.mode == "weighted":
        sel = df.copy()
        strat_text = "class_weights"

    sel = sel.drop_duplicates(subset=["patient_id"]).sort_values("patient_id").reset_index(drop=True)

    # Сплиты
    split_df = stratified_split(sel[["patient_id","label"]], seed=args.seed, ratios=tuple(args.split))
    split_df = split_df.sort_values("patient_id").reset_index(drop=True)

    # Копирование файлов (подмножество = все уникальные pid из split_df)
    copied = copy_case_files(
        split_df["patient_id"].tolist(),
        args.source_dir,
        args.spacing_dir,
        args.dest_dir,
        no_copy=args.no_copy,
        link=args.link
    )
    split_df = split_df[split_df["patient_id"].isin(copied)].reset_index(drop=True)

    # Class weights (для weighted режима)
    cw = None
    if args.mode == "weighted":
        cw, counts = compute_class_weights(split_df)
        with open(os.path.join(args.dest_dir, "class_weights.json"), "w") as f:
            json.dump({"class_weights": cw, "counts": counts}, f, indent=2)

    # Сохраняем манифесты
    split_df.to_csv(os.path.join(args.dest_dir, "split_manifest.csv"), index=False)
    sel_labs = sel.merge(df[["patient_id","laa_percent","mean_lung_hu","perc15_hu","tlv_liters","label"]], on=["patient_id","label"], how="left")
    sel_labs.to_csv(os.path.join(args.dest_dir, "labels_selected.csv"), index=False)

    manifest = build_manifest(
        split_df.merge(sel_labs, on=["patient_id","label"], how="left"),
        args.source_dir,
        args.spacing_dir,
        args.dest_dir,
        args.mode,
        strat_text,
        class_weights=cw,
        no_copy=args.no_copy
    )
    manifest.to_csv(os.path.join(args.dest_dir, "dataset_manifest.csv"), index=False)

    # Лог
    counts_sel = split_df["label"].value_counts().to_dict()
    log = {
        "mode": args.mode,
        "balance_strategy": strat_text,
        "seed": args.seed,
        "split": args.split,
        "selected_total": int(len(split_df)),
        "selected_by_class": {int(k): int(v) for k, v in counts_sel.items()}
    }
    with open(os.path.join(args.dest_dir, "build_log.json"), "w") as f:
        json.dump(log, f, indent=2)
    print("\n[✓] Dataset built.")
    print(json.dumps(log, indent=2))

if __name__ == "__main__":
    main()
