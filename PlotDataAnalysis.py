# PlotDataAnalysis.py — сбор графиков, QC-диаграмм и контуров LAA (nodule_visual)
# Требования: numpy, pandas, matplotlib, scikit-image (контуры), (опц.) tqdm
# Результаты: ./figures/*.png, ./nodule_visual/*.png, ./nodule_visual/index.html, methods_report.txt

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from skimage import measure
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection

FIG_DIR = "./figures"
NODULE_VIS_DIR = "./nodule_visual"
VOLUME_DIR = "./resampled_volumes"   # из SegmentAndMark v2
MASK_DIR = "./lung_masks"
LAA_MAP_DIR = "./laa_maps"

LABEL_CSV = "./laa_analysis/labels.csv"
HU_STATS_CSV = "./hu_statistics.csv"     # из ExtractVolumes.py
SERIES_INFO_CSV = "./series_info.csv"    # из ExtractVolumes.py

os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(NODULE_VIS_DIR, exist_ok=True)

def load_df_safe(path, required=False):
    if os.path.exists(path):
        return pd.read_csv(path)
    if required:
        raise FileNotFoundError(f"Required file not found: {path}")
    return pd.DataFrame()

def fig_save(name):
    out = os.path.join(FIG_DIR, name)
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    return out

def plot_hist_with_thresholds(df):
    if df.empty: return
    vals = df["laa_percent"].dropna().values
    plt.figure(figsize=(6,4))
    plt.hist(vals, bins=40)
    plt.axvline(6, linestyle="--", linewidth=2)
    plt.axvline(14, linestyle="--", linewidth=2)
    plt.title("Distribution of %LAA-950 (Thresholds: 6% and 14%)")
    plt.xlabel("%LAA-950")
    plt.ylabel("Number of Cases")
    fig_save("hist_laa.png")

def plot_scatter_laa_vs_meanhu(df):
    if df.empty: return
    x = df["laa_percent"].values
    y = df["mean_lung_hu"].values
    plt.figure(figsize=(6,4))
    plt.scatter(x, y, s=10, alpha=0.6)
    plt.xlabel("%LAA-950")
    plt.ylabel("Mean Lung HU")
    plt.title("Relationship between %LAA-950 and Mean Lung Density")
    try:
        from scipy.stats import pearsonr, spearmanr
        pr = pearsonr(x, y)
        sr = spearmanr(x, y)
        plt.suptitle(f"Pearson r={pr[0]:.2f} (p={pr[1]:.1e}), Spearman ρ={sr.correlation:.2f} (p={sr.pvalue:.1e})", y=0.98, fontsize=9)
    except Exception:
        pass
    fig_save("scatter_laa_vs_meanhu.png")

def plot_scatter_laa_vs_tlv(df):
    if df.empty or "tlv_liters" not in df.columns: return
    x = df["tlv_liters"].values
    y = df["laa_percent"].values
    plt.figure(figsize=(6,4))
    plt.scatter(x, y, s=10, alpha=0.6)
    plt.xlabel("TLV (liters)")
    plt.ylabel("%LAA-950")
    plt.title("Effect of Inspiration: %LAA-950 vs TLV")
    fig_save("scatter_laa_vs_tlv.png")

def plot_series_selection():
    s = load_df_safe(SERIES_INFO_CSV)
    if s.empty: return
    plt.figure(figsize=(6,4))
    s["num_slices"].hist(bins=50)
    plt.xlabel("Number of Slices in Series")
    plt.ylabel("Number of Series")
    plt.title("Distribution of Slice Count per Series (series_info.csv)")
    fig_save("hist_num_slices.png")

def plot_status_bars():
    h = load_df_safe(HU_STATS_CSV)
    if h.empty or "status" not in h.columns: return
    counts = h["status"].fillna("unknown").value_counts()
    plt.figure(figsize=(6,4))
    counts.plot(kind="bar")
    plt.ylabel("Number of Patients")
    plt.title("HU Statistics Status (why not 1018?)")
    fig_save("bar_hu_status.png")

def plot_hu_histograms():
    h = load_df_safe(HU_STATS_CSV)
    if h.empty: return
    plt.figure(figsize=(6,4))
    h["mean"].dropna().hist(bins=50)
    plt.xlabel("Mean HU (entire volume)")
    plt.ylabel("Number of Cases")
    plt.title("Distribution of Mean HU (for threshold justification)")
    fig_save("hist_mean_hu.png")

    plt.figure(figsize=(6,4))
    h["std"].dropna().hist(bins=50)
    plt.xlabel("Std HU (entire volume)")
    plt.ylabel("Number of Cases")
    plt.title("Distribution of Std HU (for threshold justification)")
    fig_save("hist_std_hu.png")


def draw_laa_contours_on_slice(ax, slice_img, laa_slice):
    ax.imshow(slice_img, cmap="gray")
    # Находим контуры LAA (уровень 0.5 для бинарной карты)
    contours = measure.find_contours(laa_slice, 0.5)
    patches = []
    for contour in contours:
        poly = Polygon(np.fliplr(contour), closed=True, fill=False, linewidth=1.2)
        patches.append(poly)
    pc = PatchCollection(patches, match_original=True, edgecolor="red", facecolor="none", linewidths=1.2)
    ax.add_collection(pc)
    ax.axis("off")

def create_nodule_visuals(df, max_slices_per_case=6):
    # Для всех label=1 (LAA>=6%) строим обводку LAA на срезах с максимальной долей LAA
    pos = df[df["label"] == 1]
    index_rows = []
    for _, row in pos.iterrows():
        pid = row["patient_id"]
        vol_path = os.path.join(VOLUME_DIR, f"{pid}.npy")
        mask_path = os.path.join(MASK_DIR, f"{pid}_mask.npy")
        laa_path  = os.path.join(LAA_MAP_DIR, f"{pid}_laa.npy")
        if not (os.path.exists(vol_path) and os.path.exists(mask_path) and os.path.exists(laa_path)):
            continue

        vol = np.load(vol_path)
        laa = np.load(laa_path).astype(bool)

        # Выберем срезы с наибольшей LAA-площадью
        laa_by_slice = laa.sum(axis=(1,2))
        order = np.argsort(laa_by_slice)[::-1]
        top = [i for i in order[:max_slices_per_case] if laa_by_slice[i] > 0]
        if not top:
            continue

        # Рисуем сеткой 3x2 (или меньше)
        rows = int(np.ceil(len(top)/2))
        fig, axes = plt.subplots(rows, 2, figsize=(8, 3*rows))
        axes = np.atleast_1d(axes).ravel()
        for ax, sl in zip(axes, top):
            draw_laa_contours_on_slice(ax, vol[sl], laa[sl])
            ax.set_title(f"{pid} | slice {sl}")

        for ax in axes[len(top):]:
            ax.axis("off")

        out_png = os.path.join(NODULE_VIS_DIR, f"{pid}.png")
        plt.tight_layout()
        plt.savefig(out_png, dpi=150)
        plt.close()
        index_rows.append({"patient_id": pid, "laa_percent": row["laa_percent"], "image": os.path.basename(out_png)})

    # Простой HTML-индекс
    if index_rows:
        df_idx = pd.DataFrame(index_rows).sort_values("laa_percent", ascending=False)
        html = ["<html><head><meta charset='utf-8'><title>nodule_visual</title></head><body>",
                "<h2>Cases with LAA ≥ 6% (Contours of LAA regions)</h2><ul>"]
        for _, r in df_idx.iterrows():
            html.append(f"<li>{r['patient_id']} — LAA={r['laa_percent']:.2f}%<br><img src='{r['image']}' style='max-width:900px'></li>")
        html.append("</ul></body></html>")
        with open(os.path.join(NODULE_VIS_DIR, "index.html"), "w", encoding="utf-8") as f:
            f.write("\n".join(html))

def write_methods_report():
    # Краткое текстовое резюме с ссылками (для приложения к статье/репо)
    text = """
METHODS / QC SUMMARY (for reviewers)
------------------------------------
Metrics:
- %LAA-950 := доля вокселей < -950 HU внутри маски лёгких (инспираторная КТ).
- Бинаризация: LAA ≥ 6% → положительно (эмфизема), иначе отрицательно. Порог 14% отмечается как "тяжёлая" в графиках.

Harmonization / QC:
- Ресемплинг к 1мм изотропно перед метриками (%LAA, Perc15, TLV).
- TLV (л) считается как число лёгочных вокселей / 1e6; используем для анализа влияния вдоха.
- Извлекаем mean HU и Perc15 (легочная паренхима).
- Примечание: ядра реконструкции/денойзинг не нормализуются пост-фактум (нет тегов), но обсуждаются как источник смещения и отражены в отчёте.

Files / Plots:
- figures/hist_laa.png — распределение %LAA (пороговые линии 6% и 14%).
- figures/scatter_laa_vs_meanhu.png — связь с mean HU.
- figures/scatter_laa_vs_tlv.png — влияние вдоха.
- figures/hist_mean_hu.png, hist_std_hu.png — обоснование фильтров извлечения.
- figures/bar_hu_status.png — объяснение "почему не 1018" (статусы из hu_statistics.csv).
- figures/hist_num_slices.png — иллюстрация выбора самой полной серии.
- nodule_visual/ — контуры LAA на срезах с наибольшей LAA-долей.

Citations (evidence):
- LIDC-IDRI description: 1018 CT scans; two-phase annotation by four radiologists.
  Armato SG et al., Med Phys 2011. PubMed: 21452728.
- Emphysema threshold −950 HU widely used (2022 review).
  Abadi E et al., J Med Imaging 2022. PMC10206513.
- Clinical cut points: 6% (no significant), 14% (severe).
  Occhipinti M et al., Respir Res 2019. doi:10.1186/s12931-019-1049-3.
- Denoising / IR reduce %LAA-950 (bias).
  de Boer E et al., Insights Imaging 2019. doi:10.1186/s13244-019-0776-9.
- Kernel conversion improves correlation to clinical measures.
  An TJ et al., Tuberc Respir Dis 2025. e-TRD 2025.
- Inspiration (TLV) bias and correction.
  Kavuri A et al., Acad Radiol 2025. PubMed 40348708.
- Emphysema scores applied directly on IDRI (n=460).
  Wiemker R et al., SPIE 2009. doi:10.1117/12.811549.
- GOLD clinical GT (post-BD FEV1/FVC<0.70).
  GOLD 2024 Pocket Guide (ver.1.2), Jan 2024.
"""
    with open(os.path.join(FIG_DIR, "methods_report.txt"), "w", encoding="utf-8") as f:
        f.write(text.strip()+"\n")

def main():
    df = load_df_safe(LABEL_CSV, required=True)
    plot_hist_with_thresholds(df)
    plot_scatter_laa_vs_meanhu(df)
    plot_scatter_laa_vs_tlv(df)
    plot_series_selection()
    plot_status_bars()
    plot_hu_histograms()
    create_nodule_visuals(df, max_slices_per_case=6)
    write_methods_report()
    print("✔️ Figures saved to ./figures, LAA contours saved to ./nodule_visual")

if __name__ == "__main__":
    main()
