# SegmentAndMark.py  — v2 (QC+metrics+LAA maps)
# Дополняет оригинал: сохраняет TLV (л), Perc15, счётчики вокселей, QC-флаги, LAA-карты.
# Пороговая логика LAA-950 и LABEL_THRESHOLD=6.0% — как обсуждали.
# Основания и риски: см. методы и ссылки в PlotDataAnalysis.py (methods_report.txt).

import os
import json
import numpy as np
import pandas as pd
import SimpleITK as sitk
import matplotlib.pyplot as plt
from lungmask import mask as lm_mask
from concurrent.futures import ThreadPoolExecutor, as_completed

# === Пути ===
VOLUME_DIR = "./npy_volumes"
MASK_DIR = "./lung_masks"
VIS_DIR = "./slices_visual"
DEBUG_DIR = "./debug_slices"
RESAMPLED_DIR = "./resampled_volumes"
LAA_MAP_DIR = "./laa_maps"
LABEL_FILE = "./laa_analysis/labels.csv"
QC_FILE = "./laa_analysis/qc_cases.csv"
LOG_PATH = "./laa_analysis/segment_log.txt"

# === Параметры ===
LAA_THRESHOLD = -950
LABEL_THRESHOLD = 6.0  # %
MEAN_HU_CUTOFF = -850
MIN_MASK_VOXELS = 15000
MAX_LAA_SANITY = 85
RESAMPLE_SPACING = [1.0, 1.0, 1.0]  # мм

# === Init folders ===
for path in [MASK_DIR, VIS_DIR, DEBUG_DIR, RESAMPLED_DIR, LAA_MAP_DIR, os.path.dirname(LABEL_FILE)]:
    os.makedirs(path, exist_ok=True)

def percentile(x, q):
    if x.size == 0:
        return np.nan
    return float(np.percentile(x, q))

def process_volume(path):
    patient_id = os.path.basename(path).replace(".npy", "")
    qc_flags = []
    try:
        spacing_path = os.path.join(VOLUME_DIR, f"{patient_id}_spacing.json")
        if not os.path.exists(spacing_path):
            return f"[!] {patient_id}: missing spacing", None, ("missing_spacing", patient_id)

        with open(spacing_path, "r") as f:
            spacing = json.load(f)["spacing"]  # [sx, sy, sz] из ExtractVolumes

        volume = np.load(path).astype(np.float32)

        # === To ITK с корректными метаданными
        itk = sitk.GetImageFromArray(volume)
        itk.SetSpacing(spacing)
        itk.SetOrigin([0.0, 0.0, 0.0])
        itk.SetDirection(np.identity(3).flatten().tolist())

        # === Resample to 1×1×1 mm (гармонизация метрики)
        new_spacing = RESAMPLE_SPACING
        new_size = [int(round(itk.GetSize()[i] * (spacing[i] / new_spacing[i]))) for i in range(3)]

        resampler = sitk.ResampleImageFilter()
        resampler.SetSize(new_size)
        resampler.SetOutputSpacing(new_spacing)
        resampler.SetInterpolator(sitk.sitkLinear)
        resampler.SetDefaultPixelValue(-1024)
        resampler.SetOutputOrigin(itk.GetOrigin())
        resampler.SetOutputDirection(itk.GetDirection())

        itk_resampled = resampler.Execute(itk)
        vol_resampled = sitk.GetArrayFromImage(itk_resampled).astype(np.float32)
        np.save(os.path.join(RESAMPLED_DIR, f"{patient_id}.npy"), vol_resampled)

        # === Lung segmentation (lungmask)
        mask = lm_mask.apply(itk_resampled)
        mask = (mask > 0).astype(np.uint8)
        lung_voxels = int(mask.sum())           # после ресемплинга 1 мм → 1 воксель = 1 мм^3
        if lung_voxels < MIN_MASK_VOXELS:
            qc_flags.append("mask_small")

        # === Метрики внутри лёгких
        lung_vals = vol_resampled[mask == 1]
        mean_hu = float(lung_vals.mean()) if lung_voxels else np.nan
        perc15_hu = percentile(lung_vals, 15)

        # === LAA карта и %LAA
        laa_map = ((vol_resampled < LAA_THRESHOLD) & (mask == 1)).astype(np.uint8)
        emphysema_voxels = int(laa_map.sum())
        laa_percent = (emphysema_voxels / lung_voxels * 100) if lung_voxels else np.nan

        if laa_percent is not np.nan and laa_percent > MAX_LAA_SANITY:
            qc_flags.append("implausible_laa")

        # === TLV (литры) — после 1мм ресемплинга
        tlv_liters = float(lung_voxels) / 1e6  # мм^3 → л

        # === Label (с учётом порога mean HU как sanity)
        label = int((laa_percent is not np.nan) and (laa_percent >= LABEL_THRESHOLD) and (mean_hu < MEAN_HU_CUTOFF))

        # === Сохранения
        np.save(os.path.join(MASK_DIR, f"{patient_id}_mask.npy"), mask)
        np.save(os.path.join(LAA_MAP_DIR, f"{patient_id}_laa.npy"), laa_map)

        # Быстрая визуализация (центральный срез)
        mid = vol_resampled.shape[0] // 2
        fig, ax = plt.subplots(1, 2, figsize=(8, 4))
        ax[0].imshow(vol_resampled[mid], cmap="gray")
        ax[0].set_title("Slice (HU)")
        ax[1].imshow(vol_resampled[mid], cmap="gray")
        ax[1].imshow(mask[mid], cmap="Reds", alpha=0.4)
        ax[1].set_title(f"LAA={laa_percent:.1f}%")
        for a in ax: a.axis("off")
        plt.tight_layout()
        plt.savefig(os.path.join(VIS_DIR, f"{patient_id}.png"))
        plt.close()

        # LAA по срезам
        laa_by_slice = np.sum(laa_map, axis=(1, 2))
        plt.figure(figsize=(6, 2))
        plt.plot(laa_by_slice)
        plt.title(f"{patient_id} – LAA по срезам")
        plt.xlabel("Slice")
        plt.ylabel("LAA voxels")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(DEBUG_DIR, f"{patient_id}_laa_curve.png"))
        plt.close()

        qc_str = ";".join(qc_flags) if qc_flags else ""
        log = (f"[+] {patient_id}: LAA={laa_percent:.2f}%, HU={mean_hu:.1f}, "
               f"Perc15={perc15_hu:.1f}, TLV={tlv_liters:.2f}L → label={label} {('['+qc_str+']') if qc_str else ''}")
        return log, (patient_id, laa_percent, emphysema_voxels, lung_voxels, mean_hu, perc15_hu, tlv_liters, label, qc_str), None

    except Exception as e:
        return f"[!] {patient_id}: failed — {e}", None, ("failed", patient_id)

if __name__ == "__main__":
    log_lines = []
    records = []
    qc_rows = []
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)

    files = sorted([f for f in os.listdir(VOLUME_DIR) if f.endswith(".npy") and "_spacing" not in f])

    from tqdm import tqdm
    with ThreadPoolExecutor(max_workers=6) as executor:
        futures = {executor.submit(process_volume, os.path.join(VOLUME_DIR, f)): f for f in files}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Segment+LAA"):
            log, result, qc = fut.result()
            print(log)
            log_lines.append(log)
            if result:
                records.append(result)
            if qc:
                qc_rows.append(qc)

    # === Сохраняем CSV и лог
    df = pd.DataFrame(records, columns=[
        "patient_id", "laa_percent", "laa_voxels", "lung_voxels",
        "mean_lung_hu", "perc15_hu", "tlv_liters", "label", "qc_flags"
    ])
    os.makedirs(os.path.dirname(LABEL_FILE), exist_ok=True)
    df.to_csv(LABEL_FILE, index=False)

    if qc_rows:
        qcdf = pd.DataFrame(qc_rows, columns=["issue", "patient_id"])
        qcdf.to_csv(QC_FILE, index=False)

    with open(LOG_PATH, "w") as f:
        f.write("\n".join(log_lines))

    print(f"\n✔️ Done. Labels saved to {LABEL_FILE}, log saved to {LOG_PATH}")
