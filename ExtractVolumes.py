import os
import json
import numpy as np
import pydicom
from glob import glob
from tqdm import tqdm
import concurrent.futures
import csv
from datetime import datetime

# === Загрузка DICOM и преобразование в HU ===
def load_dicom_volume(series_path):
    slices = []
    for fname in sorted(os.listdir(series_path)):
        if not fname.endswith(".dcm"):
            continue
        try:
            dcm = pydicom.dcmread(os.path.join(series_path, fname))
            _ = dcm.pixel_array
            slices.append(dcm)
        except Exception as e:
            print(f"[!] Skipped corrupted slice: {fname} ({e})")
            continue

    if len(slices) < 10:
        raise ValueError("Too few valid slices")

    slices.sort(key=lambda s: float(s.ImagePositionPatient[2]))
    image = np.stack([s.pixel_array for s in slices]).astype(np.int16)

    intercept = getattr(slices[0], "RescaleIntercept", -1024)
    slope = getattr(slices[0], "RescaleSlope", 1)
    image = slope * image + intercept  # → HU

    spacing = get_spacing_from_dicom(slices)
    return image, spacing

def get_spacing_from_dicom(slices):
    pixel_spacing = slices[0].PixelSpacing
    try:
        spacing_z = abs(slices[1].ImagePositionPatient[2] - slices[0].ImagePositionPatient[2])
    except:
        spacing_z = float(slices[0].SliceThickness)
    return [float(pixel_spacing[1]), float(pixel_spacing[0]), spacing_z]  # [X, Y, Z]

# === Поиск главной серии DICOM ===
def find_main_dicom_series(patient_path):
    series_info = []
    for root, dirs, files in os.walk(patient_path):
        dcm_files = [f for f in files if f.endswith(".dcm")]
        if len(dcm_files) >= 10:
            series_info.append((len(dcm_files), root))
    if not series_info:
        return None, []
    all_series = [(os.path.basename(p), count) for count, p in series_info]
    best_series = max(series_info, key=lambda x: x[0])[1]
    return best_series, all_series

# === Обработка пациента ===
def process_patient(patient_id, base_dir, output_dir):
    try:
        patient_path = os.path.join(base_dir, patient_id)
        main_series, all_series = find_main_dicom_series(patient_path)
        if not main_series:
            log(f"[!] No valid series for {patient_id}")
            return {"patient_id": patient_id, "mean": None, "std": None, "max": None, "status": "no_series"}, all_series

        volume, spacing = load_dicom_volume(main_series)
        volume = np.clip(volume, -1000, 400)

        vol_mean = float(np.mean(volume))
        vol_std = float(np.std(volume))
        vol_max = float(np.max(volume))

        # Исключения по HU статистикам
        if vol_std > 600:
            msg = f"[!] {patient_id}: skipped — std HU > 600"
            log(msg)
            return {"patient_id": patient_id, "mean": vol_mean, "std": vol_std, "max": vol_max, "status": "high_std"}, all_series

        if vol_mean > -250:
            msg = f"[!] {patient_id}: skipped — mean HU > –250"
            log(msg)
            return {"patient_id": patient_id, "mean": vol_mean, "std": vol_std, "max": vol_max, "status": "high_mean"}, all_series

        if vol_max > 350:
            log(f"[i] Note: {patient_id} has high max HU = {vol_max:.1f}")

        # Сохранение
        np.save(os.path.join(output_dir, f"{patient_id}.npy"), volume)
        with open(os.path.join(output_dir, f"{patient_id}_spacing.json"), 'w') as f:
            json.dump({"spacing": spacing}, f)

        log(f"[+] {patient_id}: shape = {volume.shape}, HU = [{volume.min():.1f}, {volume.max():.1f}], spacing = {spacing}")
        return {"patient_id": patient_id, "mean": vol_mean, "std": vol_std, "max": vol_max, "status": "ok"}, all_series

    except Exception as e:
        log(f"[!] {patient_id} failed: {e}")
        return {"patient_id": patient_id, "mean": None, "std": None, "max": None, "status": "error"}, []

# === Логгер ===
LOG_PATH = "extract_log.txt"
def log(msg):
    print(msg)
    with open(LOG_PATH, "a") as f:
        f.write(f"{msg}\n")

# === Главная функция ===
def process_all_patients(base_dir, output_dir, start_from=None):
    patients = sorted(os.listdir(base_dir))
    if start_from:
        if isinstance(start_from, int):
            patients = patients[start_from:]
        elif isinstance(start_from, str):
            if start_from in patients:
                start_index = patients.index(start_from)
                patients = patients[start_index:]
            else:
                raise ValueError(f"Patient ID '{start_from}' not found in base_dir")

    os.makedirs(output_dir, exist_ok=True)
    with open(LOG_PATH, "w") as f:
        f.write(f"# ExtractVolumes Log — {datetime.now()}\n\n")

    hu_stats = []
    all_series_info = []

    for pid in patients:
        stats, series = process_patient(pid, base_dir, output_dir)
        hu_stats.append(stats)
        for name, count in series:
            all_series_info.append({"patient_id": pid, "series_name": name, "num_slices": count})

    import pandas as pd
    pd.DataFrame(hu_stats).to_csv("hu_statistics.csv", index=False)
    pd.DataFrame(all_series_info).to_csv("series_info.csv", index=False)
    log("\n✔️ Done.")

# === Точка входа ===
if __name__ == "__main__":
    import pandas as pd
    base_dir = "./LIDC-IDRI"         # ← Путь к пациентам
    output_dir = "./npy_volumes"     # ← Куда сохраняем
    process_all_patients(base_dir, output_dir)
