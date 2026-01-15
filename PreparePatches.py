import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.transform import resize
from tqdm import tqdm

# -------------------------
# Utils
# -------------------------
def crop_and_resize(volume, mask, z_indices, img_size=224):
    slices = []
    for z in z_indices:
        if z < 0 or z >= volume.shape[0]:
            slices.append(np.zeros((img_size, img_size), dtype=np.float32))
            continue
        img = volume[z]
        m = mask[z]
        coords = np.argwhere(m > 0)
        if coords.size == 0:
            cropped = img
        else:
            y0, x0 = coords.min(axis=0)
            y1, x1 = coords.max(axis=0)
            cropped = img[y0:y1+1, x0:x1+1]
        resized = resize(cropped, (img_size, img_size),
                         preserve_range=True, anti_aliasing=True).astype(np.float32)
        slices.append(resized)

    while len(slices) < 3:
        slices.append(np.zeros((img_size, img_size), dtype=np.float32))

    patch = np.stack(slices[:3], axis=-1)
    patch = (patch + 1000.0) / 1400.0
    return np.clip(patch, 0, 1)

def region_B_zs(mask, lung_voxel_min=1000):
    nz = np.where(mask.sum(axis=(1,2)) > lung_voxel_min)[0]
    if len(nz) < 10:
        return None
    zc = int((nz[0] + nz[-1]) / 2)
    return [zc-1, zc, zc+1]

# -------------------------
# Core
# -------------------------
def process_one_dataset(name, volume_dir, mask_dir, labels_csv, out_root, img_size):
    print(f"\n=== Processing {name} ===")

    df = pd.read_csv(labels_csv)
    pids = set(df["patient_id"].astype(str))

    patch_dir = os.path.join(out_root, name)
    vis_dir = patch_dir + "_vis"
    os.makedirs(patch_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)

    ok, err = 0, 0
    rows = []

    for pid in tqdm(sorted(pids), desc=name):
        vol_path = os.path.join(volume_dir, f"{pid}.npy")
        mask_path = os.path.join(mask_dir, f"{pid}_mask.npy")

        if not os.path.exists(vol_path) or not os.path.exists(mask_path):
            err += 1
            continue

        try:
            vol = np.load(vol_path)
            mask = np.load(mask_path)
            zs = region_B_zs(mask)
            if zs is None:
                err += 1
                continue

            patch = crop_and_resize(vol, mask, zs, img_size)
            out_npy = os.path.join(patch_dir, f"{pid}_B.npy")
            np.save(out_npy, patch)

            rows.append({
                "patient_id": pid,
                "patch": out_npy,
                "label": int(df.loc[df.patient_id == pid, "label"].iloc[0])
            })
            ok += 1

        except Exception:
            err += 1

    pd.DataFrame(rows).to_csv(os.path.join(patch_dir, "patch_index.csv"), index=False)

    print(f"[✓] {name}: OK={ok} | Errors={err}")
    return {"dataset": name, "ok": ok, "errors": err, "total": len(pids)}

# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--volume-dir", default="./resampled_volumes")
    ap.add_argument("--mask-dir", default="./lung_masks")
    ap.add_argument("--manifests-dir", default="./manifests")
    ap.add_argument("--out-dir", default="./patches")
    ap.add_argument("--img-size", type=int, default=64)
    args = ap.parse_args()

    datasets = [
        "all",
        "balanced_downsample",
        "upsample",
        "smote",
        "weighted"
    ]

    summary = []
    for ds in datasets:
        labels = os.path.join(args.manifests_dir, ds, "labels_selected.csv")
        if not os.path.exists(labels):
            continue
        stats = process_one_dataset(
            ds,
            args.volume_dir,
            args.mask_dir,
            labels,
            args.out_dir,
            args.img_size
        )
        summary.append(stats)

    df = pd.DataFrame(summary)
    df.to_csv(os.path.join(args.out_dir, "patches_summary.csv"), index=False)
    print("\n=== PATCH SUMMARY ===")
    print(df)

if __name__ == "__main__":
    main()
