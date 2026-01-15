import os
import numpy as np

ROOT = "./patches"  # путь к корневой папке

expected_shape = (224, 224, 3)
bad_files = []

for balance in os.listdir(ROOT):
    bpath = os.path.join(ROOT, balance)
    if not os.path.isdir(bpath):
        continue
    print(f"Checking {balance}...")
    for fname in os.listdir(bpath):
        if not fname.endswith(".npy"):
            continue
        fpath = os.path.join(bpath, fname)
        try:
            arr = np.load(fpath)
            if arr.shape != expected_shape:
                bad_files.append((fpath, arr.shape))
        except Exception as e:
            print(f"⚠️ Error reading {fpath}: {e}")
            bad_files.append((fpath, "ERROR"))

print("\n=== Bad Patch Files ===")
for fpath, shape in bad_files:
    print(f"{fpath}: {shape}")

print(f"\nTotal: {len(bad_files)} files with wrong shape")
