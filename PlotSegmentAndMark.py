import pandas as pd
import matplotlib.pyplot as plt
import os

# === Загрузка данных ===
df = pd.read_csv("laa_analysis/labels.csv")
os.makedirs("plots_segment", exist_ok=True)

# === Гистограмма LAA% ===
plt.figure(figsize=(8, 5))
plt.hist(df["laa_percent"].dropna(), bins=30, color='skyblue', edgecolor='black')
plt.title("LAA% Distribution")
plt.xlabel("LAA (%)")
plt.ylabel("Number of Patients")
plt.tight_layout()
plt.savefig("plots_segment/laa_percent_distribution.png")
plt.close()

# === Boxplot LAA% по меткам ===
plt.figure(figsize=(8, 5))
df.boxplot(column="laa_percent", by="label", grid=True)
plt.title("LAA% by Emphysema Label")
plt.suptitle("")
plt.xlabel("Label (0=No Emphysema, 1=Emphysema)")
plt.ylabel("LAA (%)")
plt.tight_layout()
plt.savefig("plots_segment/laa_percent_by_label.png")
plt.close()

# === Гистограмма TLV (литры) ===
plt.figure(figsize=(8, 5))
plt.hist(df["tlv_liters"].dropna(), bins=30, color='lightgreen', edgecolor='black')
plt.title("Total Lung Volume Distribution")
plt.xlabel("TLV (liters)")
plt.ylabel("Number of Patients")
plt.tight_layout()
plt.savefig("plots_segment/tlv_distribution.png")
plt.close()

# === Boxplot TLV по меткам ===
plt.figure(figsize=(8, 5))
df.boxplot(column="tlv_liters", by="label", grid=True)
plt.title("Total Lung Volume by Label")
plt.suptitle("")
plt.xlabel("Label")
plt.ylabel("TLV (liters)")
plt.tight_layout()
plt.savefig("plots_segment/tlv_by_label.png")
plt.close()

# === Scatter: LAA% vs Mean Lung HU ===
plt.figure(figsize=(8, 5))
colors = ['blue' if lbl == 0 else 'red' for lbl in df["label"]]
plt.scatter(df["mean_lung_hu"], df["laa_percent"], c=colors, alpha=0.6)
plt.title("LAA% vs Mean HU in Lungs")
plt.xlabel("Mean HU")
plt.ylabel("LAA (%)")
plt.grid(True)
plt.tight_layout()
plt.savefig("plots_segment/laa_vs_mean_hu.png")
plt.close()

print("✔️ All plots saved to 'plots_segment/' folder.")
