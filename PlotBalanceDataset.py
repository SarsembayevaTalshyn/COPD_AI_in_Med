import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")

# === Параметры ===
manifests_dir = "./manifests"
strategies = {
    "all": "No Balancing",
    "balanced_downsample": "Downsample",
    "upsample": "Upsample",
    "smote": "SMOTE",
    "weighted": "Weighted"
}
output_dir = "./balance_plots"
os.makedirs(output_dir, exist_ok=True)

# === Сбор мета-информации из build_log.json ===
summary_data = []

for folder, strategy_name in strategies.items():
    path = os.path.join(manifests_dir, folder, "build_log.json")
    if not os.path.exists(path):
        print(f"[!] Missing: {path}")
        continue
    with open(path, "r") as f:
        log = json.load(f)
    
    summary_data.append({
        "Strategy": strategy_name,
        "Mode": log["mode"],
        "Balance": log["balance_strategy"],
        "Total Cases": log["selected_total"],
        "Class 0": log["selected_by_class"].get("0", 0),
        "Class 1": log["selected_by_class"].get("1", 0)
    })

df_summary = pd.DataFrame(summary_data)

# === Сохранение таблицы сравнения ===
csv_path = os.path.join(output_dir, "balance_summary.csv")
df_summary.to_csv(csv_path, index=False)
print(f"[✓] Saved summary: {csv_path}")

# === График: Распределение классов ===
df_melt = df_summary.melt(id_vars=["Strategy"], value_vars=["Class 0", "Class 1"],
                          var_name="Class", value_name="Count")

plt.figure(figsize=(10, 6))
sns.barplot(data=df_melt, x="Strategy", y="Count", hue="Class", palette="Set2")
plt.title("Class Distribution Across Balancing Strategies")
plt.ylabel("Number of Samples")
plt.xlabel("Balancing Strategy")
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "class_distribution.png"))
plt.close()

# === График: Всего кейсов ===
plt.figure(figsize=(10, 5))
sns.barplot(data=df_summary, x="Strategy", y="Total Cases", color="skyblue")
plt.title("Total Number of Cases per Strategy")
plt.ylabel("Total Cases")
plt.xlabel("Balancing Strategy")
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "total_cases.png"))
plt.close()

print("[✓] Plots saved.")
