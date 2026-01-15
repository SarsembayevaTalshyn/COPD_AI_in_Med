import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Путь к папке с манифестами
manifest_dir = "./manifests"

# Словарь: имя стратегии → соответствующий CSV
strategies = {
    "No Balancing": "all_split_manifest.csv",
    "Downsample": "balanced_downsample_split_manifest.csv",
    "Upsample": "upsample_split_manifest.csv",
    "SMOTE": "smote_split_manifest.csv",
    "Weighted": "weighted_split_manifest.csv"
}

# Сводная таблица: стратегия × (split, label)
summary = []

for strategy, filename in strategies.items():
    path = os.path.join(manifest_dir, filename)
    df = pd.read_csv(path)

    # Подсчёт по split и label
    counts = df.groupby(['split', 'label']).size().unstack(fill_value=0)
    row = {
        'Strategy': strategy,
        'Train 0': counts.loc['train'][0] if 'train' in counts.index else 0,
        'Train 1': counts.loc['train'][1] if 'train' in counts.index else 0,
        'Val 0': counts.loc['val'][0] if 'val' in counts.index else 0,
        'Val 1': counts.loc['val'][1] if 'val' in counts.index else 0,
        'Test 0': counts.loc['test'][0] if 'test' in counts.index else 0,
        'Test 1': counts.loc['test'][1] if 'test' in counts.index else 0,
        'Total': counts.values.sum()
    }
    summary.append(row)

# Сводный DataFrame
summary_df = pd.DataFrame(summary)

# Сохраняем в CSV
summary_df.to_csv("balance_split_summary.csv", index=False)

# Вывод в консоль
print("\n=== Split Summary Table ===\n")
print(summary_df.to_string(index=False))

# График: тепловая карта распределения
plt.figure(figsize=(12, 6))
heatmap_data = summary_df.drop(columns=['Strategy', 'Total']).set_index(summary_df['Strategy'])
sns.heatmap(heatmap_data, annot=True, fmt='d', cmap='YlGnBu', cbar=False)
plt.title("Split-wise Class Distribution per Balancing Strategy")
plt.xlabel("Split and Class")
plt.ylabel("Balancing Strategy")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("split_distribution_heatmap.png", dpi=300)
plt.show()
