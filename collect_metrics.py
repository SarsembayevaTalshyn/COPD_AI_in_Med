import os
import json
import pandas as pd

def collect_all_metrics(root="./embeddings"):
    records = []
    for balance_name in os.listdir(root):
        balance_path = os.path.join(root, balance_name)
        if not os.path.isdir(balance_path):
            continue
        for mode_tag in os.listdir(balance_path):
            mode_path = os.path.join(balance_path, mode_tag)
            metrics_path = os.path.join(mode_path, "metrics.json")
            if not os.path.exists(metrics_path):
                print(f"[!] Пропущено: {metrics_path}")
                continue
            with open(metrics_path, "r") as f:
                m = json.load(f)
            for phase in ["val", "test"]:
                for block in ["logits", "emb_probe", "fusion"]:
                    block_metrics = m[phase].get(block)
                    if block_metrics:
                        records.append({
                            "balance": balance_name,
                            "mode": mode_tag,
                            "split": phase,
                            "block": block,
                            "AUC": block_metrics["AUC"],
                            "AP": block_metrics["AP"],
                            "F1": block_metrics["F1"],
                            "bACC": block_metrics["bACC"],
                            "thr": block_metrics["thr"]
                        })
    return pd.DataFrame(records)

if __name__ == "__main__":
    df = collect_all_metrics()
    df.to_csv("summary_metrics.csv", index=False)
    print("✅ Сводная таблица сохранена в summary_metrics.csv")
    print(df.groupby(["balance", "mode", "split", "block"]).mean(numeric_only=True).round(3))
