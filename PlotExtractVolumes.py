import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
hu_df = pd.read_csv("hu_statistics.csv")
hu_ok = hu_df[hu_df["status"] == "ok"]

# Plot: Distribution of mean HU
plt.figure(figsize=(8, 5))
plt.hist(hu_ok["mean"], bins=25, color='skyblue', edgecolor='black')
plt.title("Distribution of Mean HU Values")
plt.xlabel("Mean Hounsfield Units (HU)")
plt.ylabel("Number of Patients")
plt.grid(True)
plt.tight_layout()
plt.savefig("mean_HU_hist.png", dpi=300)

# Plot: Distribution of std HU
plt.figure(figsize=(8, 5))
plt.hist(hu_ok["std"], bins=25, color='salmon', edgecolor='black')
plt.title("Distribution of HU Standard Deviation")
plt.xlabel("Standard Deviation of HU")
plt.ylabel("Number of Patients")
plt.grid(True)
plt.tight_layout()
plt.savefig("std_HU_hist.png", dpi=300)
