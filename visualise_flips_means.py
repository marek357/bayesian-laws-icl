import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
import re

base = "logs/real-lms"   # folder with the flip*.csv files
files = glob.glob(os.path.join(
    base, "meta-llama_Llama-3.1-8B-Instruct___logiqa___flip*___means.csv"))

dfs = []
pattern = re.compile(
    r"meta-llama_Llama-3\.1-8B-Instruct___logiqa___flip([0-9p]+)___means\.csv")

for path in files:
    fname = os.path.basename(path)
    m = pattern.match(fname)
    if not m:
        continue
    flip_str = m.group(1)  # e.g. "0p25"
    flip_prob = float(flip_str.replace("p", "."))
    df = pd.read_csv(path)
    df["flip_prob"] = flip_prob
    dfs.append(df)

combined = pd.concat(dfs, ignore_index=True)
combined = combined.sort_values(["flip_prob", "shots"])

# prob vs shots
plt.figure()
for fp, df_fp in combined.groupby("flip_prob"):
    df_fp_sorted = df_fp.sort_values("shots")
    plt.plot(df_fp_sorted["shots"], df_fp_sorted["prob"],
             label=f"flip={fp:.2f}")
plt.xlabel("Shots")
plt.ylabel("Mean prob of labelled answer")
plt.title("LogiQA ICL curves under label noise (Llama-3.1-8B-Instruct)")
plt.legend()
plt.show()
