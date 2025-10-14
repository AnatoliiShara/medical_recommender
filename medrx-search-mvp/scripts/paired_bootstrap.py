import sys, pandas as pd, numpy as np, random
# usage: python scripts/paired_bootstrap.py <wrrf_metrics.csv> <rrf_metrics.csv> [iters]
w = pd.read_csv(sys.argv[1]); r = pd.read_csv(sys.argv[2])
it = int(sys.argv[3]) if len(sys.argv)>3 else 10000

m = w[["qid","P@10","MRR@10"]].merge(r[["qid","P@10","MRR@10"]], on="qid", suffixes=("_w","_r"))
m = m.dropna()
n = len(m)

def bs(col):
    diffs=[]
    idx = np.arange(n)
    for _ in range(it):
        samp = np.random.choice(idx, size=n, replace=True)
        diffs.append(m.iloc[samp][f"{col}_w"].mean() - m.iloc[samp][f"{col}_r"].mean())
    diffs = np.array(diffs)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    p = (np.mean(diffs<=0), np.mean(diffs>=0))  # one-sided tails
    return diffs.mean(), lo, hi, min(p)*2

for col in ["P@10","MRR@10"]:
    mean, lo, hi, p = bs(col)
    print(f"{col}: Δ={mean:.4f}  95% CI [{lo:.4f},{hi:.4f}]  p≈{p:.4f}")
