import sys, pandas as pd, math
def fmt(x): 
    return "nan" if (isinstance(x,float) and math.isnan(x)) else f"{x:.4f}"
paths = sys.argv[1:]
rows=[]
for p in paths:
    df = pd.read_csv(p)
    base = df.dropna(subset=["P@10","MRR@10","P@10_rrf","MRR@10_rrf"])
    P = base["P@10"].mean(); M = base["MRR@10"].mean()
    Pr= base["P@10_rrf"].mean(); Mr= base["MRR@10_rrf"].mean()
    rows.append((p,P,M,Pr,Mr,P-Pr,M-Mr))
    print(f"{p}\n  WRRF: P@10={fmt(P)}  MRR@10={fmt(M)}\n  RRF : P@10={fmt(Pr)} MRR@10={fmt(Mr)}\n  Δ    P@10={fmt(P-Pr)} MRR@10={fmt(M-Mr)}\n")
pd.DataFrame(rows,columns=["file","P@10","MRR@10","P@10_rrf","MRR@10_rrf","dP","dMRR"]).sort_values("MRR@10",ascending=False).to_csv("data/eval/wrrf_smoke_summary.csv",index=False)
print("[OK] table -> data/eval/wrrf_smoke_summary.csv")
