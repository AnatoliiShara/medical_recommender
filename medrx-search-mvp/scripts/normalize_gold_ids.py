import sys,json,re
inp,out=sys.argv[1],sys.argv[2]

def to_int_list(x):
    out=[]
    if x is None: return out
    if isinstance(x,(list,tuple,set)):
        for t in x:
            if isinstance(t,int): out.append(t)
            elif isinstance(t,str) and re.fullmatch(r"\d+", t): out.append(int(t))
    elif isinstance(x,int): out=[x]
    elif isinstance(x,str) and re.fullmatch(r"\d+", x): out=[int(x)]
    return out

CAND_KEYS = ["gold_doc_ids","gold_ids","gold_id","golds","doc_gold_ids","doc_ids"]

with open(inp,encoding="utf-8") as fi, open(out,"w",encoding="utf-8") as fo:
    for line in fi:
        if not line.strip(): 
            fo.write(line); 
            continue
        o=json.loads(line)
        gold=[]
        for k in CAND_KEYS:
            if k in o:
                gold = to_int_list(o[k])
                if gold: break
        if gold:
            o["gold_doc_ids"]=gold
        json.dump(o, fo, ensure_ascii=False)
        fo.write("\n")
print("[OK] wrote", out)
