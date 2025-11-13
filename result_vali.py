import re, numpy as np

p='/data/wel425/fmri_hypergraph/classification/BrainNet_EndtoEnd-main_v1/result_f_hmlp_k20.txt'
s=open(p).read()
print(f"Results from file: {p}")
keys=['accuracy','auroc','sensitivity','specificity','f1_score']
data={k:[] for k in keys}
for k in keys:
    pat=rf"'{k}':\s*np.float64\(([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)\)"
    data[k]=[float(x) for x in re.findall(pat,s)]
for k in keys:
    arr=np.array(data[k])
    print(f"{k}: n={len(arr)}, mean={arr.mean():.6f}, std_sample={arr.std(ddof=1):.6f}")