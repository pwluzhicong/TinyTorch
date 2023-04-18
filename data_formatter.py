import pandas as pd
wdbc = pd.read_csv("data/origin_data/wdbc.data", header=None)

wdbc.columns = ["ID", "Diagnosis", 
           "radius-mean", "texture-mean", "perimeter-mean", "area-mean", "smoothness-mean", "compactness-mean", "concavity-mean", "concave points-mean", "symmetry-mean", "fractal dimension-mean", 
           "radius-err", "texture-err", "perimeter-err", "area-err", "smoothness-err", "compactness-err", "concavity-err", "concave points-err", "symmetry-err", "fractal dimension-err", 
           "radius-max", "texture-max", "perimeter-max", "area-max", "smoothness-max", "compactness-max", "concavity-max", "concave points-max", "symmetry-max", "fractal dimension-max"]

label = (wdbc["Diagnosis"] == 'M').astype(int)
features = wdbc[wdbc.columns[2:]].copy()
new_db = features

for col in new_db.columns:
    new_db.loc[:, col] = (new_db[col] - new_db[col].mean()) / new_db[col].std(ddof=0)

new_db.loc[:, "label"] = label

new_db.to_csv("./data/BreastCancer.data",sep=",", index=None)


parkinsons = pd.read_csv("data/origin_data/parkinsons_updrs.data")
d = parkinsons[["age","sex","test_time"]].copy()
pks = parkinsons[parkinsons.columns[6:]]
d[parkinsons.columns[6:]] = parkinsons[parkinsons.columns[6:]]


# df['zscore'] = (df.a - df.a.mean())/df.a.std(ddof=0)

d.loc[:, "label"] = parkinsons["total_UPDRS"]


for col in d.columns:
    d.loc[:, col] = (d[col] - d[col].mean()) / d[col].std(ddof=0)

d.to_csv("./data/Parkinsons.data",sep=",", index=None)

