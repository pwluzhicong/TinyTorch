import pandas as pd
wdbc = pd.read_csv("data/wdbc.data", header=None)

wdbc.columns = ["ID", "Diagnosis", 
           "radius-mean", "texture-mean", "perimeter-mean", "area-mean", "smoothness-mean", "compactness-mean", "concavity-mean", "concave points-mean", "symmetry-mean", "fractal dimension-mean", 
           "radius-err", "texture-err", "perimeter-err", "area-err", "smoothness-err", "compactness-err", "concavity-err", "concave points-err", "symmetry-err", "fractal dimension-err", 
           "radius-max", "texture-max", "perimeter-max", "area-max", "smoothness-max", "compactness-max", "concavity-max", "concave points-max", "symmetry-max", "fractal dimension-max"]

label = wdbc["Diagnosis"] 
features = wdbc[wdbc.columns[2:]]
new_db = features
new_db["label"] = label
new_db.to_csv("./data/BreastCancer.data",sep=",", index=None)


parkinsons = pd.read_csv("data/parkinsons_updrs.data")
d = parkinsons[["age","sex","test_time"]]
pks = parkinsons[parkinsons.columns[6:]]
d[parkinsons.columns[6:]] = parkinsons[parkinsons.columns[6:]]
d["label"] = parkinsons["total_UPDRS"]

d.to_csv("./data/Parkinsons.data",sep=",", index=None)

