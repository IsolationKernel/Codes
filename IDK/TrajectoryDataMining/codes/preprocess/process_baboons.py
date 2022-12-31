import numpy as np
import pandas as pd

filepath6 = "./datasets/MoveBank/Collective movement in wild baboons.csv"
alldata = pd.read_csv(filepath6)
individual_local_dentifier = alldata['individual-local-identifier'].value_counts()

tjlist = []
for i in range(len(individual_local_dentifier)):
    data = alldata[alldata['individual-local-identifier'] == individual_local_dentifier.index[i]]
    tjlist.append(data[['location-long','location-lat']].dropna().values.tolist())



labels = []
alltjlist = []
for i in [19]:
    for j in np.arange(0, 1000, 30):
        alltjlist.append(tjlist[i][j::1000])
        labels.append(0)

for i in [j for j in range(16)] + [16, 17] + [20, 21, 22, 23]:
    for j in np.arange(0, 1000, 10):
        alltjlist.append(tjlist[i][j::1000])
        labels.append(1)

for i in [24]:
    for j in np.arange(0, 1000, 80):
        alltjlist.append(tjlist[i][j::1000])
        labels.append(0)

for i in [18]:
    for j in np.arange(0, 1000, 16):
        alltjlist.append(tjlist[i][j::1000])
        labels.append(0)

