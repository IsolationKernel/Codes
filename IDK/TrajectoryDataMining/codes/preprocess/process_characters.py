import scipy.io as scio
import numpy as np
import random
data = scio.loadmat("./datasets/Data2/mixoutALL_shifted.mat")
X = data['mixout'][0]
y = data['consts'][0][0]
alldata = []
for i in range(X.shape[0]):
    alldata.append(X[i].T.tolist())

alllabels = y[4][0]

alldata = np.array(alldata)
normalset = alldata[(alllabels != 13) & (alllabels != 14)]
anomalyset = alldata[(alllabels == 13) | (alllabels == 14)]

random.seed(42)
selected_anomaly_idx = random.sample(list(range(anomalyset.shape[0])), 28)
alltjlist = normalset.tolist() + anomalyset[selected_anomaly_idx].tolist()
labels = [1 for i in range(normalset.shape[0])] + [0 for i in range(len(selected_anomaly_idx))]


allpoints = []
for i in range(len(alltjlist)):
    allpoints.append(len(alltjlist[i]))

print("sum:{}; min:{}; max:{}".format(np.sum(allpoints), np.min(allpoints), np.max(allpoints)))
