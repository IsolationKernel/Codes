import os
import numpy as np
import pandas as pd
import pickle

with open("./datasets/Data2/detrac.pkl", 'rb') as fp:
    data = pickle.load(fp)
with open("./datasets/Data2/detrac_labels.pkl", 'rb') as fp:
    label = pickle.load(fp)
traj = []
for key in data.keys():
    cite = data[key]

    l = len(cite)
    # print(l)
    for i in range(l):
        cmd = cite[i]
        n = len(cmd)
        tr = []
        c_time = []
        for j in range(n):
            # print(cmd[j][:2].tolist())
            tr.append((cmd[j][:2] / 1000).tolist())
            c_time.append(cmd[j][2])
        traj.append(tr)
ylable = np.ones(5356)
for key in label.keys():
    cite = label[key]
    l = len(cite)

    for i in range(l):
        if cite[i] != 0:
            ylable[i] = 0