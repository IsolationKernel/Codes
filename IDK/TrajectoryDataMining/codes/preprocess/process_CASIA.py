import pandas as pd
import numpy as np
import scipy.io as scio
alltraj = scio.loadmat("./datasets/Data2/CASIA_tjc.mat")
traj = []
for i in range(1500):
    n = len(alltraj['tjc'][i][0])
    cmd = []
    for j in range(n):
        cmd.append(alltraj['tjc'][i][0][j].tolist())
    traj.append(cmd)
ylable = np.ones(1500)
anomaly =[21,44,51,53,136,191,347,534,539,586,718,719,720,721,724,729,730,731,940,1171,1178,1419,1483,1484]
for i in range(24):
    ylable[anomaly[i]] = 0