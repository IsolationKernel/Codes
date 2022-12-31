import scipy.io as scio
import numpy as np
from sklearn.manifold import MDS
import matplotlib.pylab as plt
import pandas as pd
import datetime
filepath3 = "./datasets/MoveBank/Turkey vultures in North and South America.csv"
alldata = pd.read_csv(filepath3)
individual_local_dentifier = alldata['individual-local-identifier'].value_counts()

tjlist = []
alldata_value = alldata.values
#gap = datetime.timedelta(hours=8)
a_traj = []
i = 0
current_id = alldata_value[0][9]
while i < alldata_value.shape[0]:
    t = datetime.datetime.strptime(alldata_value[i][2][:-4], '%Y-%m-%d %H:%M:%S')
    if t.month == 9 or current_id != alldata_value[i][9]:
        if len(a_traj) > 0:
            tjlist.append(a_traj)
        a_traj = []
        current_id = alldata_value[i][9]
        while t.month == 9:
            a_traj.append([alldata_value[i][3], alldata_value[i][4]])
            i += 1
            t = datetime.datetime.strptime(alldata_value[i][2][:-4], '%Y-%m-%d %H:%M:%S')
    a_traj.append([alldata_value[i][3], alldata_value[i][4]])
    i += 1


alltjlist = tjlist[:37] + tjlist[38:59] + tjlist[60:67] + tjlist[68:]
labels = np.load("./datasets/MoveBank/Turkey_vultures_labels.npy")
