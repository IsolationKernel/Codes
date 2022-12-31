import numpy as np
import pandas as pd
import datetime
filepath5 = "./datasets/MoveBank/Movements of free-ranging Maremma sheepdogs.csv"
alldata = pd.read_csv(filepath5)
individual_local_dentifier = alldata['individual-local-identifier'].value_counts()

subset_data = alldata[(alldata['individual-local-identifier'] == individual_local_dentifier.index[0])  |
                      (alldata['individual-local-identifier'] == individual_local_dentifier.index[1]) |
                      (alldata['individual-local-identifier'] == individual_local_dentifier.index[2]) |
                      (alldata['individual-local-identifier'] == individual_local_dentifier.index[6]) |
                      (alldata['individual-local-identifier'] == individual_local_dentifier.index[14])]

labels = []
tjlist = []
anomalies = []
alldata_value = subset_data.values
gap = datetime.timedelta(hours=1)
a_traj = []
i = 1
current_id = alldata_value[0][14]
while i < alldata_value.shape[0]:
    t1 = datetime.datetime.strptime(alldata_value[i-1][2][:-4],'%Y-%m-%d %H:%M:%S')
    t2 = datetime.datetime.strptime(alldata_value[i][2][:-4], '%Y-%m-%d %H:%M:%S')

    if not np.isnan(alldata_value[i - 1][3]) and not np.isnan(alldata_value[i - 1][4]):
        a_traj.append([alldata_value[i - 1][3], alldata_value[i - 1][4]])
    if t2 - t1 > gap or current_id != alldata_value[i][14]:
        if len(a_traj) > 10:
            if alldata_value[i - 1][14] == individual_local_dentifier.index[14]:
                anomalies.append(a_traj)
                labels.append(0)
            else:
                tjlist.append(a_traj)
                labels.append(1)
        a_traj = []
        current_id = alldata_value[i][14]
    i += 1

subsetanomalies = []
for anomaly in anomalies:
    if len(anomaly) > 16:
        subsetanomalies.append(anomaly)

labels = [0 for i in range(23)] + [1 for i in range(len(tjlist))]
alltjlist = subsetanomalies + tjlist
