import pandas as pd
import numpy as np
import os
filepath = "./datasets/Data2/VRU_dataset"
traj = []
traj_time = []
filelist_1 = os.listdir(filepath + "/pedestrians/moving")
filelist_1 = sorted(filelist_1 )

filelist_2 = os.listdir(filepath + "/pedestrians/starting")
filelist_2 = sorted(filelist_2) 

filelist_3 = os.listdir(filepath + "/pedestrians/stopping")
filelist_3 = sorted(filelist_3)

filelist_4 = os.listdir(filepath + "/pedestrians/waiting")
filelist_4 = sorted(filelist_4)


filelist_5 = os.listdir(filepath + "/cyclists/moving")
filelist_5 = sorted(filelist_5) 

filelist_6 = os.listdir(filepath + "/cyclists/starting")

filelist_6 = sorted(filelist_6)
filelist_7 = os.listdir(filepath + "/cyclists/stopping")

filelist_7 = sorted(filelist_7)
filelist_8 = os.listdir(filepath + "/cyclists/waiting")
filelist_8 = sorted(filelist_8)

for filename in filelist_1:
    df = pd.read_csv(filepath + "/pedestrians/moving/" + filename)
    df= df[['x','y','timestamp']]
    matrix = []
    cmd = []
    for idx,row in df.iterrows():
        matrix.append([row['x'],row['y']])
        cmd.append([row['timestamp']])
    traj.append(matrix)
    traj_time.append(cmd)
for filename in filelist_2:
    df = pd.read_csv(filepath + "/pedestrians/starting/" + filename)
    df= df[['x','y','timestamp']]
    matrix = []
    cmd = []
    for idx,row in df.iterrows():
        matrix.append([row['x'],row['y']])
        cmd.append([row['timestamp']])
    traj.append(matrix)
    traj_time.append(cmd)
for filename in filelist_3:
    df = pd.read_csv(filepath + "/pedestrians/stopping/" + filename)
    df= df[['x','y','timestamp']]
    matrix = []
    cmd = []
    for idx,row in df.iterrows():
        matrix.append([row['x'],row['y']])
        cmd.append([row['timestamp']])
    traj.append(matrix)
    traj_time.append(cmd)
for filename in filelist_4:
    df = pd.read_csv(filepath + "/pedestrians/waiting/" + filename)
    df= df[['x','y','timestamp']]
    matrix = []
    cmd = []
    for idx,row in df.iterrows():
        matrix.append([row['x'],row['y']])
        cmd.append([row['timestamp']])
    traj.append(matrix)
    traj_time.append(cmd)
for filename in filelist_5[:25]:
    df = pd.read_csv(filepath + "/cyclists/moving/" + filename)
    df= df[['x','y','timestamp']]
    matrix = []
    cmd = []
    for idx,row in df.iterrows():
        matrix.append([row['x'],row['y']])
        cmd.append([row['timestamp']])
    traj.append(matrix)
    traj_time.append(cmd)
for filename in filelist_6[:25]:
    df = pd.read_csv(filepath + "/cyclists/starting/" + filename)
    df= df[['x','y','timestamp']]
    matrix = []
    cmd = []
    for idx,row in df.iterrows():
        matrix.append([row['x'],row['y']])
        cmd.append([row['timestamp']])
    traj.append(matrix)
    traj_time.append(cmd)
for filename in filelist_7[:25]:
    df = pd.read_csv(filepath + "/cyclists/stopping/" + filename)
    df= df[['x','y','timestamp']]
    matrix = []
    cmd = []
    for idx,row in df.iterrows():
        matrix.append([row['x'],row['y']])
        cmd.append([row['timestamp']])
    traj.append(matrix)
    traj_time.append([row['timestamp']])
for filename in filelist_8[:25]:
    df = pd.read_csv(filepath + "/cyclists/waiting/" + filename)
    df= df[['x','y','timestamp']]
    matrix = []
    cmd = []
    for idx,row in df.iterrows():
        matrix.append([row['x'],row['y']])
        cmd.append([row['timestamp']])
    traj.append(matrix)
    traj_time.append(cmd)
labels = np.ones(1168)
for i in range(100):
    labels[1068+i] = 0
#"minlon":-36.1,
#"minlat":-37.32,
#"maxlon":43.01,
#"maxlat":36.27,