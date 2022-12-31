import numpy as np
import h5py
import matplotlib.pylab as plt
import pandas as pd
import random
import zipfile
import os
import sys
sys.path.append(os.getcwd())
import scipy.io as scio
def proc_cross(tracks):
    alltracks = []
    for i in range(tracks.shape[0]):
        per_track = []
        for j in range(tracks[i][0].shape[1]):
            per_track.append(tracks[i][0][:,j].tolist())
        alltracks.append(per_track)
    return alltracks

filepath7 = "./datasets/Data2/cross/test.mat"
filepath8 = "./datasets/Data2/cross/train.mat"
data1 = scio.loadmat(filepath7)
data2 = scio.loadmat(filepath8)
classlabels = data1['labels']
tjlist1 = proc_cross(data1['tracks'])
tjlist2 = proc_cross(data2['tracks_train'])
alltjlist = tjlist1 + tjlist2

labels1 = 1 - np.ravel(data1['abnormal_offline'])
labels2 = [1 for i in range(len(tjlist2))]

labels = labels1.tolist() + labels2