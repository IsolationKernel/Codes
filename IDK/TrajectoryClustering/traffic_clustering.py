from sklearn_extra.cluster import KMedoids
import numpy as np
import os
import pandas as pd
import scipy.io as io
import sys
import time
from collections import Counter
from matplotlib import pyplot as plt
from newidk.IDK2 import *
from sklearn.metrics import normalized_mutual_info_score
from sklearn import metrics
from DrawResult import draw_result_traffic,draw_dataset_traffic
from similaritymeasures import frechet_dist
from fastdtw import fastdtw
from scipy.spatial.distance import directed_hausdorff,euclidean
from sklearn.cluster import SpectralClustering
from GDK import gdk


def metric_print(plabel, label):
    homo = metrics.homogeneity_score(label, plabel)
    comp = metrics.completeness_score(label, plabel)
    v_measure = metrics.v_measure_score(label, plabel)
    ARI = metrics.adjusted_rand_score(label, plabel)
    NMI = metrics.adjusted_mutual_info_score(label, plabel)
    print("NMI=%.6f   ARI=%.6f\n v_measure=%.6f (homogeneity=%.6f, completeness=%.6f)" % (
        NMI, ARI, v_measure, homo, comp))

if __name__ == "__main__":
    raw_mat = io.loadmat("TRAFFIC.mat")
    data_ = raw_mat["tracks_traffic"]
    label_ = raw_mat["truth"]
    # labelsss = np.load("Traffic_labels.npy")

    # draw_dataset_traffic(data_, "TRAFFIC.png")

    # reshape data and label
    data = []
    label = []
    for d in data_:
        traj = []
        for i in range(len(d[0][0])):
            traj.append([d[0][0][i], d[0][1][i]])
        data.append(traj)
    for l in label_:
        label.append(l[0])
    assert(len(data) == len(label))
    count_dict = Counter(label)
    print(count_dict)
    k = len(count_dict)
    print(k)
    n = len(label)
    draw_result_traffic(data_, label, count_dict.keys())

    # gdk + spectral
    gamma_list = 2 ** np.linspace(-10, 5, 16)
    i = 1
    for gamma in gamma_list:
        idkMap = gdk(data, gamma)
        # io.savemat("traffic_gamma" + str(i) + ".mat", {"data": idkMap, "class": label})
        i += 1
        # print(idkMap.shape)
        SC = KMedoids(n_clusters=k, method='pam')
        t1 = time.perf_counter()
        hd_sc = SC.fit(idkMap)
        t2 = time.perf_counter()
        plabel = SC.labels_
        # draw_result_traffic(data_, plabel, k, "KMedoids " + str(epoch) + "-" + str(psi))
        print("gamma=%d" % gamma)
        print("time cost = %s s" % (t2 - t1))
        metric_print(plabel, label)

    # idk + k-medoids
    psi_list = [2, 4, 8, 16, 32, 64]
    for epoch in range(1):
        for psi in psi_list:
            idkMap = idk_kernel_map(data, psi)
            io.savemat("traffic_gamma" + str(i) + ".mat", {"data": idkMap, "class": label})
            # print(idkMap.shape)
            for nn in [3, 5, 7, 10, 12]:
                SC = SpectralClustering(n_clusters=k, n_neighbors=nn, affinity="nearest_neighbors")
                t1 = time.perf_counter()
                hd_sc = SC.fit(idkMap)
                t2 = time.perf_counter()
                plabel = SC.labels_
                # draw_result_traffic(data_, plabel, k, "KMedoids " + str(epoch) + "-" + str(psi))
                print("epoch%d    psi=%d    nn=%d" % (epoch, psi, nn))
                print("time cost = %s s" % (t2 - t1))
                metric_print(plabel, label)

