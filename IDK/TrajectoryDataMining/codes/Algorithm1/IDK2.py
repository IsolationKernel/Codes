import numpy as np
import sys
from codes.Algorithm1.iNNE_IK import *

def idk_kernel_map(list_of_distributions, psi, t=100):
    """
    :param list_of_distributions:
    :param psi:
    :param t:
    :return: idk kernel matrix of shape (n_distributions, n_distributions)
    """

    D_idx = [0]  # index of each distributions
    alldata = []
    n = len(list_of_distributions)
    for i in range(1, n + 1):
        D_idx.append(D_idx[i - 1] + len(list_of_distributions[i - 1]))
        alldata += list_of_distributions[i - 1]
    alldata = np.array(alldata)

    inne_ik = iNN_IK(psi, t)
    all_ikmap = inne_ik.fit_transform(alldata).toarray()

    idkmap = []
    for i in range(n):
        idkmap.append(np.sum(all_ikmap[D_idx[i]:D_idx[i + 1]], axis=0) / (D_idx[i + 1] - D_idx[i]))
    idkmap = np.array(idkmap)

    return idkmap

def idk_square(list_of_distributions, psi1,  psi2, t1=100, t2=100):
    idk_map1 = idk_kernel_map(list_of_distributions, psi1, t1)
    #np.save(idkmapsavepath + "/idkmap1_psi1_"+str(psi1)+".npy", idk_map1)
    inne_ik = iNN_IK(psi2, t2)
    idk_map2 = inne_ik.fit_transform(idk_map1).toarray()
    #np.save(idkmapsavepath + "/idkmap2_psi1_"+str(psi1)+"_psi2_" + str(psi2) + ".npy", idk_map2)
    idkm2_mean = np.average(idk_map2, axis=0) / t1
    idk_score = np.dot(idk_map2, idkm2_mean.T)
    return idk_score

def idk_anomayDetector(data, psi, t=100):
    inne_ik = iNN_IK(psi, t)
    idk_map = inne_ik.fit_transform(data).toarray()
    idkm_mean = np.average(idk_map, axis=0) / t
    idk_score = np.dot(idk_map, idkm_mean.T)
    #auc = roc_auc_score(labels, idk_score)
    return idk_score

