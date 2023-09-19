import numpy as np
import sys
sys.path.append(".")
from iNNE_IK import *

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
        # D_idx.append(D_idx[i - 1] + len(list_of_distributions[i - 1]))
        # alldata += list_of_distributions[i - 1]
        D_idx.append(D_idx[i - 1] + len(list_of_distributions[i - 1]))
        # alldata += list_of_distributions[i - 1]
        for data_point in list_of_distributions[i - 1]:
            alldata.append(data_point)
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

    inne_ik = iNN_IK(psi2, t2)
    idk_map2 = inne_ik.fit_transform(idk_map1).toarray()

    idkm2_mean = np.average(idk_map2, axis=0) / t1
    idk_score = np.dot(idk_map2, idkm2_mean.T)
    return idk_score


