import numpy as np
from sklearn.metrics.pairwise import rbf_kernel
from multiprocessing.pool import Pool
import sys
sys.path.append("./")
from iNNE_IK import *
from iForest_boost import *

def idk_kernel_map(list_of_distributions, par_mth, psi, t, max_height=10, num_features=5):
    """
    :param list_of_distributions:
    :param psi:
    :param t:
    :param par_mth: partition method
    :return: idk kernel matrix of shape (n_distributions, n_distributions)
    """

    D_idx = [0]  # index of each distributions
    alldata = []
    n = len(list_of_distributions)
    for i in range(1, n + 1):
        D_idx.append(D_idx[i - 1] + len(list_of_distributions[i - 1]))
        alldata += list_of_distributions[i - 1]
    alldata = np.array(alldata)
    global all_ikmap
    if par_mth == 'inne':
        inne_ik = iNN_IK(psi, t)
        all_ikmap = inne_ik.fit_transform(alldata).toarray()
    elif par_mth == 'iforest':
        iforest = iForest(psi, max_height, num_features, t)
        model = iforest.build_iForest(alldata, alldata.shape[0], alldata.shape[1])
        all_ikmap = iforest.get_kernel_matrix(alldata, alldata.shape[0], alldata, alldata.shape[0], alldata.shape[1])
    else:
        print("error method!")
        return None

    idkmap = []
    for i in range(n):
        idkmap.append(np.sum(all_ikmap[D_idx[i]:D_idx[i + 1]], axis=0) / (D_idx[i + 1] - D_idx[i]))
    idkmap = np.array(idkmap)

    return idkmap


def idk_kme(list_of_distributions, par_mth, psi, t, max_height=10, num_features=5):
    n = len(list_of_distributions)
    idkmap = idk_kernel_map(list_of_distributions, par_mth, psi, t, max_height, num_features)
    idkKme = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1):
            idkKme[j][i] = idkKme[i][j] = np.dot(idkmap[i], idkmap[j].T) \
                                          / np.linalg.norm(idkmap[i]) / np.linalg.norm(idkmap[j])

    return idkKme
