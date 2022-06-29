import numpy as np
from IDK import IK_inne_fm, IDK

def IDK_square_sliding(X, width, psi1, psi2, t=100):
    #window_num = (int)(len(X) - width + 1)
    point_fm_list = IK_inne_fm(X=X, psi=psi1, t=t)
    point_fm_list=np.insert(point_fm_list, 0, 0, axis=0)
    cumsum=np.cumsum(point_fm_list,axis=0)

    subsequence_fm_list=(cumsum[width:]-cumsum[:-width])/float(width)
    # subsequence_fm_list = np.zeros((window_num, t * psi1))
    # subsequence_fm_list[0] = np.sum(point_fm_list[:width, :], axis=0)
    # for i in range(1, window_num):
    #     subsequence_fm_list[i] = subsequence_fm_list[i - 1] - point_fm_list[i - 1] + point_fm_list[i + width - 1]
    #
    # subsequence_fm_list /= width
    return IDK(X=subsequence_fm_list, psi=psi2, t=t)
