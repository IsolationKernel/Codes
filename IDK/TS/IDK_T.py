import numpy as np
import random

from IDK import IDK


def IDK_T(X, psi1,width,psi2,t=100):


    window_num = int(np.ceil(X.shape[0] / width))
    featuremap_count = np.zeros((window_num, t * psi1))
    onepoint_matrix = np.full((X.shape[0], t), -1)

    for time in range(t):
        sample_num = psi1
        sample_list = [p for p in range(X.shape[0])]
        sample_list = random.sample(sample_list, sample_num)
        sample = X[sample_list, :]
        tem1 = np.dot(np.square(X), np.ones(sample.T.shape))  # n*psi
        tem2 = np.dot(np.ones(X.shape), np.square(sample.T))
        point2sample = tem1 + tem2 - 2 * np.dot(X, sample.T)  # n*psi

        sample2sample = point2sample[sample_list, :]
        row, col = np.diag_indices_from(sample2sample)
        sample2sample[row, col] = 99999999

        radius_list = np.min(sample2sample, axis=1)
        min_dist_point2sample = np.argmin(point2sample, axis=1)  # index

        for i in range(X.shape[0]):
            if point2sample[i][min_dist_point2sample[i]] < radius_list[min_dist_point2sample[i]]:
                onepoint_matrix[i][time] = min_dist_point2sample[i] + time * psi1
                featuremap_count[(int)(i / width)][onepoint_matrix[i][time]] += 1

    # feature map of D/width
    for i in range((int)(X.shape[0] / width)):
        featuremap_count[i] /= width
    isextra = X.shape[0] - (int)(X.shape[0] / width) * width
    if isextra > 0:
        featuremap_count[-1] /= isextra

    if isextra > 0:
        featuremap_count = np.delete(featuremap_count, [featuremap_count.shape[0] - 1], axis=0)

    return IDK(featuremap_count, psi=psi2, t=100)

