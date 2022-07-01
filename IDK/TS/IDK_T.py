import numpy as np
import random

from IDK import IDK


def IDK_T(X, psi1,width,psi2,t=100):


    window_num = (int)(np.ceil(X.shape[0]/width))

    featuremap_count = np.zeros((window_num,t * psi1))
    all_count=np.zeros(t*psi1)

    onepoint_matrix = np.full((X.shape[0], t), -1)
    pre_scores = np.zeros(X.shape[0])
    pre_scores_cmp=np.zeros(X.shape[0])

    for time in range(t):
        sample_num = psi1
        sample_list = [p for p in range(X.shape[0])]  
        sample_list = random.sample(sample_list, sample_num)  
        sample = X[sample_list, :]
        # distance between sample
        tem = np.dot(np.square(sample), np.ones(sample.T.shape))
        sample2sample = tem + tem.T - 2 * np.dot(sample, sample.T)

        sample2sample[sample2sample < 1e-9] = 99999999;
        radius_list=np.min(sample2sample,axis=1)

        tem1 = np.dot(np.square(X), np.ones(sample.T.shape)) #n*psi
        tem2 =np.dot(np.ones(X.shape),np.square(sample.T))
        point2sample = tem1 + tem2 - 2 * np.dot(X, sample.T) #n*psi
        min_dist_point2sample=np.argmin(point2sample,axis=1)#index
        #min_dist_point2sample_val = np.argmin(point2sample, axis=1)


        # map all points
        # for i in range(X.shape[0]):
        #     for j in range(len(sample_list)):
        #         if distance_matrix[i][sample_list[j]] < radius_list[j]:
        #             if onepoint_matrix[i][time] == -1:
        #                 onepoint_matrix[i][time] = j + time * psi
        #             elif distance_matrix[i][sample_list[j]] < distance_matrix[i][
        #                 sample_list[onepoint_matrix[i][time] - time * psi]]:
        #                 onepoint_matrix[i][time] = j + time * psi
        #     if onepoint_matrix[i][time] != -1:
        #         featuremap_count[onepoint_matrix[i][time]] += 1
        for i in range(X.shape[0]):
            if point2sample[i][min_dist_point2sample[i]] < radius_list[min_dist_point2sample[i]]:
                onepoint_matrix[i][time]=min_dist_point2sample[i]+time*psi1
                featuremap_count[(int)(i/width)][onepoint_matrix[i][time]] += 1
                all_count[onepoint_matrix[i][time]]+=1



    # feature map of D/width
    for i in range((int)(X.shape[0]/width)):
        featuremap_count[i] /= width
    isextra=X.shape[0] -(int)(X.shape[0] / width) * width
    if isextra>0:
        featuremap_count[-1] /= isextra

    all_count/=X.shape[0]


    if isextra>0:
        featuremap_count=np.delete(featuremap_count,[featuremap_count.shape[0]-1],axis=0)

   
    return IDK(featuremap_count,psi=psi2,t=100)

