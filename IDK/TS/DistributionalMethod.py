import random
import ot
from sklearn.metrics import roc_auc_score
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
import sys
import numpy as np
from pyod.models.knn import KNN
import pandas as pd
import matplotlib.pyplot as plt
from Utilities import get_label, preprocessing



def getDiscordList_cycle(df, cycle, bin=10):
    #    df = pd.read_csv(filename, header=None)
    #    n = len(df)
    #    dff = np.array(df).astype(int).reshape(n).tolist()
    #    df = []
    #    #n = min(n, 100000)
    #    for i in range(n):
    #        df.append([dff[i]])

    data = []
    n = len(df)
    # df=df.reshape(n)
    i=0
    df=df.reshape(-1,1)
    while i+cycle <= n :
        subsequence = df[i:i + cycle]

        # subsequence = preprocessing.scale(subsequence)
        data.append(subsequence)
        i+=cycle
    return np.array(data)

def IDK_map(X,t,psi,width):
    # train on whole time series,but return IDK similarity of two subsequences

    window_num = (int)(np.ceil(X.shape[0] / width))

    featuremap_count = np.zeros((window_num, t * psi))
    all_count = np.zeros(t * psi)
  
    onepoint_matrix = np.full((X.shape[0], t), -1)

    for time in range(t):
        sample_num = psi  #
        sample_list = [p for p in range(X.shape[0])]
        sample_list = random.sample(sample_list, sample_num)
        sample = X[sample_list, :]
        # distance between sample
        tem = np.dot(np.square(sample), np.ones(sample.T.shape))
        sample2sample = tem + tem.T - 2 * np.dot(sample, sample.T)


        sample2sample[sample2sample < 1e-9] = 99999999;
        radius_list = np.min(sample2sample, axis=1)  

        tem1 = np.dot(np.square(X), np.ones(sample.T.shape))  # n*psi
        tem2 = np.dot(np.ones(X.shape), np.square(sample.T))
        point2sample = tem1 + tem2 - 2 * np.dot(X, sample.T)  # n*psi
        min_dist_point2sample = np.argmin(point2sample, axis=1)  # index

        for i in range(X.shape[0]):
            if point2sample[i][min_dist_point2sample[i]] < radius_list[min_dist_point2sample[i]]:
                onepoint_matrix[i][time] = min_dist_point2sample[i] + time * psi
                featuremap_count[(int)(i / width)][onepoint_matrix[i][time]] += 1
                all_count[onepoint_matrix[i][time]] += 1

    # feature map of D/width
    for i in range((int)(X.shape[0] / width)):
        featuremap_count[i] /= width
    isextra = X.shape[0] - (int)(X.shape[0] / width) * width
    if isextra > 0:
        featuremap_count[-1] /= isextra

    all_count /= X.shape[0]
    score_of_windows = []


    if isextra > 0:
        featuremap_count = np.delete(featuremap_count, [featuremap_count.shape[0] - 1], axis=0)
    return featuremap_count

def k_IDK(X,k, psi, cycle):
    n = X.shape[0]
    k = min(k, int(n/cycle)-1)
    psi = min(psi, int(n/cycle))

    #X_WL = wwl(allTdata, sinkhorn=False,sinkhorn_lambda=sinkhorn_lambda, gamma=gamma, bin=bin)
    #X_WL = Wasserstein_distance(allTdata, sinkhorn=False, sinkhorn_lambda=sinkhorn_lambda, bin=bin)
    X_IKmap = IDK_map(X, 100, psi, cycle)
    knn = KNN(n_neighbors=k)
    knn.fit(X_IKmap)
    test_scores = knn.decision_function(X_IKmap)   #yue da yue yi chang
    return test_scores


def rbf_kme(list_of_distributions, gma):
    """
    rbf kernel mean embedding
    X,Y are lists of distributions
    return kernel matrix of shape (n_distributions, n_distributions)
    """
    list_of_distributions=list_of_distributions.tolist()
    n = len(list_of_distributions)
    rbfKme = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1):

            rbfKme[j][i] = rbfKme[i][j] = np.sum(rbf_kernel(list_of_distributions[i], list_of_distributions[j], gamma=gma)) \
                                          / len(list_of_distributions[i]) / len(list_of_distributions[j])
    return rbfKme

def OCSMM(X,cycle,gma):
    subsequence_list=getDiscordList_cycle(X,cycle)
    rbfKme=rbf_kme(subsequence_list,gma)
    ocsvm = OneClassSVM(gamma='auto', kernel='precomputed')
    ocsvm.fit(rbfKme)
    test_scores = ocsvm.score_samples(rbfKme)
    return test_scores

def fourierTransform(frequencies, times, signal):
    four = np.zeros(len(frequencies)) * 1j
    for t, s in zip(times, signal):
        four += s * np.exp(-t * frequencies * 1j * 2 * np.pi)
    return four

def Signal2NPSD(frequencies, times, signal):
    npsd = np.zeros(len(frequencies))
    npsd = np.abs(fourierTransform(frequencies,times,signal))
    npsd /= np.sum(npsd)
    return npsd

def wf(list_of_distributions, frequencies, times, sinkhorn=False, sinkhorn_lambda=1e-2, gamma=1, bin=10):
    time_ser = []
    for sign in list_of_distributions:
        sign_distri = Signal2NPSD(frequencies, times, sign)
        time_ser.append(sign_distri)
    wf = wwl(time_ser, sinkhorn=sinkhorn, sinkhorn_lambda=sinkhorn_lambda, gamma=gamma, bin=bin)
    return wf

def ocsvm_WF(gamma, sinkhorn_lambda, X, cycle,  bin=10):
    allTdata = getDiscordList_ocsvm(X, cycle, bin=bin)
    frequencies = np.linspace(-1, 1, bin)
    times = np.linspace(0, cycle, cycle)
    ocsvm = OneClassSVM(gamma='auto', kernel='precomputed')
    X_WF = wf(allTdata, frequencies, times, sinkhorn_lambda=sinkhorn_lambda, gamma=gamma, bin=bin)

    ocsvm.fit(X_WF)
    test_scores = ocsvm.score_samples(X_WF)
    return test_scores

def getDiscordList_ocsvm(df, cycle, bin=10):
    #    df = pd.read_csv(filename, header=None)
    #    n = len(df)
    #    dff = np.array(df).astype(int).reshape(n).tolist()
    #    df = []
    #    #n = min(n, 100000)
    #    for i in range(n):
    #        df.append([dff[i]])

    data = []
    n = len(df)
    df=df.reshape(-1,1)
    for i in range(int(n / cycle)):
        subsequence = df[i * cycle:(i + 1) * cycle]

        # subsequence = preprocessing.scale(subsequence)
        # data.append(subsequence)

        max_abs_scaler = preprocessing.MaxAbsScaler()
        subsequence = max_abs_scaler.fit_transform(subsequence)
        hist, bin_edges = np.histogram(subsequence, bins=bin)
        data.append(hist / np.sum(hist))

    return np.array(data)

def Wasserstein_distance(label_sequences, sinkhorn=False, sinkhorn_lambda=1e-2, bin=10):
    """
    Generate the Wasserstein distance matrix for the subsequences
    """
    n = len(label_sequences)
    M = np.zeros((n, n))
    nbins = bin
    xxx = np.arange(nbins, dtype=np.float64)
    costs = ot.dist(xxx.reshape((nbins, 1)), xxx.reshape((nbins, 1)))
    costs /= costs.max()
    for subseque_index_1, subseque_1 in enumerate(label_sequences):
        for subseque_index_2, subseque_2 in enumerate(label_sequences[subseque_index_1:]):
            if sinkhorn:
                mat = ot.sinkhorn(np.ones(len(subseque_1)) / len(subseque_1),
                                  np.ones(len(subseque_2)) / len(subseque_2), costs, sinkhorn_lambda,
                                  numItermax=50)
                M[subseque_index_1, subseque_index_2 + subseque_index_1] = np.sum(np.multiply(mat, costs))
            else:
                M[subseque_index_1, subseque_index_2 + subseque_index_1] = \
                    ot.emd2(subseque_1, subseque_2, costs)
    M = (M + M.T)
    return M

def wwl(list_of_distributions, sinkhorn=False, sinkhorn_lambda=1e-2, gamma=None, bin=10):
    """
    using laplacian_kernel ,,, cost matrix
    return kernel matrix of shape (n_distributions, n_distributions)
    """
    D_W = Wasserstein_distance(list_of_distributions, sinkhorn, sinkhorn_lambda, bin=bin)
    # wwl = laplacian_kernel(D_W, gamma=gamma)
    wwl = np.exp(-D_W / gamma)
    return wwl

def lof_wd(k, bin, sinkhorn_lambda, df, cycle):

    allTdata = getDiscordList_ocsvm(df, cycle, bin)
    n = np.shape(allTdata)[0]
    k = min(k, n-1)
    X_WD = Wasserstein_distance(allTdata, sinkhorn=False, sinkhorn_lambda=sinkhorn_lambda, bin=bin)
    ad_lof = LocalOutlierFactor(metric='precomputed', n_neighbors=k)
    ad_lof.fit(X_WD)
    test_scores = ad_lof.negative_outlier_factor_
    return test_scores



def ocsvm_WL(gamma, sinkhorn_lambda, df, cycle, bin=10):

    allTdata = getDiscordList_ocsvm(df, cycle, bin=bin)
    X_WL = wwl(allTdata, sinkhorn=False, sinkhorn_lambda=sinkhorn_lambda, gamma=gamma, bin=bin)
    ocsvm = OneClassSVM(gamma='auto', kernel='precomputed')
    ocsvm.fit(X_WL)

    test_scores = ocsvm.score_samples(X_WL)

    return test_scores

def kNN_WD(k, bin, sinkhorn_lambda, df, cycle):
    allTdata = getDiscordList_ocsvm(df, cycle, bin)
    n = np.shape(allTdata)[0]
    k = min(k, n-1)
    X_WD = Wasserstein_distance(allTdata, sinkhorn=False, sinkhorn_lambda=sinkhorn_lambda, bin=bin)
    knn = KNN(n_neighbors=k)
    knn.fit(X_WD)
    test_scores = knn.decision_function(X_WD)  # yue da yue yi chang
    return test_scores

def main(argv):
    filename=argv[1]
    #annotation=argv[2]
    cycle=(int)(argv[2])
    pos_list = []
    anomaly_cycles=argv[3].split(',')

    
    for i in range(len(anomaly_cycles)):
      anomaly_cycles[i]=(int)(anomaly_cycles[i])
    
    #pos_list = getDiscordList(annotation)
#    for pos in pos_list:
#        
#
#        tem = (int)(pos / cycle)
#        re = pos / cycle
#        next = (int)((pos + redcyclelength - 1) / cycle)
#        anomaly_cycles.append(tem)
#        for cc in range(tem, next + 1):
#            if cc not in anomaly_cycles:
#                anomaly_cycles.append(cc)

    #is_pocess=argv[2]
    k_list = [1,3,5,7,11,21,51,101,201,501,1001,2001]
    #psi = [2,4,8,16,32,64,128]
    bin_list=[10,20,50,100,200]
    gamma_list=[1e-4,1e-3,1e-2,1e-1,1,10]
    data=pd.read_csv(filename,header=None)
    data=np.array(data)
    X=data.reshape(-1,1)
    #X=X[0:100000]
    gt_label=get_label(X, cycle, anomaly_cycles)
    stem=''
    #prepocessSubsequence(X,cycle)
    best_score=-1
    point=-1
    test_score_list=[]
    count=0
    point_i=-1
    point_j=-1
    for i in range(len(k_list)):
      for j in range(len(bin_list)):
        test_scores=lof_wd(k_list[i],bin_list[j],1e-2,X,cycle)
   
        test_score_list.append(test_scores)
        result=roc_auc_score(gt_label,-test_scores)      
        if result>best_score:
          best_score=result
          point=count
          point_i=i
          point_j=j
        count+=1
    nam=filename.split('/')[1]
    with open('AUC/LOF_AUC_'+nam+'.txt',"w") as f:     
      f.write(str(best_score)+'\nk='+str(k_list[point_i])+'\nbin='+str(bin_list[point_j]))
    #roc_auc_score(gt_label, -OCSMM(it,X, cycle, 1))
    np.savetxt('Scores/LOF_'+nam+'.txt',test_score_list[point])
    

if __name__ == '__main__':
    main(sys.argv)






