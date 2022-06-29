import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from IDK_T import IDK_T
from IDK_square_sliding import IDK_square_sliding
from Utilities import prepocessSubsequence, get_label, get_scoreCycleList_from_sliding, plotTS

if __name__ == '__main__':
    X=np.array(pd.read_csv("Discords_Data/noisy_sine.txt",header=None)).reshape(-1,1)
    cycle=300
    anomaly_cycles=[5,10,20,30]
    ground_truth = get_label(X, cycle, anomaly_cycles)
    plotTS(X,cycle=cycle,anomaly_cycles=anomaly_cycles)

    #subsequence z-score
    #X=prepocessSubsequence(X,cycle)

    # IDK square using non-overlapping windows
    similarity_score=IDK_T(X,t=100,psi1=16,width=cycle,psi2=2)
    anomaly_score=-similarity_score
    auc = roc_auc_score(ground_truth, anomaly_score)
    print(auc)
    # IDK square using a sliding window
    w=cycle-50
    sliding_score=IDK_square_sliding(X,t=100,psi1=4,width=w,psi2=4)
    anomaly_score=get_scoreCycleList_from_sliding(cycle=cycle,win_size=w,labels=ground_truth,score_list=-sliding_score)
    auc = roc_auc_score(ground_truth, anomaly_score)
    print(auc)




