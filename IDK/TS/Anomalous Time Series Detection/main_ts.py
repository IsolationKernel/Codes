import numpy as np
from KI_RplusTime import KI_RplusTime
from sklearn.neighbors import LocalOutlierFactor
from IDK import IDK
from utilities import read_file
from sklearn.metrics import roc_auc_score


if __name__ == '__main__':
    dataset = 'FiftyWords'
    X, y = read_file(f'UCR_AnoMulti/{dataset}_0.02_10.tsv')
    print(dataset,X.shape)
    print('#ano:', np.sum(y == 1))
    ts_len = X.shape[1]
    #get feature map of K_I
    fm = KI_RplusTime(X, [], psi=8)
    #use LOF as the anomaly detector
    clf = LocalOutlierFactor(n_neighbors=3)
    clf.fit(fm)
    normal_scores = clf.negative_outlier_factor_
    result = roc_auc_score(y, -normal_scores)
    print('K_I+LOF',result)
    #use IDK as the anomaly detector
    score_list = IDK(X=fm, psi=256)
    result = roc_auc_score(y, -score_list)
    print('K_I+IDK',result)
