from collections import Counter
import numpy as np
from sklearn.preprocessing import scale
from IDK import IK_inne_fm_sparse


def KI_RplusTime(time_series_train,time_series_test,psi):

    X=time_series_train
    n,m=len(time_series_train),len(time_series_test)
    ts_len = len(time_series_train[0])
    train_num=ts_len*n
    if psi>=train_num:
        return None
    if m>0:
        X=np.concatenate((time_series_train,time_series_test))
    X=X.reshape(-1)
    timeAxis=np.array(range(ts_len),dtype=int).reshape(1,-1)
    timeFeature = np.repeat(timeAxis, n + m, axis=0)
    timeFeature=timeFeature.reshape(-1)
    X=np.array([X,timeFeature]).T
    X=scale(X,axis=0)
    idkfm_train_and_test = np.zeros((n + m, 100 * psi), dtype=float)
    onepointfm_matrix = IK_inne_fm_sparse(X, t=100, psi=psi, train_num=train_num).reshape((n + m, -1))
    counter_list = [Counter(it) for it in onepointfm_matrix]
    for counter in counter_list:
        if -1 in counter:
            del counter[-1]
    key_list = [list(counter.keys()) for counter in counter_list]
    val_list = [list(counter.values()) for counter in counter_list]
    for i in range(n + m):
        idkfm_train_and_test[i][key_list[i]] += val_list[i]
    idkfm_train_and_test /= ts_len
    return idkfm_train_and_test



def KI_RplusTime_Multivariate(time_series_train,time_series_test,psi):

    X=time_series_train
    n,m=len(time_series_train),len(time_series_test)
    ts_len = len(time_series_train[0])
    train_num=ts_len*n
    dim=X.shape[2]
    if psi>=train_num:
        return None
    if m>0:
        X=np.concatenate((time_series_train,time_series_test))
    X=X.reshape(-1,dim)
    timeAxis=np.array(range(ts_len),dtype=int).reshape(1,-1)
    timeFeature = np.repeat(timeAxis, n + m, axis=0)
    timeFeature=timeFeature.reshape(-1)
    X=np.insert(X,dim,timeFeature,axis=1)
    X=scale(X,axis=0)
    idkfm_train_and_test = np.zeros((n + m, 100 * psi), dtype=float)
    onepointfm_matrix = IK_inne_fm_sparse(X, t=100, psi=psi, train_num=train_num).reshape((n + m, -1))
    counter_list = [Counter(it) for it in onepointfm_matrix]
    for counter in counter_list:
        if -1 in counter:
            del counter[-1]
    key_list = [list(counter.keys()) for counter in counter_list]
    val_list = [list(counter.values()) for counter in counter_list]
    for i in range(n + m):
        idkfm_train_and_test[i][key_list[i]] += val_list[i]
    idkfm_train_and_test /= ts_len
    return idkfm_train_and_test