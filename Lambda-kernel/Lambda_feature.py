import numpy as np
from Lambda_map import *
from joblib import Parallel, delayed
import warnings

warnings.filterwarnings('ignore')

def normalization_partition(data,eta):
    a = data.T
    psi = a.shape[0]
    a = np.expand_dims(a,0).repeat(psi,axis=0)
    tmp_2 = data.T.reshape((psi,1,data.shape[0]))
    M = a-tmp_2
    m = 1/np.sqrt(np.sum(np.exp(-2*eta*M),axis=0))
    m = m.T
    return m

def normalization_vect(vect,eta,psi,t):
    assert vect.shape[1] == (psi*t)
    data_lst =  Parallel(n_jobs=-1)(delayed(normalization_partition)(vect[:,int(i*psi):int((i+1)*psi)],eta) for i in range(t))
    feature_map = np.concatenate(data_lst,axis=1)
    assert feature_map.shape==vect.shape,print(feature_map.shape,vect.shape)
    assert np.all(feature_map<=1)
    assert np.all(feature_map>=0)
    return feature_map/np.sqrt(t)

def lambda_feature_infty(distribution,newdata,psi,t=100):
    # produce feature and distance matrix for X and query_points
    lm = Lambda_map(psi,t)
    lm.fit(distribution)
    dis_map = lm.transform(distribution).toarray()
    new_map = lm.transform(newdata).toarray()
    dis_map = dis_map/np.sqrt(t)
    new_map = new_map/np.sqrt(t)
    dm = 1-np.dot(dis_map,dis_map.T)
    dm[np.where(dm<0)]=0
    dm2 = 1-np.dot(new_map,dis_map.T)
    dm2[np.where(dm2<0)]=0
    return dis_map,new_map,dm,dm2

def lambda_feature_continous(distribution,newdata,eta,psi,t=100):
    # produce feature and distance matrix for X and query_points
    lm = Lambda_map(psi,t)
    lm.fit(distribution)
    dis_map = lm.transform_continous(distribution).toarray()
    dis_map = normalization_vect(dis_map,eta,psi,t)
    new_map = lm.transform_continous(newdata).toarray()
    new_map = normalization_vect(new_map,eta,psi,t)
    dm = 1-np.dot(dis_map,dis_map.T)
    dm[np.where(dm<0)]=0
    dm2 = 1-np.dot(new_map,dis_map.T)
    dm2[np.where(dm2<0)]=0
    return dis_map,new_map,dm,dm2

