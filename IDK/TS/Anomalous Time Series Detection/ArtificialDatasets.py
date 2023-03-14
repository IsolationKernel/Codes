import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import roc_auc_score, euclidean_distances
from sklearn.neighbors import LocalOutlierFactor
from tslearn.metrics import cdist_dtw
from minirocket import fit,transform
from KI_RplusTime import KI_RplusTime
from utilities import read_file

def create_noisysine():
    time1 = np.linspace(0, 2 * np.pi, 100)
    sinwave = np.sin(time1)
    normal_list=[]
    rep = 1
    sinwave = np.repeat(sinwave.reshape(1,-1), rep,axis=0).reshape(-1)
    np.random.seed(5)
    for _ in range(496):
        normal_list.append(sinwave+np.random.normal(scale=0.8,size=sinwave.shape))
    X=normal_list
    ano1=sinwave
    ano2=sinwave + np.random.normal(scale=0.2, size=sinwave.shape)
    ano3=sinwave +np.random.normal(scale=0.5, size=sinwave.shape)
    ano4 = sinwave + np.random.normal(scale=0.7, size=sinwave.shape)
    ano5=sinwave +np.random.normal(scale=0.9, size=sinwave.shape)
    X=np.append(X,[ano1,ano2,ano3,ano4],axis=0)
    #X=scale(X,axis=1)
    # for i in range(1,6):
    #     plt.plot(X[-i])
    #     plt.show()
    y=np.zeros(len(X),dtype=int)
    y[-4:]=1
    np.random.seed()
    return X,y

if __name__ == '__main__':

    X, y = read_file('Artificial_datasets/noisysine.tsv')
    figure, axes = plt.subplots(1, 1)
    axes.plot(X[2], color='#48c072')
    axes.plot(X[-4]-1, color='red')
    axes.plot(X[-3]-2, color='red')
    axes.get_yaxis().set_visible(False)
    axes.get_xaxis().set_visible(False)
    plt.show()
    neiborNum_list = [1, 3, 5, 7, 11, 21, 51, 101, 201, 501, 1001, 2001]
    maxNeiborNum = min(max(neiborNum_list) + 1, len(X))
    result_list=[]
    for nei in neiborNum_list:
        if nei>=maxNeiborNum:
            break
        clf = LocalOutlierFactor(n_neighbors=nei)
        clf.fit(X)
        normal_scores = clf.negative_outlier_factor_
        result = roc_auc_score(y, -normal_scores)
        result_list.append(result)
        print(nei,result)
    print('LOF(ED) result',max(result_list))
    norm2norm1=euclidean_distances([X[0],X[1]])
    norm2norm2 = euclidean_distances([X[0], X[2]])
    norm2ano1=euclidean_distances([X[0],X[-1]])
    norm2ano2 = euclidean_distances([X[0], X[-2]])
    norm2ano2 = euclidean_distances([X[0], X[-3]])
    print(norm2norm1,'\n',norm2norm2,'\n',norm2ano1,'\n',norm2ano2)
    ed_matrix=euclidean_distances(X)
    ts_len=X.shape[1]
    #df.to_csv(f"Artificial_datasets/noisysine_final.tsv", sep='\t', index=None, header=None)
    dtw5_matrix=cdist_dtw(X, global_constraint="sakoe_chiba", sakoe_chiba_radius=max(1, int(0.05 * ts_len)),n_jobs=-1)
    result_list = []
    for nei in neiborNum_list:
        if nei>=maxNeiborNum:
            break
        clf = LocalOutlierFactor(n_neighbors=nei,metric='precomputed')
        clf.fit(dtw5_matrix)
        normal_scores = clf.negative_outlier_factor_
        result = roc_auc_score(y, -normal_scores)
        result_list.append(result)
        print(nei,result)
    print('LOF(DTW) result',max(result_list))
    X_training = np.array(X, dtype=np.float32)
    parameters = fit(X_training)
    X_training_transform = transform(X_training, parameters)
    rocket_matrix=euclidean_distances(X_training_transform)
    result_list = []
    for nei in neiborNum_list:
        if nei >= maxNeiborNum:
            break
        clf = LocalOutlierFactor(n_neighbors=nei)
        clf.fit(X_training_transform)
        normal_scores = clf.negative_outlier_factor_
        result = roc_auc_score(y, -normal_scores)
        result_list.append(result)
        print(nei, result)
    print('LOF(Rocket) result', max(result_list))


    #IDK
    psi_list=[2,4,8,16,32,64,128,256,512,1024,2048]
    for psi in psi_list:
        idk_fm=KI_RplusTime(X, [], psi=psi)
        result_list = []
        for nei in neiborNum_list:
            if nei >= maxNeiborNum:
                break
            clf = LocalOutlierFactor(n_neighbors=nei)
            clf.fit(idk_fm)
            normal_scores = clf.negative_outlier_factor_
            result = roc_auc_score(y, -normal_scores)
            result_list.append(result)
            print(nei, result)
        print('LOF(IDK) result', psi,max(result_list))
        idk_matrix = euclidean_distances(idk_fm)







