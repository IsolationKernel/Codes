import copy
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler

#k is the number of anomalies
def get_acc(k, value_list, anomaly_cycles):
    true_pos = 0
    sorted_value_list = np.argsort(value_list)
    for it in sorted_value_list[0:k]:
        if it in anomaly_cycles:
            true_pos += 1
    count = k;
    for index in range(k, len(sorted_value_list)):
        if value_list[sorted_value_list[index]] != value_list[sorted_value_list[k-1]]:
            break
        count+=1
        if sorted_value_list[index] in anomaly_cycles:
            true_pos+=1
    return true_pos / count





def get_label(X, cycle, anomaly_list):
    window_num = (int)(X.shape[0] / cycle)
    label_list = np.zeros(window_num)
    label_list[anomaly_list] = 1
    return label_list

def subsequenceMinMaxScale(X, cycle):
    lo = 0
    hi = cycle
    scaler = preprocessing.MinMaxScaler()
    arr = copy.deepcopy(X)
    while lo < X.shape[0]:
        hi = min(hi, X.shape[0])
        arr[lo:hi] = scaler.fit_transform(X[lo:hi])
        lo = hi
        hi += cycle
    return arr

def prepocessSubsequence(X, cycle):
    lo = 0
    hi = cycle
    scaler = StandardScaler()
    arr=copy.deepcopy(X)
    while lo < X.shape[0]:
        hi = min(hi, X.shape[0])
        scaler = scaler.fit(X[lo:hi])
        arr[lo:hi] = scaler.transform(X[lo:hi])
        lo = hi
        hi += cycle
    return arr

def findthredsomeInSlidingWindowIDK(scorelist, redlist, width):
    mini = 1;
    for i in range(1, len(redlist)):
        if redlist[i][0] - redlist[i - 1][1] >= width:
            tem = scorelist[range(redlist[i - 1][1], redlist[i][0] - width + 1)]
            mini = min(mini, min(tem))
    return mini


def findthredsomeInCycleIDK(scorelist, anomaly_cycle):
    tem = np.delete(scorelist, anomaly_cycle, axis=0)
    return min(tem)

def getDiscordList(filename):
    df = pd.read_csv(filename, header=None)
    df = np.array(df).astype(int).reshape(len(df))
    return df

def drawIDK_T_discords(TS, cycle,idk_scores, number=3):
    sorted_index = np.argsort(idk_scores)
    color_list = ['r', 'y', 'b', 'g', 'c', 'm', 'k'] 
    cur = 0
    plt.plot(TS)
    for index in sorted_index:
        if number <= 0:
            break;
        number -= 1
        ls = range(index * cycle, (int)(index * cycle + cycle))
        ly = TS[ls]
        plt.plot(ls, ly, color=color_list[cur])
        cur = (cur + 1) % 7
    plt.show()

def ProduceMultiTS(str):
    ts_names=str.split('+')
    df=np.array(pd.read_csv("RealDatasets/MBA_ECG_"+ts_names[0]+".ts", header=None))[:(int)(1e5)].reshape(-1)
    pos_list = getDiscordList(
        "RealDatasets/ANNOTATIONS/MBA_"+ts_names[0]+"(1-100K).txt")
    dataTS=copy.deepcopy(df)
    annotation=copy.deepcopy(pos_list)
    for i,ts_name in enumerate(ts_names):
        if i==0:
            continue
        df = np.array(pd.read_csv("RealDatasets/MBA_ECG_" + ts_name + ".ts", header=None))[:(int)(1e5)].reshape(-1)
        pos_list = getDiscordList("RealDatasets/ANNOTATIONS/MBA_" + ts_name + "(1-100K).txt")+(int)(i*1e5)
        dataTS=np.concatenate((dataTS,df))
        annotation=np.concatenate((annotation,pos_list))
    return dataTS,annotation

def get_scoreCycleList_from_sliding(cycle, win_size, labels, score_list, zone=None):
    if zone is None:
        zone=win_size//2
    zone = min(zone, win_size - 1)
    result = np.zeros(len(labels))
    for i in range(len(result)):
        lo = max(0, i * cycle - zone)
        if lo >= len(score_list):
            print("have 0")
            break
        hi = min(len(score_list),
                 (i + 1) * cycle + zone - win_size + 1)
        assert lo < hi
        # if labels[i]==1:
        #     result[i]=np.max(score_list[lo:hi])
        # else:
        #     result[i]=np.min(score_list[lo:hi])
        result[i] = np.max(score_list[lo:hi])
    return result

def findthredsomeInSlidingWindowIDK(scorelist, redlist, width):
    maxi = -np.inf;index=-1
    for i in range(1, len(redlist)):
        if redlist[i][0] - redlist[i - 1][1] >= width:
            tem = scorelist[range(redlist[i - 1][1], min(redlist[i][0] - width + 1,len(scorelist)))]
            # if max(tem)>maxi:
            #     maxi=max(tem)
            #     index=i
            maxi = max(maxi, max(tem))
    return maxi

def drawDiscords(TS,score_list,width,number=3):
    color_list=['r','y','b', 'g',  'c', 'm',  'k']
    sorted_score_list = np.argsort(-score_list)
    cur=0
    top_k_idx =[]
    top_k_idx.append(sorted_score_list[0])
    for i in range(1,len(sorted_score_list)):
        if len(top_k_idx)>=number:
            break
        candidate=True
        for jt in top_k_idx:
            if(abs(jt-sorted_score_list[i])<width):
                candidate=False
                break
        if candidate:
            top_k_idx.append(sorted_score_list[i])


    plt.plot(TS,color='black')
    for index in top_k_idx:
        ls=range(index,(int)(index+width))
        ly=TS[ls]
        plt.plot(ls,ly,color='orange')
        cur=(cur+1)%7
    return top_k_idx

def step_subsequence(time_series, L, S=1):  # Window len = L, Stride len/stepsize = S
    S=max(1,S)
    time_series = np.asarray(time_series)
    nrows = ((time_series.size-L)//S)+1
    n = time_series.strides[0]
    return np.lib.stride_tricks.as_strided(time_series, shape=(nrows,L), strides=(S*n,n))

def plotTS(ts,cycle,anomaly_cycles):
    redlist = []
    redlist.append((0, 0))
    for it in anomaly_cycles:
        redlist.append((cycle * it, cycle * it + cycle))
    redlist.append((len(ts), len(ts)))
    print("anomaly_cycles num:", len(anomaly_cycles))
    plt.plot(ts, color='b')
    for it in redlist:
        ls = range(it[0], it[1])
        ly = ts[ls]
        plt.plot(ls, ly, color='r')

    plt.show()
    return redlist

def read_file(filename):
    file = pd.read_csv(filename, sep='\t', header=None)
    file = np.array(file)
 
    X = file[:, 1:]
    label = np.array(file[:, 0], dtype=int)
    # for i in range(len(X)):
    #     scaler = StandardScaler()
    #     X[i]=scaler.fit_transform(X[i].reshape(-1, 1)).reshape(-1)

    return X, label