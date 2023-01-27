import random
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, scale

#from artificialConstruction import construct_sbd_matrix, clustering_mst


def step_subsequence(time_series, L, S=1):  # Window len = L, Stride len/stepsize = S
    S = max(1, S)
    time_series = np.asarray(time_series)
    nrows = ((time_series.size - L) // S) + 1
    n = time_series.strides[0]
    return np.lib.stride_tricks.as_strided(time_series, shape=(nrows, L), strides=(S * n, n))


def read_file(filename):
    file = pd.read_csv(filename, sep='\t', header=None)
    file = np.array(file)
    # 取出第一列
    X = file[:, 1:]
    label = np.array(file[:, 0], dtype=int)
    # for i in range(len(X)):
    #     scaler = StandardScaler()
    #     X[i]=scaler.fit_transform(X[i].reshape(-1, 1)).reshape(-1)

    return X, label


def read_dataDirectory(directory):
    str = directory.split('/')[1]
    train_x, train_y = read_file(directory + '/' + str + '_TRAIN.tsv')
    test_x, test_y = read_file(directory + '/' + str + '_TEST.tsv')
    return train_x, train_y, test_x, test_y


def datasetProcess(InputName, alpha=0.02, seednum=10):
    train_X, train_y, test_X, test_y = read_dataDirectory('UCR_data/' + InputName)
    all_X = np.concatenate((train_X, test_X))
    all_y = np.concatenate((train_y, test_y))
    all_X = scale(all_X, axis=1)
    bin = np.bincount(all_y)
    majority_label = np.argmax(bin)
    normal_num = max(bin)
    fac = (1 - alpha) / alpha
    anomaly_num = max(int(normal_num / fac), 1)
    index = (all_y == majority_label)
    sample_y = all_y[~index]
    sample_x = all_X[~index]
    normal_X = all_X[index]
    normal_y = [0] * len(normal_X)
    sample_list = [p for p in range(len(sample_x))]
    sample_list = random.sample(sample_list, anomaly_num)
    anomaly_x = sample_x[sample_list, :]
    anomaly_y = [1] * len(anomaly_x)
    X = np.concatenate((normal_X, anomaly_x))
    y = np.concatenate((normal_y, anomaly_y))
    all_data = np.insert(X, 0, y, axis=1)
    df = pd.DataFrame(all_data, columns=None)
    df.to_csv(f"UCR_Ano/{InputName}_{alpha}_{seednum}.tsv", sep='\t', index=None, header=None)
    return X, y


def datasetProcessHard(InputName, alpha=0.02, seednum=10):
    train_X, train_y, test_X, test_y = read_dataDirectory('UCR_data/' + InputName)
    all_X = np.concatenate((train_X, test_X))
    all_y = np.concatenate((train_y, test_y))
    all_X = scale(all_X, axis=1)
    bin = np.bincount(all_y)
    majority_label = np.argmax(bin)
    normal_num = max(bin)
    index = (all_y == majority_label)
    sample_y = all_y[~index]
    sample_x = all_X[~index]
    normal_X = all_X[index]
    normal_y = [0] * len(normal_X)
    anomaly_num = max(int(alpha * len(sample_y)), 1)
    sample_list = [p for p in range(len(sample_x))]
    sample_list = random.sample(sample_list, anomaly_num)
    anomaly_x = sample_x[sample_list, :]
    anomaly_y = [1] * len(anomaly_x)
    X = np.concatenate((normal_X, anomaly_x))
    y = np.concatenate((normal_y, anomaly_y))
    all_data = np.insert(X, 0, y, axis=1)
    df = pd.DataFrame(all_data, columns=None)
    df.to_csv(f"UCR_AnoHard/{InputName}_{alpha}_{seednum}.tsv", sep='\t', index=None, header=None)
    return X, y


def datasetProcessMultiNormal(InputName, alpha=0.02, seednum=10):

    train_X, train_y, test_X, test_y = read_dataDirectory('UCR_data/' + InputName)
    all_X = np.concatenate((train_X, test_X))
    all_y = np.concatenate((train_y, test_y))
    all_X = scale(all_X, axis=1)
    counter = Counter(all_y)
    counter_tuple = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
    # bin=np.bincount(all_y)
    # rank_label=np.argsort(-bin)
    rank_label = [ele[0] for ele in counter_tuple]
    # normalClassNum=int(np.ceil(len(np.unique(all_y))/2))
    normalClassNum = len(np.unique(all_y)) // 2
    majority_labels = rank_label[:normalClassNum]
    index = np.full(len(all_y), False, bool)
    index[[i for i in range(len(all_y)) if all_y[i] in majority_labels]] = True
    sample_y = all_y[~index]
    sample_x = all_X[~index]
    normal_X = all_X[index]
    normal_y = [0] * len(normal_X)
    anomaly_num = max(int(alpha * len(sample_y)), 1)
    sample_list = [p for p in range(len(sample_x))]
    sample_list = random.sample(sample_list, anomaly_num)
    anomaly_x = sample_x[sample_list, :]
    anomaly_y = [1] * len(anomaly_x)
    X = np.concatenate((normal_X, anomaly_x))
    y = np.concatenate((normal_y, anomaly_y))
    all_data = np.insert(X, 0, y, axis=1)
    df = pd.DataFrame(all_data, columns=None)
    #df.to_csv(f"UCR_AnoMulti/{InputName}_{alpha}_{seednum}.tsv", sep='\t', index=None, header=None)
    return X, y


def datasetProcessMultiNormalOneAno(InputName, alpha=0.02, seednum=10):
    train_X, train_y, test_X, test_y = read_dataDirectory('UCR_data/' + InputName)
    all_X = np.concatenate((train_X, test_X))
    all_y = np.concatenate((train_y, test_y))
    all_X = scale(all_X, axis=1)
    counter = Counter(all_y)
    counter_tuple = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
    # bin=np.bincount(all_y)
    # rank_label=np.argsort(-bin)
    rank_label = [ele[0] for ele in counter_tuple]
    # normalClassNum=int(np.ceil(len(np.unique(all_y))/2))
    normalClassNum = len(np.unique(all_y)) // 2
    majority_labels = rank_label[:normalClassNum]
    ano_label = rank_label[normalClassNum]

    index = np.full(len(all_y), False, bool)
    index[[i for i in range(len(all_y)) if all_y[i] in majority_labels]] = True

    #sample_y = all_y[~index]
    #sample_x = all_X[~index]
    sample_y = all_y[[i for i in range(len(all_y)) if all_y[i]==ano_label]]
    sample_x = all_X[[i for i in range(len(all_y)) if all_y[i]==ano_label],:]
    normal_X = all_X[index]
    normal_y = [0] * len(normal_X)
    anomaly_num = max(int(alpha * len(sample_y)), 1)
    sample_list = [p for p in range(len(sample_x))]
    sample_list = random.sample(sample_list, anomaly_num)
    anomaly_x = sample_x[sample_list, :]
    anomaly_y = [1] * len(anomaly_x)
    X = np.concatenate((normal_X, anomaly_x))
    y = np.concatenate((normal_y, anomaly_y))
    all_data = np.insert(X, 0, y, axis=1)
    df = pd.DataFrame(all_data, columns=None)
    df.to_csv(f"UCR_AnoClustered/{InputName}_{alpha}_{seednum}.tsv", sep='\t', index=None, header=None)
    return X, y


def datasetProcessGivenNormalClass(InputName, normalClass, alpha=0.02, seednum=10):
    train_X, train_y, test_X, test_y = read_dataDirectory('UCR_data/' + InputName)
    all_X = np.concatenate((train_X, test_X))
    all_y = np.concatenate((train_y, test_y))
    majority_label = normalClass
    normal_num = sum(all_y == normalClass)
    fac = (1 - alpha) / alpha
    anomaly_num = max(int(normal_num / fac), 1)
    index = (all_y == majority_label)
    sample_y = all_y[~index]
    sample_x = all_X[~index]
    normal_X = all_X[index]
    normal_y = [0] * len(normal_X)
    sample_list = [p for p in range(len(sample_x))]
    sample_list = random.sample(sample_list, anomaly_num)
    anomaly_x = sample_x[sample_list, :]
    anomaly_y = [1] * len(anomaly_x)
    X = np.concatenate((normal_X, anomaly_x))
    y = np.concatenate((normal_y, anomaly_y))
    all_data = np.insert(X, 0, y, axis=1)
    df = pd.DataFrame(all_data, columns=None)
    df.to_csv(f"UCR_MixClass/{InputName}_{alpha}_{seednum}.tsv", sep='\t', index=None, header=None)
    return X, y
def datasetProcessMultiNormalGivenNormal(InputName, normalClasses,alpha=0.02, seednum=10):
    train_X, train_y, test_X, test_y = read_dataDirectory('UCR_data/' + InputName)
    all_X = np.concatenate((train_X, test_X))
    all_y = np.concatenate((train_y, test_y))
    all_X = scale(all_X, axis=1)
    counter = Counter(all_y)
    counter_tuple = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
    # bin=np.bincount(all_y)
    # rank_label=np.argsort(-bin)
    rank_label = [ele[0] for ele in counter_tuple]
    # normalClassNum=int(np.ceil(len(np.unique(all_y))/2))
    normalClassNum = len(np.unique(all_y)) // 2
    majority_labels = rank_label[:normalClassNum]
    majority_labels=normalClasses
    index = np.full(len(all_y), False, bool)
    index[[i for i in range(len(all_y)) if all_y[i] in majority_labels]] = True
    sample_y = all_y[~index]
    sample_x = all_X[~index]
    normal_X = all_X[index]
    normal_y = [0] * len(normal_X)
    anomaly_num = max(int(alpha * len(sample_y)), 1)
    sample_list = [p for p in range(len(sample_x))]
    sample_list = random.sample(sample_list, anomaly_num)
    anomaly_x = sample_x[sample_list, :]
    anomaly_y = [1] * len(anomaly_x)
    X = np.concatenate((normal_X, anomaly_x))
    y = np.concatenate((normal_y, anomaly_y))
    all_data = np.insert(X, 0, y, axis=1)
    df = pd.DataFrame(all_data, columns=None)
    df.to_csv(f"UCR_AnoMixLess/{InputName}_{alpha}_{seednum}.tsv", sep='\t', index=None, header=None)
    return X, y

def datasetProcessMixClass(InputName, alpha=0.02, seednum=10):
    train_X, train_y, test_X, test_y = read_dataDirectory('UCR_data/' + InputName)
    all_X = np.concatenate((train_X, test_X))
    all_y = np.concatenate((train_y, test_y))
    all_y[all_y==-1]+=3
    if 0 in all_y:
        all_y+=1
    adjacencyMatrix = construct_sbd_matrix(all_X, all_y)
    clusterIndex = clustering_mst(adjacencyMatrix)
    bin = np.bincount(clusterIndex)
    more = np.argmin(bin)  #gai
    majority_classes=np.argwhere(clusterIndex==more).reshape(-1)+1
    datasetProcessMultiNormalGivenNormal(InputName,majority_classes,alpha,seednum)


def load_UEA(dataset):
    train_data = arff.loadarff(open(f'datasets/UEA/{dataset}/{dataset}_TRAIN.arff',encoding='utf-8'))[0]
    test_data = arff.loadarff(open(f'datasets/UEA/{dataset}/{dataset}_TEST.arff',encoding='utf-8'))[0]

    def extract_data(data):
        res_data = []
        res_labels = []
        for t_data, t_label in data:
            t_data = np.array([d.tolist() for d in t_data])
            t_label = t_label.decode("utf-8")
            res_data.append(t_data)
            res_labels.append(t_label)
        return np.array(res_data).swapaxes(1, 2), np.array(res_labels)

    train_X, train_y = extract_data(train_data)
    test_X, test_y = extract_data(test_data)

    scaler = StandardScaler()
    scaler.fit(train_X.reshape(-1, train_X.shape[-1]))
    train_X = scaler.transform(train_X.reshape(-1, train_X.shape[-1])).reshape(train_X.shape)
    test_X = scaler.transform(test_X.reshape(-1, test_X.shape[-1])).reshape(test_X.shape)

    labels = np.unique(train_y)
    transform = {k: i for i, k in enumerate(labels)}
    train_y = np.vectorize(transform.get)(train_y)
    test_y = np.vectorize(transform.get)(test_y)
    return train_X, train_y, test_X, test_y



from scipy.io import arff


def read_arff(filename):
    train_X_list = []
    train_y_list = []
    test_x_list = []
    test_y_list = []
    for num in range(30):
        data_train, _ = arff.loadarff(f'{filename}{num}_TRAIN.arff')
        data_test, _ = arff.loadarff(f'{filename}{num}_TEST.arff')
        data_train = np.array(data_train.tolist(), dtype=float)
        data_test = np.array(data_test.tolist(), dtype=float)
        train_y = data_train[:, -1].astype(int)
        train_X = data_train[:, :-1]
        test_y = data_test[:, -1].astype(int)
        test_X = data_test[:, :-1]
        train_X_list.append(train_X)
        train_y_list.append(train_y)
        test_x_list.append(test_X)
        test_y_list.append(test_y)
    return train_X_list, train_y_list, test_x_list, test_y_list

