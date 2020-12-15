from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import numpy as np

def generate_partitions(X,t,psi):
    # train iForest models
    X = np.concatenate(X, axis = 0)
    knn_models = []
    for i in range(t):
        sample_index = np.random.permutation(len(X))
        sample_index = sample_index[0:psi]
        Y = [i for i in range(psi)]
        sample_X = X[sample_index, :]
        neigh = DecisionTreeClassifier(splitter='random')
        neigh.fit(sample_X, Y)
        knn_models.append(neigh)
    return knn_models

def generate_partitions1(X,t,psi):
    # train KNN models
    print("kNN")
    X = np.concatenate(X, axis = 0)
    knn_models = []
    for i in range(t):
        sample_index = np.random.permutation(len(X))
        sample_index = sample_index[0:psi]
        Y = [i for i in range(psi)]
        sample_X = X[sample_index, :]
        neigh = KNeighborsClassifier()
        neigh.fit(sample_X, Y)
        knn_models.append(neigh)
    return knn_models

def convert_to_cell_index(partitions, X):
    cell_indexs = []
    t = len(partitions)
    num_graph = len(X)
    X_concate = np.concatenate(X, axis=0)

    graph_cell_indexs = []
    for i in range(t):
        knn_model = partitions[i]
        cell_index = knn_model.predict(X_concate)
        graph_cell_indexs.append(cell_index)
    graph_cell_indexs = np.array(graph_cell_indexs)

    flag = 0
    for i in range(num_graph):
        point_num = X[i].shape[0]
        cell_indexs.append(graph_cell_indexs[:,flag:flag + point_num])
        flag += point_num
    return cell_indexs


def convert_index_to_vector(cell_indexs, psi):
    igk_vectors = []
    [t, _] = cell_indexs[0].shape
    for cell_index in cell_indexs:
        num_point = cell_index.shape[1]
        graph_vectors = np.zeros((num_point, psi * t))
        for i in range(num_point):
            instance_vector = np.zeros((1, psi * t))
            instance_index = cell_index[:,i].flatten()
            bias = [t * psi for t in range(t)]
            instance_index = instance_index + bias
            instance_vector[0, instance_index.flatten().tolist()] += 1
            graph_vectors[i,:] = instance_vector
        igk_vectors.append(graph_vectors)
    return igk_vectors


def compute_graph_node_weight(graph, t,epsilon = 0.85):
    link_matrix = np.matmul(graph, graph.T)/t
    link_matrix[link_matrix>epsilon] = 1
    link_matrix[link_matrix <= epsilon] = 0
    weight = 1/np.sum(link_matrix,axis=0)
    weight = weight/np.sum(weight)
    return weight


def average_graph(X,t):
    graph_vectors = []
    for graph in X:
        # weight = compute_graph_node_weight(graph,t)
        # weight = np.matrix(weight)
        # weight = weight.reshape((weight.shape[1],1))
        # graph = np.multiply(weight, graph)
        graph_vector = np.average(graph,axis=0)
        graph_vector = np.array(graph_vector)
        graph_vectors.append(graph_vector.flatten())
    return graph_vectors


def precompute_normalize_factor(igk_vectors1,igk_vectors2):
    normalize_factors1 = []
    normalize_factors2 = []
    for igk_vector in igk_vectors1:
        normalize_factors1.append(np.sqrt(np.matmul(igk_vector, igk_vector.T)))
    for igk_vector in igk_vectors2:
        normalize_factors2.append(np.sqrt(np.matmul(igk_vector, igk_vector.T)))
    return normalize_factors1,normalize_factors2


def compute_igk_similarity(igk_vector1, igk_vector2,normalize_factor1,normalize_factor2):
    # func2
    similarity = np.matmul(igk_vector1, igk_vector2.T)
    # normalize
    similarity = similarity / normalize_factor1 / normalize_factor2

    return similarity


def compute_IGK_kernel(X1, X2, psi):
    t = X1.shape[1]/psi
    igk_vectors1 = average_graph(X1,t)
    igk_vectors2 = average_graph(X2,t)

    kernel_matrix = np.zeros((X1.shape[0], X2.shape[0]))
    num1 = X1.shape[0]
    num2 = X2.shape[0]
    normalize_factors1,normalize_factors2  = precompute_normalize_factor(igk_vectors1,igk_vectors2)
    for i in range(num1):
        for j in range(num2):
            kernel_matrix[i,j] = compute_igk_similarity(igk_vectors1[i],igk_vectors2[j],normalize_factors1[i],normalize_factors2[j])
    return kernel_matrix
