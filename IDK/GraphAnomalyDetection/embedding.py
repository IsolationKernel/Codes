import numpy as np
from scipy.spatial.distance import cdist


def create_adj_avg(adj_cur):
    '''
    create adjacency
    '''
    deg = np.sum(adj_cur, axis=1)
    deg = np.asarray(deg).reshape(-1)

    deg[deg != 1] -= 1

    deg = 1/deg
    deg_mat = np.diag(deg)
    adj_cur = adj_cur.dot(deg_mat.T).T

    return adj_cur

# embedding of a graph


def createWlEmbedding(node_features, adj_mat, h):
    graph_feat = []
    for it in range(h+1):
        if it == 0:
            graph_feat.append(node_features)
        else:
            adj_cur = adj_mat+np.identity(adj_mat.shape[0])

            adj_cur = create_adj_avg(adj_cur)

            np.fill_diagonal(adj_cur, 0)
            graph_feat_cur = 0.5 * \
                (np.dot(adj_cur, graph_feat[it-1]) + graph_feat[it-1])
            graph_feat.append(graph_feat_cur)
    return np.mean(np.concatenate(graph_feat, axis=1), axis=0)
    # return np.mean(graph_feat_cur, axis=0)


# embedding of each node
def createWlEmbedding1(node_features, adj_mat, h):
    graph_feat = []
    for it in range(h+1):
        if it == 0:
            graph_feat.append(node_features)
        else:
            adj_cur = adj_mat+np.identity(adj_mat.shape[0])

            adj_cur = create_adj_avg(adj_cur)

            np.fill_diagonal(adj_cur, 0)
            graph_feat_cur = 0.5 * \
                (np.dot(adj_cur, graph_feat[it-1]) + graph_feat[it-1])
            graph_feat.append(graph_feat_cur)
    return np.concatenate(graph_feat, axis=1)
