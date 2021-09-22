import numpy as np
import os
import igraph as ig
import pandas as pd
from utilities import *

def load_continuous_graphs1(filenames):
    dataset_name = os.path.basename(filenames)
    if dataset_name=="ENZYMES":
        return load_continuous_graphs(filenames)
    edge_filename = os.path.join(filenames, dataset_name+'_A.txt')
    graph_indicator_filename = os.path.join(filenames, dataset_name+'_graph_indicator.txt')
    node_attr_filename = os.path.join(filenames, dataset_name+'_node_attributes.txt')

    # initialize
    node_features = []
    adj_mat = []
    n_nodes = []

    edges = pd.read_csv(edge_filename,header=None,index_col=None).values
    indica = pd.read_csv(graph_indicator_filename,header=None,index_col=None).values
    node_attr = pd.read_csv(node_attr_filename,header=None,index_col=None).values
    num_graph = np.max(indica)
    node_flag = 0

    # calculate number of nodes
    ind_flag = 1
    n_nodes_list = []
    now_n_nodes = 0
    for ind in indica:
        if ind == ind_flag:
            now_n_nodes += 1
        else:
            n_nodes_list.append(now_n_nodes)
            now_n_nodes = 1
            ind_flag += 1
    n_nodes_list.append(now_n_nodes)

    ind_flag = 0
    edge_flag = 0
    for i in range(num_graph):
        now_n_nodes = n_nodes_list[i]
        n_nodes.append(now_n_nodes)
        now_adj_mat = np.zeros((now_n_nodes,now_n_nodes))
        now_attr = node_attr[ind_flag:ind_flag+now_n_nodes,:]
        node_features.append(now_attr)
        ind_flag += now_n_nodes
        while True:
            if edge_flag>=len(edges):
                break
            if edges[edge_flag, 0]>=node_flag+1 and edges[edge_flag, 0]<=node_flag+now_n_nodes:
                now_adj_mat[edges[edge_flag, 0] - 1- node_flag, edges[edge_flag, 1]-1- node_flag] += 1
                edge_flag += 1
            else:
                break
        node_flag += now_n_nodes
        adj_mat.append(now_adj_mat)

    n_nodes = np.asarray(n_nodes)
    node_features = np.asarray(node_features)
    print("finish load graph")
    return node_features, adj_mat, n_nodes

