from sklearn.preprocessing import scale
from sklearn.base import TransformerMixin

import argparse
import igraph as ig
import os
import copy
from collections import defaultdict
from typing import List

from utilities import load_continuous_graphs, create_labels_seq_cont, retrieve_graph_filenames
from read_dataset import load_continuous_graphs1
from IGK_kernel_S import *
import time

####################
# Embedding schemes
####################
def compute_wl_embeddings_continuous(data_directory, h):
    '''
    Continuous graph embeddings
    TODO: for package implement a class with same API as for WL
    '''
    node_features, adj_mat, n_nodes = load_continuous_graphs1(data_directory)

    node_features_data = scale(np.concatenate(node_features, axis=0), axis=0)
    splits_idx = np.cumsum(n_nodes).astype(int)
    node_features_split = np.vsplit(node_features_data, splits_idx)
    node_features = node_features_split[:-1]

    # Generate the label sequences for h iterations
    labels_sequence = create_labels_seq_cont(node_features, adj_mat, h)

    return labels_sequence


def compute_wl_embeddings_continuous_by_IK(data_directory, h,t,psi):
    # trainsform into isolation vector at first
    node_features, adj_mat, n_nodes = load_continuous_graphs1(data_directory)

    node_features_data = scale(np.concatenate(node_features, axis=0), axis=0)
    splits_idx = np.cumsum(n_nodes).astype(int)
    node_features_split = np.vsplit(node_features_data, splits_idx)
    node_features = node_features_split[:-1]

    # transform into Isolation Vector
    partitions = generate_partitions(node_features, t, psi)
    cell_indexs = convert_to_cell_index(partitions, node_features)
    igk_vectors = convert_index_to_vector(cell_indexs, psi)

    # Generate the label sequences for h iterations
    labels_sequence = create_labels_seq_cont(igk_vectors, adj_mat, h)
    return labels_sequence



####################
# Weisfeiler-Lehman
####################
class WeisfeilerLehman(TransformerMixin):
    """
    Class that implements the Weisfeiler-Lehman transform
    Credits: Christian Bock and Bastian Rieck
    """

    def __init__(self):
        self._relabel_steps = defaultdict(dict)
        self._label_dict = {}
        self._last_new_label = -1
        self._preprocess_relabel_dict = {}
        self._results = defaultdict(dict)
        self._label_dicts = {}

    def _reset_label_generation(self):
        self._last_new_label = -1

    def _get_next_label(self):
        self._last_new_label += 1
        return self._last_new_label

    def _relabel_graphs(self, X: List[ig.Graph]):
        num_unique_labels = 0
        preprocessed_graphs = []
        for i, g in enumerate(X):
            x = g.copy()

            if not 'label' in x.vs.attribute_names():
                x.vs['label'] = list(map(str, [l for l in x.vs.degree()]))
            labels = x.vs['label']

            new_labels = []
            for label in labels:
                if label in self._preprocess_relabel_dict.keys():
                    new_labels.append(self._preprocess_relabel_dict[label])
                else:
                    self._preprocess_relabel_dict[label] = self._get_next_label()
                    new_labels.append(self._preprocess_relabel_dict[label])
            x.vs['label'] = new_labels
            self._results[0][i] = (labels, new_labels)
            preprocessed_graphs.append(x)
        self._reset_label_generation()
        return preprocessed_graphs

    def fit_transform(self, X: List[ig.Graph], num_iterations: int = 3):
        X = self._relabel_graphs(X)
        for it in np.arange(1, num_iterations + 1, 1):
            self._reset_label_generation()
            self._label_dict = {}
            for i, g in enumerate(X):
                # Get labels of current interation
                current_labels = g.vs['label']

                # Get for each vertex the labels of its neighbors
                neighbor_labels = self._get_neighbor_labels(g, sort=True)

                # Prepend the vertex label to the list of labels of its neighbors
                merged_labels = [[b] + a for a, b in zip(neighbor_labels, current_labels)]

                # Generate a label dictionary based on the merged labels
                self._append_label_dict(merged_labels)

                # Relabel the graph
                new_labels = self._relabel_graph(g, merged_labels)
                self._relabel_steps[i][it] = {idx: {old_label: new_labels[idx]} for idx, old_label in
                                              enumerate(current_labels)}
                g.vs['label'] = new_labels

                self._results[it][i] = (merged_labels, new_labels)
            self._label_dicts[it] = copy.deepcopy(self._label_dict)
        return self._results

    def _relabel_graph(self, X: ig.Graph, merged_labels: list):
        new_labels = []
        for merged in merged_labels:
            new_labels.append(self._label_dict['-'.join(map(str, merged))])
        return new_labels

    def _append_label_dict(self, merged_labels: List[list]):
        for merged_label in merged_labels:
            dict_key = '-'.join(map(str, merged_label))
            if dict_key not in self._label_dict.keys():
                self._label_dict[dict_key] = self._get_next_label()

    def _get_neighbor_labels(self, X: ig.Graph, sort: bool = True):
        neighbor_indices = [[n_v.index for n_v in X.vs[X.neighbors(v.index)]] for v in X.vs]
        neighbor_labels = []
        for n_indices in neighbor_indices:
            if sort:
                neighbor_labels.append(sorted(X.vs[n_indices]['label']))
            else:
                neighbor_labels.append(X.vs[n_indices]['label'])
        return neighbor_labels
