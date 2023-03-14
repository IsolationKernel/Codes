# Copyright 2022 Xin Han
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np


def dendrogram_purity(dendrogram: np.ndarray, y: np.array):
    """
    A :math:`(n-1)` by 4 matrix ``Z`` is returned. At the
    :math:`i`-th iteration, clusters with indices ``Z[i, 0]`` and
    ``Z[i, 1]`` are combined to form cluster :math:`n + i`. A
    cluster with an index less than :math:`n` corresponds to one of
    the :math:`n` original observations. The distance between
    clusters ``Z[i, 0]`` and ``Z[i, 1]`` is given by ``Z[i, 2]``. The
    fourth value ``Z[i, 3]`` represents the number of original
    observations in the newly formed cluster.
    """
    n_instance = len(y)
    parent_matrix = _get_parent(dendrogram=dendrogram, n_instance=n_instance)
    node_purity = _get_node_purity(parent_matrix=parent_matrix, y=y)
    y_label = np.unique(y)
    purity = 0
    s = 0
    for ci in range(len(y_label)):
        current_instances = np.argwhere(y == y_label[ci]).flatten()
        for i in range(len(current_instances)):
            for j in range(len(current_instances))[i + 1:]:
                purity += _purity_score(current_instances[i], current_instances[j], ci,
                                        parent_matrix, node_purity, n_instance)
                s += 1
    purity = purity / s
    return purity


def _purity_score(i: int, j: int, ci: int, parent_matrix: np.ndarray, node_purity: np.ndarray, n_instances: int):
    if i == j:
        score = node_purity[ci, i]
    else:
        lca = np.argwhere(parent_matrix[i, :] *
                          parent_matrix[j, :] == 1).flatten()[0]
        score = node_purity[ci, lca + n_instances]
    return score


def _get_parent(dendrogram: np.ndarray, n_instance: int):
    """
    A :math:`(n-1)` by 4 matrix ``Z`` is returned. At the
    :math:`i`-th iteration, clusters with indices ``Z[i, 0]`` and
    ``Z[i, 1]`` are combined to form cluster :math:`n + i`. A
    cluster with an index less than :math:`n` corresponds to one of
    the :math:`n` original observations. The distance between
    clusters ``Z[i, 0]`` and ``Z[i, 1]`` is given by ``Z[i, 2]``. The
    fourth value ``Z[i, 3]`` represents the number of original
    observations in the newly formed cluster.
    """
    parent = [[i + n_instance] for i in range(n_instance - 1)]
    dendrogram = np.append(dendrogram, parent, axis=1)
    parent_matrix = np.zeros(
        shape=[n_instance, 2 * n_instance - 1], dtype=np.int8)

    for i in range(n_instance - 1):
        current_ind = []
        if dendrogram[i, 0] >= n_instance:
            ind = np.argwhere(
                parent_matrix[:, int(dendrogram[i, 0])] == 1).flatten()
            for item in ind:
                current_ind.append(item)
        else:
            current_ind.append(int(dendrogram[i, 0]))
        if dendrogram[i, 1] >= n_instance:
            ind = np.argwhere(
                parent_matrix[:, int(dendrogram[i, 1])] == 1).flatten()
            for item in ind:
                current_ind.append(item)
        else:
            current_ind.append(int(dendrogram[i, 1]))
        parent_matrix[current_ind, int(dendrogram[i, 4])] = 1
    parent_matrix = np.delete(parent_matrix, np.s_[:n_instance], axis=1)
    return parent_matrix


def _get_node_purity(parent_matrix: np.ndarray, y: np.array):
    n_instance = len(y)
    y_label = np.unique(y)
    node_purity = np.zeros(shape=(len(y_label), 2 * n_instance - 1))
    subtree_sum = np.ones(n_instance, dtype=np.int64)

    for ci in range(len(y_label)):
        ind = np.argwhere(y == y_label[ci]).flatten()
        node_purity[ci, ind] = 1
        for ti in range(2 * n_instance - 1)[n_instance:]:
            Tinstances = np.argwhere(
                parent_matrix[:, ti - n_instance] == 1).flatten()
            p = 0
            for item in Tinstances:
                p = p + node_purity[ci, item] * subtree_sum[item]
            node_purity[ci, ti] = p / sum(subtree_sum[Tinstances])
    return node_purity
