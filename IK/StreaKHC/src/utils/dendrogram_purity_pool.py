"""
Copyright (C) 2021 Xin Han.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import time
from functools import partial
from itertools import combinations, groupby
from multiprocessing import Pool, Manager

import threading
from queue import Queue

import numpy as np
import sys
sys.setrecursionlimit(50000)
queueLock = threading.Lock()


class Producer(threading.Thread):
    def __init__(self, samp_queue, cluser_to_leaves, leaf_to_cluster, result_queue, non_singleton_leaves, *args, **kwargs):
        super(Producer, self).__init__(*args, **kwargs)
        self.samp_queue = samp_queue
        self.result_queue = result_queue
        self.non_singleton_leaves = non_singleton_leaves
        self.cluster_to_leaves = cluser_to_leaves
        self.leaf_to_cluster = leaf_to_cluster

    def run(self):
        while True:
            queueLock.acquire()
            if self.samp_queue.empty():
                print('bye')
                queueLock.release()
                break
            print('剩余采样数：', self.samp_queue.qsize())
            samp = self.samp_queue.get()
            queueLock.release()
            self.get_purity()

    def get_purity(self):
        rand_leaf = np.random.choice(self.non_singleton_leaves)
        cluster = self.leaf_to_cluster[rand_leaf]
        rand_cluster_member = np.random.choice(self.cluster_to_leaves[cluster])
        # Make sure we get two distinct leaves
        while rand_cluster_member == rand_leaf:
            #assert(leaf_to_cluster[rand_leaf] == leaf_to_cluster[rand_cluster_member])
            rand_cluster_member = np.random.choice(
                self.cluster_to_leaves[cluster])
        lca = rand_leaf.lca(rand_cluster_member)
        purity = lca.purity(cluster=cluster)
        # print(purity)
        queueLock.acquire()
        self.result_queue.put(purity)
        queueLock.release()
        # print(self.result_queue.qsize())


def expected_dendrogram_purity(root):
    """Compute the expected dendrogram purity.

    Sample a leaf uniformly at random. Then sample another leaf from the same
    true class uniformly at random. Find their lowest common ancestor in the
    tree and compute purity with respect to that class. (This is one of the
    evaluations used in the Bayesian Hierarchical Clustering paper).

    Args:
      root - the root with respect to which we compute purity.

    Returns:
      A float [0, 1] that represents expected dendrogram purity.
    """

    # Construct a map from leaf to cluster and from cluster to a list of leaves.
    # Filter out the singletons in the leaf to cluster map.
    leaves = root.leaves()

    def get_cluster(x):
        return x.pts[0][0]

    cluster_to_leaves = {c: list(ls)
                         for c, ls in groupby(sorted(leaves, key=get_cluster),
                                              get_cluster)}
    leaf_to_cluster = {l: l.pts[0][0] for l in leaves}
    non_singleton_leaves = [l for l in leaf_to_cluster.keys()
                            if len(cluster_to_leaves[leaf_to_cluster[l]]) > 1]
    if len(non_singleton_leaves) == 0.0:
        return 1.0

    # For n samples, sample a leaf uniformly at random then select another leaf
    # from the same class unformly at random.
    samps = len(non_singleton_leaves) * 5  # TODO (AK): pick 5 in a better way.
    with Pool(processes=6) as pool:
        res = pool.starmap(
            process, [(non_singleton_leaves, leaf_to_cluster, cluster_to_leaves)]*samps)
    return sum(res) / samps


def process(non_singleton_leaves, leaf_to_cluster, cluster_to_leaves):

    rand_leaf = np.random.choice(non_singleton_leaves)
    cluster = leaf_to_cluster[rand_leaf]
    rand_cluster_member = np.random.choice(cluster_to_leaves[cluster])
    # Make sure we get two distinct leaves
    while rand_cluster_member == rand_leaf:
        rand_cluster_member = np.random.choice(cluster_to_leaves[cluster])
    lca = rand_leaf.lca(rand_cluster_member)
    purity = lca.purity(cluster=cluster)
    return purity


def dendrogram_purity(root):
    """
    Exact dendrogram purity
    """
    leaves = root.leaves()

    def get_cluster(x):
        return x.pts[0][0]

    sorted_lvs = sorted(leaves, key=get_cluster)
    leaves_by_true_class = {c: list(ls) for c, ls in groupby(sorted_lvs,
                                                             key=get_cluster)}
    leaf_pairs_by_true_class = {}
    for class_lbl, lvs in leaves_by_true_class.items():
        # leaf_pairs_by_true_class[class_lbl] = combinations(leaves_by_true_class[class_lbl], 2)
        leaf_pairs_by_true_class[class_lbl] = combinations(lvs, 2)
    sum_purity = 0.0
    count = 0.0
    for class_lbl in leaf_pairs_by_true_class:
        for pair in leaf_pairs_by_true_class[class_lbl]:
            lca = pair[0].lca(pair[1])
            sum_purity += lca.purity(get_cluster(pair[0]))
            assert(get_cluster(pair[0]) == get_cluster(pair[1]))
            count += 1.0
    if count == 0.0:
        return 1.0
    else:
        return sum_purity / count
