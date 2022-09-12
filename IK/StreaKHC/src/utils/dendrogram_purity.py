"""
Copyright (C) 2017 University of Massachusetts Amherst.
This file is part of "xcluster"
http://github.com/iesl/xcluster
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

from itertools import combinations, groupby

import numpy as np
from tqdm import tqdm
from tqdm._tqdm import trange


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
    def get_cluster(x): return x.pts[0][0]
    cluster_to_leaves = {c: list(ls)
                         for c, ls in groupby(sorted(leaves, key=get_cluster),
                                              get_cluster)}
    leaf_to_cluster = {l: l.pts[0][0] for l in leaves}
    non_singleton_leaves = [l for l in leaf_to_cluster.keys()
                            if len(cluster_to_leaves[leaf_to_cluster[l]]) > 1]
    if len(non_singleton_leaves) == 0.0:
        return 1.0
    assert(len(non_singleton_leaves) > 0)

    # For n samples, sample a leaf uniformly at random then select another leaf
    # from the same class unformly at random.
    samps = len(non_singleton_leaves) * 5  # TODO (AK): pick 5 in a better way.
    unnormalized_purity = 0.0
    for i in trange(samps):
        rand_leaf = np.random.choice(non_singleton_leaves)
        cluster = leaf_to_cluster[rand_leaf]
        rand_cluster_member = np.random.choice(cluster_to_leaves[cluster])
        # Make sure we get two distinct leaves
        while rand_cluster_member == rand_leaf:
            assert(leaf_to_cluster[rand_leaf] ==
                   leaf_to_cluster[rand_cluster_member])
            rand_cluster_member = np.random.choice(cluster_to_leaves[cluster])

        # Find their lowest common ancestor and compute cluster purity.
        assert(leaf_to_cluster[rand_leaf] ==
               leaf_to_cluster[rand_cluster_member])
        lca = rand_leaf.lca(rand_cluster_member)
        unnormalized_purity += lca.purity(cluster=cluster)
    return unnormalized_purity / samps


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
