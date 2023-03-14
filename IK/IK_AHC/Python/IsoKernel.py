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

import numbers
from warnings import warn

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import euclidean_distances
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted, check_random_state

MAX_INT = np.iinfo(np.int32).max
MIN_FLOAT = np.finfo(float).eps


class IsolationKernel(TransformerMixin, BaseEstimator):
    """  Build Isolation Kernel feature vector representations via the feature map
    for a given dataset.

    Isolation kernel is a data dependent kernel measure that is
    adaptive to local data distribution and has more flexibility in capturing
    the characteristics of the local data distribution. It has been shown promising
    performance on density and distance-based classification and clustering problems.

    This version uses Voronoi diagrams to split the data space and calculate Isolation
    kernel Similarity. Based on this implementation, the feature
    in the Isolation kernel space is the index of the cell in Voronoi diagrams. Each
    point is represented as a binary vector such that only the cell the point falling
    into is 1.

    Parameters
    ----------
    n_estimators : int, default=200
        The number of base estimators in the ensemble.

    max_samples : int, default="auto"
        The number of samples to draw from X to train each base estimator.

            - If int, then draw `max_samples` samples.
            - If float, then draw `max_samples` * X.shape[0]` samples.
            - If "auto", then `max_samples=min(8, n_samples)`.

    random_state : int, RandomState instance or None, default=None
        Controls the pseudo-randomness of the selection of the feature
        and split values for each branching step and each tree in the forest.

        Pass an int for reproducible results across multiple function calls.
        See :term:`Glossary <random_state>`.

    References
    ----------
    .. [1] Qin, X., Ting, K.M., Zhu, Y. and Lee, V.C. 
    "Nearest-neighbour-induced isolation similarity and its impact on density-based clustering". 
    In Proceedings of the AAAI Conference on Artificial Intelligence, Vol. 33, 2019, July, pp. 4755-4762

    Examples
    --------
    >>> from IsoKernel import IsoKernel
    >>> import numpy as np
    >>> X = [[0.4,0.3], [0.3,0.8], [0.5, 0.4], [0.5, 0.1]]
    >>> ik = IsoKernel.fit(X)
    >>> ik.transform(X)
    """

    def __init__(self, n_estimators=200, max_samples="auto", random_state=None) -> None:
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.random_state = random_state
        self.embedding = None

    def fit(self, X, y=None):
        """ Fit the model on data X.
        Parameters
        ----------
        X : np.array of shape (n_samples, n_features)
            The input instances.
        Returns
        -------
        self : object
        """

        X = check_array(X)
        n_samples = X.shape[0]
        if isinstance(self.max_samples, str):
            if self.max_samples == "auto":
                max_samples = min(16, n_samples)
            else:
                raise ValueError(
                    "max_samples (%s) is not supported."
                    'Valid choices are: "auto", int or'
                    "float"
                    % self.max_samples
                )
        elif isinstance(self.max_samples, numbers.Integral):
            if self.max_samples > n_samples:
                warn(
                    "max_samples (%s) is greater than the "
                    "total number of samples (%s). max_samples "
                    "will be set to n_samples for estimation."
                    % (self.max_samples, n_samples)
                )
                max_samples = n_samples
            else:
                max_samples = self.max_samples
        else:  # float
            if not 0.0 < self.max_samples <= 1.0:
                raise ValueError(
                    "max_samples must be in (0, 1], got %r" % self.max_samples_
                )
            max_samples = int(self.max_samples_ * X.shape[0])
        self.max_samples_ = max_samples
        self._fit(X)
        self.is_fitted_ = True
        return self

    def _fit(self, X):
        n_samples = X.shape[0]
        self.max_samples_ = min(self.max_samples_, n_samples)
        random_state = check_random_state(self.random_state)
        self._seeds = random_state.randint(MAX_INT, size=self.n_estimators)

        for i in range(self.n_estimators):
            rnd = check_random_state(self._seeds[i])
            center_index = rnd.choice(
                n_samples, self.max_samples_, replace=False)
            if i == 0:
                self.center_index_set = np.array([center_index])
            else:
                self.center_index_set = np.append(
                    self.center_index_set, np.array([center_index]), axis=0)
        self.unique_index = np.unique(self.center_index_set)
        self.center_data = X[self.unique_index]
        return self

    def similarity(self, X):
        """ Compute the isolation kernel simalarity matrix of X.
        Parameters
        ----------
        X: array-like of shape (n_instances, n_features)
            The input instances.
        Returns
        -------
        The simalarity matrix are organised as a n_instances * n_instances matrix.
        """

        embed_X = self.transform(X)
        return np.inner(embed_X, embed_X) / self.n_estimators

    def transform(self, X):
        """ Compute the isolation kernel feature of X.
        Parameters
        ----------
        X: array-like of shape (n_instances, n_features)
            The input instances.
        Returns
        -------
        The finite binary features based on the kernel feature map.
        The features are organised as a n_instances by psi*t matrix.
        """

        check_is_fitted(self)
        X = check_array(X)
        n, m = X.shape
        X_dists = euclidean_distances(X, self.center_data)

        for i in range(n):
            mapping_array = np.zeros(self.unique_index.max() + 1,
                                     dtype=X_dists.dtype)
            mapping_array[self.unique_index] = X_dists[i]
            x_center_dist_mat = mapping_array[self.center_index_set]

            nearest_center_index = np.argmin(x_center_dist_mat, axis=1)
            ik_value = np.eye(self.max_samples_, dtype=int)[
                nearest_center_index].flatten()[np.newaxis]
            if i == 0:
                embedding = ik_value
            else:
                embedding = np.append(embedding, ik_value, axis=0)
        self.embedding = embedding
        return embedding.astype(float)
