# Copyright 2021 Xin Han
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

import time

import numpy as np
from numpy.core.defchararray import center
from scipy.spatial.distance import cdist


class IKMapper():

    def __init__(self,
                 t,
                 psi,
                 ) -> None:
        self._t = t
        self._psi = psi
        self._embeding_metrix = None
        self._center_index_set = None
        self._center_data = None
        self._unique_index = None

    def fit(self, X):
        """ Fit the model on data X.

        Parameters
        ----------
        X : np.array of shape (n_samples, n_features)
            The input instances. 
        Returns
        -------
        self : object
        """
        # correct sample size
        n = X.shape[0]
        self._psi = min(self._psi, n)
        for i in range(self._t):
            center_index = np.random.choice(n, self._psi, replace=False)
            if i == 0:
                self._center_index_set = np.array([center_index])
            else:
                self._center_index_set = np.append(
                    self._center_index_set, np.array([center_index]), axis=0)
            center = X[center_index]
            nt_dis = cdist(center, X)
            nearest_center_index = np.argmin(nt_dis, 0)
            ik_value = np.eye(self._psi, dtype=int)[nearest_center_index]
            if i == 0:
                self._embeding_metrix = ik_value
            else:
                self._embeding_metrix = np.concatenate(
                    (self._embeding_metrix, ik_value), axis=1)
        assert self._embeding_metrix.shape == (
            n, self._psi * self._t), "Shape of ik_value is incorrect ! Except: %s, Get:%s" % ((n, self._psi * self._t), self._embeding_metrix.shape)
        self._unique_index = np.unique(self._center_index_set)
        self._center_data = X[self._unique_index]
        return self

    @property
    def embeding_mat(self):
        """Get the isolation kernel map feature of fit dataset.
        """
        return self._embeding_metrix

    def transform(self, x):
        """ Compute the isolation kernel map feature of x.

        Parameters
        ----------
        x: array-like of shape (1, n_features)
            The input instances.

        Returns
        -------
        ik_value: np.array of shape (sample_size times n_members,)
            The isolation kernel map of the input instance.
        """

        x_dists = cdist(x.reshape(1, -1), self._center_data).flatten()
        mapping_array = np.zeros(self._unique_index.max()+1,
                                 dtype=x_dists.dtype)
        mapping_array[self._unique_index] = x_dists
        x_center_dist_mat = mapping_array[self._center_index_set]

        nearest_center_index = np.argmin(x_center_dist_mat, axis=1)
        ik_value = np.eye(self._psi, dtype=int)[
            nearest_center_index].flatten()
        assert ik_value.shape == (
            self._psi * self._t,), "Shape of ik_value is incorrect ! Except: %s, Get:%s" % ((self._psi * self._t,), ik_value.shape)
        return ik_value


if __name__ == '__main__':

    from src.utils.deltasep_utils import create_dataset
    dataset = create_dataset(5, 5000, num_clusters=3)
    n = 200
    data = np.array([pt[:3] for pt in dataset[:n]])
    sts = time.time()
    ikm = IKMapper(t=200, psi=13)
    print("start train")
    ets = time.time()
    ik_maper = ikm.fit(data)
    f_ets = time.time()
    cal_time = 0
    for dt in dataset[n:]:
        addx = dt[:3]
        test = ik_maper.transform(addx)
    add_end = time.time()
    print("add time:%s" % (add_end-f_ets))
    print("fit time: %s" % (f_ets-ets))
