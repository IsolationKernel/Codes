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

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# temporary solution for relative imports in case pyod is not installed
# if pyod is installed, no need to use the following line

import pytest
from IsoKAHC import IsoKAHC
from IsoKernel import IsolationKernel
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import load_wine
from utils import metrics


@pytest.fixture
def data():
    return load_wine(return_X_y=True)


def test_isokhc_performance(data):
    X, y = data
    test_idk = IsoKAHC(n_estimators=200, max_samples=3, method="average")
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    ik_den = test_idk.fit_transform(X)
    assert metrics.dendrogram_purity(ik_den, y) > 0.8


@pytest.mark.parametrize("method", ['single', 'complete', 'average', 'weighted'])
def test_isokhc_work(data, method):
    X, y = data
    # Test IsoKHC
    clf = IsoKAHC(max_samples=8, method=method)
    clf.fit(X)
    pred = clf.fit_transform(X)


def test_pre_kernel(data):
    X, y = data
    ik = IsolationKernel(n_estimators=200, max_samples=3)
    ik = ik.fit(X[:44])
    test_idk = IsoKAHC(method="single", iso_kernel=ik)
    ik_den = test_idk.fit_transform(X)
    print(metrics.dendrogram_purity(ik_den, y))
