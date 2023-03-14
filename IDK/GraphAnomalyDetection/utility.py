import numpy as np
import scipy.io as scio
import os
from sklearn.neighbors import KernelDensity
from scipy.spatial.distance import cdist


def load_graph(filename):
    dir = 'dataset'
    graph = scio.loadmat(os.path.join(dir, filename))
    attr = graph['Attributes'].A
    adj = graph['Network']
    label = graph['Label']
    return attr, adj, label