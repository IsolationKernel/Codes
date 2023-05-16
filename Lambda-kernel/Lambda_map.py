import numpy as np
from random import sample
from scipy.spatial.distance import cdist
from scipy.sparse import csr_matrix

class Lambda_map:
    data = None
    centroid = []
    def __init__(self, psi, t):
        self.psi = psi
        self.t = t
        self.iso = 0
        self.dm = None

    def fit(self, data):
        self.data = data
        self.centroid = []
        self.centroids_radius = []
        sn = self.data.shape[0]
        for i in range(self.t):
            subIndex = sample(range(sn), self.psi)
            self.centroid.append(subIndex)
            tdata = self.data[subIndex, :]
            tt_dis = cdist(tdata, tdata)
            radius = [] #restore centroids' radius
            for r_idx in range(self.psi):
                r = tt_dis[r_idx]
                r[r<0] = 0
                r = np.delete(r,r_idx)
                radius.append(np.min(r))
            self.centroids_radius.append(radius)

    def transform(self, newdata):
            # Lambda=infty, euclidean distance
            assert self.centroid != None, "invoke fit() first!"
            n, _ = newdata.shape
            IDX = np.array([])
            V = []
            for i in range(self.t):
                subIndex = self.centroid[i]
                radius = self.centroids_radius[i]
                tdata = self.data[subIndex, :]
                dis = cdist(tdata, newdata) #-------------------------
                centerIdx = np.argmin(dis, axis=0)
                for j in range(n):
                    # HyperSphere
                    #V.append(int(dis[centerIdx[j], j] <= radius[centerIdx[j]]))
                    # Voronoi diagram
                    V.append(1) 
                IDX = np.concatenate((IDX, centerIdx + i * self.psi), axis=0)
            IDR = np.tile(range(n), self.t)
            ndata = csr_matrix((V, (IDR, IDX)), shape=(n, self.t * self.psi))
            return ndata

    def transform_continous(self, newdata):
        # Lambda<infty, euclidean distance
        assert self.centroid != None, "invoke fit() first!"
        n, _ = newdata.shape
        IDX = np.array([])
        V = []
        for i in range(self.t):
            subIndex = self.centroid[i]
            radius = self.centroids_radius[i]
            tdata = self.data[subIndex, :]
            dis = cdist(tdata, newdata)#-------------------------------
            centerIdx = np.argmin(dis, axis=0)
            for j in range(n):
                # continous case
                # V.append(np.exp(-1*eta*dis[centerIdx[j], j]))
                V.append(dis[centerIdx[j], j])
            IDX = np.concatenate((IDX, centerIdx + i * self.psi), axis=0)
        IDR = np.tile(range(n), self.t)
        ndata = csr_matrix((V, (IDR, IDX)), shape=(n, self.t * self.psi))
        return ndata


    
    

    

