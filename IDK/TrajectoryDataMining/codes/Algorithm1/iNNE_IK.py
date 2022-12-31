import numpy as np
from random import sample
from scipy.spatial.distance import cdist
from scipy.sparse import csr_matrix
from sklearn.neighbors import BallTree
class iNN_IK:
    data = None
    centroid = []
    def __init__(self, psi, t):
        self.psi = psi
        self.t = t

    def fit(self, data):
        self.data = data
        self.centroid = []
        self.centroids_radius = []
        self.trees = []
        self.radius = []
        sn = self.data.shape[0]
        
#         print('''------------------------''')
#         print(self.data)
        n, d = self.data.shape
        IDX = np.array([])  #column index
        V = []
        for i in range(self.t):
            #print('i=',i)
            subIndex = sample(range(sn), self.psi)
            self.centroid.append(subIndex)
            tdata = self.data[subIndex, :]
            #tt_dis = cdist(tdata, tdata)
            tree = BallTree(tdata)
            radius, ind = tree.query(tdata, k=2)
            self.trees.append(tree)
            self.radius.append(radius)
#            radius = [] #restore centroids' radius
#             radius, ind = tree.query(tdata, k=2)
#             self.centroids_radius.append(radius)

    def fit_transform(self, data):
        self.data = data
        self.centroid = []
        self.centroids_radius = []
        sn = self.data.shape[0]
        n, d = self.data.shape
        IDX = np.array([])  #column index
        V = []
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
            nt_dis = cdist(tdata, self.data)
            centerIdx = np.argmin(nt_dis, axis=0)
            for j in range(n):
                V.append(int(nt_dis[centerIdx[j],j] <= radius[centerIdx[j]]))
            IDX = np.concatenate((IDX, centerIdx + i * self.psi), axis=0)
        IDR = np.tile(range(n), self.t) #row index
        #V = np.ones(self.t * n) #value
        ndata = csr_matrix((V, (IDR, IDX)), shape=(n, self.t * self.psi))
        return ndata
        
    def fit_transform_1(self, data, point):
        self.data = data
        self.point = point
        self.centroid = []
        self.centroids_radius = []
        sn = self.data.shape[0]

        n, d = self.data.shape
        m, di = self.point.shape
        IDX = np.array([])#column index
        IDX_1 = np.array([])
        V = []
        U = []
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
            nt_dis = cdist(tdata, self.data)
            centerIdx = np.argmin(nt_dis, axis=0)
            nt_dis_1 = cdist(tdata, self.point)
            centerIdx_1 = np.argmin(nt_dis_1, axis=0)
            for j in range(n):
                V.append(int(nt_dis[centerIdx[j],j] <= radius[centerIdx[j]]))
            for k in range(m):
                U.append(int(nt_dis_1[centerIdx_1[k],k] <= radius[centerIdx_1[k]]))
            IDX = np.concatenate((IDX, centerIdx + i * self.psi), axis=0)
            IDX_1 = np.concatenate((IDX_1,centerIdx_1 + i * self.psi), axis=0)
        IDR = np.tile(range(n), self.t) #row index
        IDR_1 = np.tile(range(m), self.t)
        #V = np.ones(self.t * n) #value
        ndata = csr_matrix((V, (IDR, IDX)), shape=(n, self.t * self.psi))
        mdata = csr_matrix((U, (IDR_1, IDX_1)), shape=(m, self.t * self.psi))
        return ndata, mdata

    def transform_1(self,newdata):
        
        n, d = newdata.shape
        IDX = np.array([])
        V = []
        for i in range(self.t):
            tree = self.trees[i]
            #radius = [] #restore centroids' radius
            radius   = self.radius[i]
            #radius, ind = tree.query(tdata, k=2)
            centerIdx = np.zeros(n)
            dist, ind = tree.query(newdata, k=1)
            V = V+[int(dist[j][0] <= radius[ind[j][0]][1]) for j in range(n)]
            
            centerIdx = np.array([ind[j][0] for j in range(n)])
            IDX = np.concatenate((IDX, centerIdx + i * self.psi), axis=0)
       # print('V.shape = ',len(V))
        IDR = np.tile(range(n), self.t)
        #print('V.shape = ',len(V),'IDX.shape=',len(IDX[0]),'IDR.shape=',len(IDR))
        ndata = csr_matrix((V, (IDR, IDX)), shape=(n, self.t * self.psi))
        return ndata.toarray()
    
    def transform(self, newdata):
        n, d = newdata.shape
        IDX = np.array([])
        V = []
        for i in range(self.t):
            subIndex = self.centroid[i]
            radius = self.centroids_radius[i]
            tdata = self.data[subIndex, :]
            dis = cdist(tdata, newdata)
            centerIdx = np.argmin(dis, axis=0)
            for j in range(n):
                V.append(int(dis[centerIdx[j], j] <= radius[centerIdx[j]]))
            IDX = np.concatenate((IDX, centerIdx + i * self.psi), axis=0)
        IDR = np.tile(range(n), self.t)
        ndata = csr_matrix((V, (IDR, IDX)), shape=(n, self.t * self.psi))
        return ndata

