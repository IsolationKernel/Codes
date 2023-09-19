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

    def fit_transform(self, data):
        self.data = data
        self.centroid = []
        self.centroids_radius = []
        sn = self.data.shape[0]

        # print('''------------------------''')
        # print(self.data)
        n, d = self.data.shape
        IDX = np.array([])  # column index
        V = []
        for i in range(self.t):
            subIndex = sample(range(sn), self.psi)
            self.centroid.append(subIndex)
            tdata = self.data[subIndex, :]

            tree = BallTree(tdata)
            radius, ind = tree.query(tdata, k=2)

            self.centroids_radius.append(radius)
            centerIdx = np.zeros(n)
            dist, ind = tree.query(self.data, k=1)

            V = V + [int(dist[j][0] <= radius[ind[j][0]][1]) for j in range(n)]

            centerIdx = np.array([ind[j][0] for j in range(n)])
            IDX = np.concatenate((IDX, centerIdx + i * self.psi), axis=0)

        IDR = np.tile(range(n), self.t)  # row index

        ndata = csr_matrix((V, (IDR, IDX)), shape=(n, self.t * self.psi))
        return ndata

    def fit(self, data):
        self.data = data
        self.centroid = []
        self.centroids_radius = []
        self.trees = []
        self.radius = []
        sn = self.data.shape[0]
        
        print('''------------------------''')
        print(self.data)
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

    def fit_transform_1(self, data):
        self.data = data
        self.centroid = []
        self.centroids_radius = []
        sn = self.data.shape[0]
        
        # print('''------------------------''')
        # print(self.data)
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
            radius = [] #restore centroids' radius
            radius, ind = tree.query(tdata, k=2)
  
            self.centroids_radius.append(radius)
            centerIdx = np.zeros(n)
            dist, ind = tree.query(self.data, k=1)
            
            #print('radius[ind[0][0]][1] = ',radius[ind[0][0]][1],'dist[0][0] = ',dist[0][0])
            V = V+[int(dist[j][0] <= radius[ind[j][0]][1]) for j in range(n)]
            #print(V)
            centerIdx = np.array([ind[j][0] for j in range(n)])
            IDX = np.concatenate((IDX, centerIdx + i * self.psi), axis=0)
        IDR = np.tile(range(n), self.t) #row index
        #V = np.ones(self.t * n) #value
        ndata = csr_matrix((V, (IDR, IDX)), shape=(n, self.t * self.psi))
        return ndata