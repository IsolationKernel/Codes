from DTM_filtrations import *
import gudhi as gd
from Lambda_feature import *
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import distance_matrix

def check_symmetric(a, rtol=1e-05, atol=1e-08):
    # check whether array is symmetric
    return np.allclose(a, a.T, rtol=rtol, atol=atol)

def nor_dgm(dgm):
    # set max persistence to 1
    persistence = dgm[:,1]-dgm[:,0]
    num_inf = np.where(persistence == np.inf)[0].shape[0]
    max_perst = persistence[np.argsort(persistence)[-(num_inf+1)]]
    dgm = dgm/max_perst
    persistence = dgm[:,1]-dgm[:,0]
    num_inf = np.where(persistence == np.inf)[0].shape[0]
    max_perst = persistence[np.argsort(persistence)[-(num_inf+1)]]
    assert np.abs(max_perst-1) <= np.finfo(float).eps
    return dgm

def get_pd_dm(dm):
    # compute dim-0,dim-1 PD from distance matrix (Rips)
    eps = 0.001
    rps = gd.RipsComplex(distance_matrix=dm,max_edge_length=np.max(dm)+eps)
    rps_tree = rps.create_simplex_tree(max_dimension=2)
    diag = rps_tree.persistence()
    diag_1 = np.array([list(x[1]) for x in diag if x[0]==1])
    diag_0 = np.array([list(x[1]) for x in diag if x[0]==0])
    return diag_0,diag_1,diag

def get_pd_rips(X):
    # get persistence diagram from Rips filtration
    dm = distance_matrix(X,X)
    return get_pd_dm(dm)

def get_pd_lambda(X,eta,psi,t=100):
    # get persistence diagram from Lambda-filter
    if eta >= 0 :
        _,_,dm,_ = lambda_feature_continous(X,X,eta,psi,t)
    else: # eta<0 -> discrete version
        _,_,dm,_ = lambda_feature_infty(X,X,psi,t)
    return get_pd_dm(dm)

def get_pd_dtm(X,m):
    # get persistence diagram from DTM
    N = np.shape(X)[0]
    k = int(m*N)+1
    p = 1
    dimension_max = 2
    st_DTM = DTMFiltration(X, m, p, dimension_max)  # creating a simplex tree
    diag= st_DTM.persistence()
    diag_1 = np.array([list(x[1]) for x in diag if x[0]==1])
    diag_0 = np.array([list(x[1]) for x in diag if x[0]==0])
    return diag_0,diag_1,diag

def CKNN_dm(data,m):
    # return distnace matrix for CKNN
    # k = [m*N]
    dm = distance_matrix(data,data,p=2)
    N = np.shape(data)[0]
    k = int(m*N)+1
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(data)
    distances, indices = nbrs.kneighbors(data)
    n,d = distances.shape
    col = distances[:,-1].reshape((n,1))
    normalize_term = np.sqrt(np.dot(col,col.T))
    assert np.count_nonzero(np.abs(normalize_term-0)<=np.finfo(float).eps)==0
    assert dm.shape == normalize_term.shape
    dm_cknn = np.divide(dm,normalize_term)
    return dm_cknn 

def get_pd_cknn(X,m):
    # get persistence diagram from CkNN
    dm_cknn = CKNN_dm(X,m)
    return get_pd_dm(dm_cknn)