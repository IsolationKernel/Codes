import scipy.io as scio
filepath1 = "./datasets/Data2/TRAFFIC.mat"
data = scio.loadmat(filepath1)

tracks_traffic = data['tracks_traffic'][:,0]
labels = data['truth'][:,1]
alltjlist = []
for i in range(len(tracks_traffic)):
    alltjlist.append(tracks_traffic[i].T.tolist())

