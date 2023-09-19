import scipy.io as io
import time
from IDK2 import *
import os


if __name__ == "__main__":
    raw_mat = io.loadmat("./datasets/geolife.mat")
    data = raw_mat['data'][0]
    label = raw_mat['label'][0]

    # IDK
    print("===============IDK===============")
    psi = 4
    print("Start IDK calculating...")
    t1 = time.perf_counter()
    idkmap = idk_kernel_map(data, psi)
    t2 = time.perf_counter()
    if not os.path.exists('embeddings/'):
        os.makedir('embeddings/')
    io.savemat("./embeddings/geolife_idk_" + str(psi)+".mat", {"data": idkmap, "class": label})
    print("IDK_psi = %f   time cost = %s" % (psi, t2 - t1))
