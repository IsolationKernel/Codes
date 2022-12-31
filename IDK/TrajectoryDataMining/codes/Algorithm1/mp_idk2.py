from sklearn.metrics import roc_auc_score
import numpy as np
from multiprocessing.pool import Pool
from codes.Algorithm1.IDK2 import idk_square
def run_idk2(processed_tdata, truths, psi1, psi2):
    t1 = 100
    t2 = 100
    idk_score = idk_square(processed_tdata, psi1, t1, psi2, t2)
    y_labels=truths
    auc = roc_auc_score(y_labels, idk_score)
    # sorted_idk_score_idx = np.argsort(idk_score)
    print("psi1:" +str(psi1) +"; psi2:"+str(psi2) + "has done! auc: "+ str(auc))
    return auc

def mp_idk2(processed_tdata, truths):
    print("begin...")
    per_traj_count = []
    for i in range(len(processed_tdata)):
        per_traj_count.append(len(processed_tdata[i]))
    allpoints_count = np.sum(per_traj_count)
    p = Pool(128)
    psilist = np.array([2, 4, 8, 16, 32, 64])
    for psi1 in psilist[psilist < allpoints_count]:
        for psi2 in psilist[psilist < len(processed_tdata)]:
            p.apply_async(run_idk2, args=(processed_tdata, truths, psi1, psi2))
    p.close()
    p.join()
    print("end...")
