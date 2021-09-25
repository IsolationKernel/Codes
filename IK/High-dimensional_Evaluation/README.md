# Breaking the curse of dimensionality with Isolation Kernel

This code for demonstration of using Isolation kernel to break the curse of dimensionality.

Uploaded by Ye Zhu, Deakin University, Sep 2021, version 1.0.

This software is under GNU General Public License version 3.0 (GPLv3)

This code is a demo of method described by the following publication: Ting, K.M., Washio, T., Zhu, Y. and Xu, Y., 2021. Breaking the curse of dimensionality with Isolation Kernel.

The following scripts can be used to reproduce the results in the paper:

- Ne_count.m: Calculate N_e numbers under different distance measures and kernels as shown in Figure 1
- testIR.m: The retrieval precision of 5 nearestneighbour as shown in Table 2
- testIndexing.m: Brute force vs Ball tree index as shown in Table 2
- testclustering.m: Clustering evaluation results in Table 3
- testTSNE.m: t-SNE visulaisation as shown Table 5 and Table 6
- EstimateID.m: Estimated intrinsic dimensions as shown in Figure 3
- Hubness.m: Calculation for Figure5



All datasets used for SVM classification in Table 4 are directly obtained from https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/. The code for classification evaluation is implemented using Python. 
