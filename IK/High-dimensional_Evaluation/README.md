# Breaking the curse of dimensionality with Isolation Kernel

The codes provided here are a demo of the method described by the paper: 
Ting, K.M., Washio, T., Zhu, Y. and Xu, Y., 2021. Breaking the curse of dimensionality with Isolation Kernel.

Uploaded by Ye Zhu and Yang Xu, September 2021, version 1.0.

This software is under GNU General Public License version 3.0 (GPLv3)

The following scripts can be used to reproduce the results reported in the paper:

- Ne_count.m: Calculate N_e numbers under different distance measures and kernels as shown in Figure 1, 4 and Table 1.
- VDM.m: Generating results for Figure 1 (b)
- testIR.m: The retrieval precision of 5 nearest neighbour as shown in Table 2
- testIndexing.ipynb: Brute force vs Ball tree index as shown in Table 2
- testclustering.m: Clustering evaluation results in Table 3
- testTSNE.m: t-SNE visulaisation as shown Table 5 and Table 6
- EstimateID.m: Estimated intrinsic dimensions as shown in Figure 3
- Hubness.m: Calculation for Figure5
- IKtransformation.m: Transform the original data to IK feature space for indexing and SVM classification. 
- SVMclassification.ipynb: SVM classification for Table 4



Please unzip the datasets under "Clustering and Indexing datasets" before running these scripts. All datasets used for SVM classification in Table 4 can be directly obtained from https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/. The script to read the original data in the SVM format can be obtained from https://github.com/cjlin1/libsvm.



The codes for indexing search and SVM classification evaluation are implemented in Python, other tasks are based on Matlab. 

# References
    Ting, K.M., Washio, T., Zhu, Y. and Xu, Y., 2021. Breaking the curse of dimensionality with Isolation Kernel.
