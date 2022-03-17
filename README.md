# A brief history of Isolation-based methods

Isolation-based methods refer to methods that employ an isolation mechanism to construct isolating partitions in the input space. The first method is called Isolation Forest or iForest [1], a point anomaly detector, reported in IEEE ICDM 2008. The intuition is that anomalies are rare and different from normal points; thus each anomaly is more amenable to isolation than normal points. A point is said to be isolated if it is contained within an isolating partition that isolates it from the rest of the points in a sample. 


Isolation Forest is one of the most effective and efficient anomaly detectors created thus far. Since its introduction, it has been used widely in academia and industries. Its limitations, due to the use of tree structures, have been studied by different researchers. One improvement beyond tree structures is iNNE [2] which employs hyperspheres as the isolation mechanism.


The development of isolation-based methods has grown outside the confines of anomaly detection since. In 2010, Isolation Forest is shown to be a special case of mass estimation [3] (i.e., an alternative to density estimation.)  


In 2018, a data dependent kernel called Isolation Kernel [4] or IK is first introduced as an alternative to data independent kernels such as Gaussian and Laplacian kernels. It has a unique characteristic:  two points, as measured by Isolation Kernel derived with a dataset in a sparse region, are more similar than the same two points, as measured by Isolation Kernel derived with a dataset in a dense region. This characteristic is derived from data directly; and IK has no closed form expression and does not require learning. Isolation Kernel has three implementations using different isolation mechanisms up to 2021 [4,5,6]. IK has been shown to be the key in achieving large scale online kernel learning [7] and improving the efficacy & efficienct of t-SNE [15].


In 2020, Isolation Distributional Kernel or IDK is introduced to measure the similarity of two distributions [6], based on the framework of kernel mean embedding [8]. The first application of IDK is a kernel-based point anomaly detector that needs no learning, unlike OCSVM [9]. Through IDK point anomaly detector, Isolation Forest is linked to a kernel-based method for the very first time. IDK has since been applied to group anomaly detection [10], graph classification via Isolation Graph Kernel [11], multi-instance learning [12]. IDK can be interpreted as a kernel density estimator called Isolation Kernel Density Estimator [13]. IDK has also been applied to produce a new class of clustering algorithm called psKC (or point-set Kernel Clustering) [14]. Up to early 2022, psKC is the only clustering algorithm which is both effective and efficient---a quality which is all but nonexistent in current clustering algorithms. It is also the only kernel-based clustering which has linear time complexity. 


# References
[1] Fei Tony Liu, Kai Ming Ting, Zhi-Hua Zhou (2008) Isolation Forest. Proceedings of IEEE ICDM, 413-422.[[pdf]](https://github.com/IsolationKernel/Codes/blob/main/PDF/Isolation%20Forest.pdf)

[2] Tharindu R. Bandaragoda, Kai Ming Ting, David Albrecht, Fei Tony Liu, Jonathan R. Wells (2018). Isolation-based Anomaly Detection using Nearest Neighbour Ensembles. Computational Intelligence. Doi:10.1111/coin.12156.[[pdf]](https://github.com/IsolationKernel/Codes/blob/main/PDF/Isolation-based%20Anomaly%20Detection%20using%20Nearest%20Neighbour%20Ensembles.pdf)

[3] Kai Ming Ting, Guang-Tong Zhou. Fei Tony Liu, Swee Chuan Tan (2010). Mass Estimation and Its Applications. Proceedings of The 16th ACM SIGKDD Conference on Knowledge Discovery and Data Mining. 989-998.[[pdf]](https://github.com/IsolationKernel/Codes/blob/main/PDF/Mass%20Estimation%20and%20Its%20Applications.pdf)

[4] Kai Ming Ting, Yue Zhu, Zhi-Hua Zhou (2018). Isolation Kernel and Its Effect on SVM. Proceedings of The ACM SIGKDD Conference on Knowledge Discovery and Data Mining. 2329-2337.[[pdf]](https://github.com/IsolationKernel/Codes/blob/main/PDF/Isolation%20Kernel%20and%20its%20effect%20on%20SVM.pdf)

[5] Xiaoyu Qin, Kai Ming Ting, Ye Zhu, Vincent Cheng Siong Lee (2019). Nearest-Neighbour-Induced Isolation Similarity and Its Impact on Density-Based Clustering. Proceedings of The Thirty-Third AAAI Conference on Artificial Intelligence. 4755-4762.[[pdf]](https://github.com/IsolationKernel/Codes/blob/main/PDF/Nearest-Neighbour-Induced%20Isolation%20Similarity%20and%20Its%20Impact%20on%20Density-Based%20Clustering.pdf)

[6] Kai Ming Ting, Bi-Cun Xu, Washio Takashi, Zhi-Hua Zhou (2020). Isolation Distributional Kernel: A new tool for kernel based anomaly detection. Proceedings of The ACM SIGKDD Conference on Knowledge Discovery and Data Mining. 198-206.[[pdf]](https://doi.org/10.1145/3394486.3403062)

[7] Kai Ming Ting, Jonathan R. Wells, Takashi Washio (2021). Isolation Kernel: The X Factor in Efficient and Effective Large Scale Online Kernel Learning. Data Mining and Knowledge Discovery.[[pdf]](https://github.com/IsolationKernel/Codes/blob/main/PDF/Isolation%20Kernel-The%20X%20Factor%20in%20Efficient%20and%20Effective%20Large%20Scale%20Online%20Kernel%20Learning.pdf)

[8] Alex Smola, Arthur Gretton, Le Song, Bernhard Schölkopf. 2007. A Hilbert Space Embedding for Distributions. In Algorithmic Learning Theory, Marcus Hutter, Rocco A. Servedio, and Eiji Takimoto (Eds.). Springer, 13–31.

[9] Bernhard Schölkopf, John C. Platt, John C. Shawe-Taylor, Alex J. Smola, Robert C. Williamson. 2001. Estimating the Support of a High-Dimensional Distribution. Neural Computing 13, 7 (2001), 1443–1471.

[10] Kai Ming Ting, Bi-Cun Xu, Washio Takashi, Zhi-Hua Zhou (2022). Isolation Distributional Kernel: A new tool for kernel based point and group anomaly detections. IEEE Transactions on Knowledge and Data Engineering. ieeexplore.ieee.org/document/9573389.[[pdf]](https://github.com/IsolationKernel/Codes/blob/main/PDF/Isolation%20Distributional%20Kernel-A%20New%20Tool%20for%20point%20and%20group%20anomaly%20detection.pdf)

[11] Bi-Cun Xu, Kai Ming Ting, Yuan Jiang (2021). Isolation Graph Kernel. Proceedings of The Thirty-Fifth AAAI Conference on Artificial Intelligence. 10487-10495.[[pdf]](https://ojs.aaai.org/index.php/AAAI/article/view/17255)

[12] Bi-Cun Xu, Kai Ming Ting, Zhi-Hua Zhou (2019). Isolation Set-Kernel and Its Application to Multi-Instance Learning. Proceedings of The ACM SIGKDD Conference on Knowledge Discovery and Data Mining. 941-949.[[pdf]](https://github.com/IsolationKernel/Codes/blob/main/PDF/Isolation%20Set-Kernel%20and%20Its%20Application%20to%20Multi-Instance%20Learning.pdf)

[13]  Kai Ming Ting, Takashi Washio, Jonathan R. Wells, Hang Zhang (2021). Isolation Kernel Density Estimation. Proceedings of IEEE ICDM.[[pdf]](https://github.com/IsolationKernel/Codes/blob/main/PDF/Isolation_Kernel_Density_Estimation.pdf)

[14] Kai Ming Ting, Jonathan R. Wells, Ye Zhu (2022) Point-set Kernel Clustering. IEEE Transactions on Knowledge and Data Engineering.[[pdf]](https://github.com/IsolationKernel/Codes/blob/main/PDF/Point-Set%20Kernel%20Clustering.pdf)

[15] Ye Zhu, Kai Ming Ting (2021) Improving the Effectiveness and Efficiency of Stochastic Neighbour Embedding with Isolation Kernel. Journal of Artificial Intelligence Research 71, 667-695.[[pdf]](https://github.com/IsolationKernel/Codes/blob/main/PDF/Improving%20the%20Effectiveness%20and%20Efficiency%20of%20Stochastic%20Neighbour%20Embedding%20with%20Isolation%20Kernel.pdf)
