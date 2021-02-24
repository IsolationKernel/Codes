Isolation Kernel
================

Isolation Kernel (IK) is first introduced as an alternative to existing data independent kernels such as Gaussian and Laplacian kernels. IK has the following distinctive features:

* IK is derived directly from data and has no closed from expression.
* Its kernel similarity is data dependent: two points in sparse region are more similar than two points of equal inter-point distance in dense region.
* It has a sparse and finite-dimensional feature map.

 

IK's unique feature map (feature III) has led to the following advantages:

* Enabling efficient dot product and GPU acceleration (if realized using NN methods) [2, 4]
* Using Isolation Distribution Kernel (IDK) with kernel mean map, it enables linear time algorithms which would otherwise be impossible---algorithms relying on a point-to-point distance/kernel for their basic operations have at least quadratic time [5, 6, 9]

 

IK's data dependency (feature II) has led to improved task-specific performance in the following tasks:

* Classification: SVM [1], SVM multi-instance learning [3], large scale online kernel learning [4]
* Clustering: DBSCAN [2], psKC [9], Agglomerative Hierarchical Clustering [8]
* Anomaly detection: point anomalies [5], group anomalies [6]
* Graph kernel [7]

 

Note that Isolation Kernel is not one kernel function such as Gaussian kernel, but a class of kernels which has different kernel distributions depending on the space partitioning mechanism employed. There are currently three possible implementations: (a) Isolation Forest used in [1,7]; (b) Voronoi diagram used in [2,3,4,8,9]; and (c) hyperspheres used in [5,6].


The codes provided here are IK [1,2], IDK [5,6], IGK [7] and IK-OGD [4].

References
----------

[1] Kai Ming Ting, Yue Zhu, Zhi-Hua Zhou (2018). Isolation Kernel and Its Effect on SVM. Proceedings of The ACM SIGKDD Conference on Knowledge Discovery and Data Mining. 2329-2337.

[2] Xiaoyu Qin, Kai Ming Ting, Ye Zhu and Vincent Cheng Siong Lee (2019). Nearest-Neighbour-Induced Isolation Similarity and Its Impact on Density-Based Clustering.  Proceedings of The Thirty-Third AAAI Conference on Artificial Intelligence. 4755-4762.

[3] Bi-Cun Xu, Kai Ming Ting, Zhi-Hua Zhou (2019). Isolation Set-Kernel and Its Application to Multi-Instance Learning. Proceedings of The ACM SIGKDD Conference on Knowledge Discovery and Data Mining. 941-949.

[4] Kai Ming Ting, Jonathan R. Wells, Takashi Washio: Isolation Kernel: The X Factor in Efficient and Effective Large Scale Online Kernel Learning. [CoRR abs/1907.01104 (2019)](https://dblp.uni-trier.de/db/journals/corr/corr1907.html)

[5] Kai Ming Ting, Bi-Cun Xu, Washio Takashi, Zhi-Hua Zhou (2020). Isolation Distributional Kernel: A new tool for kernel based anomaly detection. Proceedings of The ACM SIGKDD Conference on Knowledge Discovery and Data Mining. 198-206.

[6] Kai Ming Ting, Bi-Cun Xu, Washio Takashi, Zhi-Hua Zhou (2020). Isolation Distributional Kernel: A new tool for kernel based point and group anomaly detections. https://arxiv.org/abs/2009.12196.

[7] Bi-Cun Xu, Kai Ming Ting, Yuan Jiang (2021). Isolation Graph Kernel. Proceedings of The Thirty-Fifth AAAI Conference on Artificial Intelligence.

[8] Xin Han, Ye Zhu, Kai Ming Ting, Gang Li (2020). The Impact of Isolation Kernel on Agglomerative Hierarchical Clustering Algorithms. [CoRR abs/2010.05473](https://dblp.org/db/journals/corr/corr2010.html#abs-2010-05473)

[9] Kai Ming Ting, Jonathan R. Wells, Ye Zhu (2020). Clustering based on Point-Set Kernel. [CoRR abs/2002.05815](https://dblp.org/db/journals/corr/corr2002.html#abs-2002-05815)



All IK related codes are under the Non-Commercial Use License.
