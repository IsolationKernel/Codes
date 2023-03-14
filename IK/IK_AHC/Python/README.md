# IK_AHC

Agglomerative hierarchical clustering (AHC) is one of the popular clustering approaches. AHC can generate a dendrogram that provides richer information and insights from a dataset than partitioning clustering. However, a major problem with existing distance-based AHC methods is: it fails to effectively identify adjacent clusters with varied densities, regardless of the cluster extraction methods applied to the resultant dendrogram. IK_AHC aims to reveal the root cause of this issue and provide a solution by using a data-dependent kernel. We analyse the condition under which existing AHC methods fail to effectively extract clusters, and give the reason why the data-dependent kernel is an effective remedy. Our extensive empirical evaluation shows that the recently introduced Isolation Kernel produces a higher quality or purer dendrogram than distance, Gaussian Kernel and adaptive Gaussian Kernel in all the above mentioned AHC algorithms. Technical details and analysis of the algorithm can be found in the paper.

Han, X., Zhu, Y., Ting, K. M., and Li, G., “The Impact of Isolation Kernel on Agglomerative Hierarchical Clustering Algorithms]”, <i>Pattern Recognition</i>, 2023. [(pdf)](https://arxiv.org/pdf/2010.05473.pdf)


---
### Citations
---

If you use it for a scientific publication, please include a reference to this paper.

* Xin Han, Ye Zhu, Kai Ming Ting, and Gang Li, [The Impact of Isolation Kernel on Agglomerative Hierarchical Clustering Algorithms](https://arxiv.org/pdf/2010.05473.pdf), <i>Pattern Recognition</i>, 2023.

`BibTex` information:

```bibtex
@article{HZTLThe2020,
  author = {Han, Xin and Zhu, Ye and Ting, Kai Ming and Li, Gang},
  title = {The Impact of Isolation Kernel on Agglomerative Hierarchical Clustering Algorithms},
  journal = {Pattern Recognition},
  year = {2023},
  url = {https://arxiv.org/abs/2010.05473},
}
```


---
###  Requirements
---

* Python 3.8+

---
### Setup
---

install requirements:

```shell
  pip install -r requirements.txt
```

---
### How to use IsoKAHC
---

```python
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import load_wine
from IsoKAHC import IsoKAHC
from utils import metrics

X, y = load_wine()
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

clf = IsoKAHC(n_estimators=200, max_samples=2, method='single')
dendrogram  = clf.fit_transform(X)
den_purity = metrics.dendrogram_purity(dendrogram, y)
```

---
### Notes
---

- Most of the program running time is used to calculate dendrogram purity.

---
### License
---

BSD license
