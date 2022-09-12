# StreaKHC

---

| Field | Value |
| --- | --- |
| Title | StreaKHC  |
| Type | Source Code |
| Language | Python, Shell |
| License | BSD license  |
| Status | Research Code |
| Update Frequency | NO |
| Date Published | 2022-05-20  |
| Date Updated |  2022-05-20 |
| Portal | https://github.com/tulip-lab/open-code |
| URL | https://github.com/tulip-lab/open-code/tree/master/StreaKHC|
| Publisher |[TULIP Lab](http://www.tulip.org.au/) |
| Point of Contact |[A/Prof. Gang Li](https://github.com/tuliplab) |

**StreaKHC**  is a novel incremental hierarchical clustering algorithm for efficiently mining massive streaming data. It uses a scalable point-set kernel to measurethe similarity between an existing cluster in the cluster tree and a new point in a stream. It also has an efficient hierarchical structure updating mechanism to continuously maintain a high-quality cluster tree in real-time. Technical details and analysis of the algorithm can be found in the paper.

---
### Citations
---

If you use it for a scientific publication, please include a reference to this paper.

* Xin Han, Ye Zhu, Kai Ming Ting, De-Chuan Zhan and Gang Li. *Streaming Hierarchical Clustering based on Point-Set Kernel.* In Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery & Data Mining (KDD '22). 2022. https://doi.org/10.1145/3534678.3539323
  
`BibTex` information:

```bibtex
@inproceedings{HZTZL22STREAMING,
  author = {Han, Xin and Zhu, Ye and Ting, Kai Ming and Zhan De-Chuan and Li, Gang},
  title = {Streaming Hierarchical Clustering based on Point-Set Kernel},
  publisher = {Association for Computing Machinery},
  booktitle = {Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery & Data Mining},
  series = {KDD '22}
  year={2020}
}
```

---
## Setup ##
---

Download and Install Anaconda's Python3

```
https://docs.continuum.io/anaconda/install
```

Install numba

```
conda install numba
```

Set environment variables:

```
source bin/setup.sh
```

If want to visulize the build tree, install Graphviz

```
sudo apt install graphviz
```

---
## Run test ##
---

Run test on data set:
```
 ./bin/run_grid_evaluation.sh
```

The evaluation result is shown in /exp_out/ default. For each of the randomly shuffled data of a specified data set, the dengrogram purity result and figure of built tree is shown in score.tsv and tree.png, respectively.

---
## Notes ##
---
  - If do not need to visualize the generated tree, you can comment out the corresponding code in the /bin/run_evaluation.sh.
  - Perl is used to shuffle the data.You'll need perl installed on your system to run experiment shell scripts.  If you can't run perl, you can change this to another shuffling method of your choice.
  - The scripts in this project use environment variables set in the setup script. You'll need to source this set up script in each shell session running this project.
  - Most of the program running time is used to calculate dendrogram purity.


---
## License ##
---
BSD license
