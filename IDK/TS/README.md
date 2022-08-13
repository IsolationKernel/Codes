
# IDK-based anomaly detection method for time series.


IDK$^2$ for anomalous subsequence detection using non-overlapping windows and a sliding-window variant
s-IDK$^2$ are implemented in Python and can be found in **IDK_T.py** and **IDK_square_sliding.py**, respectively. 

An example of applying IDK-based methods for anomalous subsequence detection on the time series noisy_sine is provided in **main.py**. Please unzip the file Discords_Data.zip and run **main.py** for the result.

## Requirements
- numpy
- pandas
- scikit-learn

## Example
This site provides a
```
# Read data
X=np.array(pd.read_csv("Discords_Data/noisy_sine.txt",header=None)).reshape(-1,1)
cycle=300
# IDK square using non-overlapping windows
# The period(cycle) length of the time series needs to be input. The number of partitionings t is 100 by default. The sample size psi1 and psi2 for two levels of IK mappings need to be search and the search range we used is provided in the paper.

similarity_score=IDK_T(X,t=100,psi1=16,width=cycle,psi2=2)
# IDK square using a sliding window
w=cycle-50
sliding_score=IDK_square_sliding(X,t=100,psi1=4,width=w,psi2=4)
```

## Reference
Kai Ming Ting, Zongyou Liu, Hang Zhang and Ye Zhu. A New Distributional Treatment for Time Series and An Anomaly Detection Investigation. PVLDB, 15(11).
