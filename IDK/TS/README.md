
# IDK-based anomaly detection method for time series.


IDK $^2$ for anomalous subsequence detection using non-overlapping windows and a sliding-window variant
s-IDK $^2$ are implemented in Python and can be found in **IDK_T.py** and **IDK_square_sliding.py**, respectively. 

An example of applying IDK-based methods for anomalous subsequence detection on the time series noisy_sine is provided in **main.py**. Please unzip the file Discords_Data.zip and run **main.py** for the result.

## Requirements
- numpy
- pandas
- scikit-learn

## Example
```python
# Read data. 
# The period length (cycle) of each time series and the locations of anomalous period subsequences are given in the appendix of the paper.
X=np.array(pd.read_csv("Discords_Data/noisy_sine.txt",header=None)).reshape(-1,1)
cycle=300


# IDK square using non-overlapping windows. The input series should have single or multiple periodicity.
# The period length (cycle) of the time series needs to be input. 
# The number of partitionings t is 100 by default. The sample sizes psi1 and psi2 for two levels of IK mappings need to be searched
# and the search range we use is provided in the paper.
# Return the similarity score for each period of time series. Anomalous period subsequences are those having lowest similaity scores.
similarity_score=IDK_T(X,t=100,psi1=16,width=cycle,psi2=2)


# IDK square using a sliding window. The input series can be periodic or aperiodic time series showing recurring normal subsequences.
# The length of the sliding window needs to be searched and the search range we use is provided in the paper.
#Return the similarity score for each subsequence of time series extracted by a sliding window. Anomalous subsequences are those having lowest similaity scores.
w=cycle-50
sliding_score=IDK_square_sliding(X,t=100,psi1=4,width=w,psi2=4)
```

## Reference
Kai Ming Ting, Zongyou Liu, Hang Zhang and Ye Zhu. A New Distributional Treatment for Time Series and An Anomaly Detection Investigation. PVLDB, 15(11).
