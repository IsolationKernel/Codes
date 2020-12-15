% An example of the mahalanobis distance learning using GMML

load('data/wine.mat')

%setting the parameters of the classifier
num_folds = 5;
knn_neighbor_size = 5;

%setting the parameters of the metric learning method
params.const_factor = 40;
params.tuning_num_fold = 5;

%performing the metric learning and tuning the parameter t:
err = CrossValidateKNN(y, X, @(y,X) MetricLearningAutotuneKnn ...
    (y, X, knn_neighbor_size, params), num_folds, knn_neighbor_size);

%performing the metric learning when the parameter t has been already choosen:
% t = 0.05;
%  err = CrossValidateKNN(y, X, @(y,X) MetricLearning(y, X, t, params), ...
%      num_folds, knn_neighbor_size);

disp(['The accuracy of k-NN is ',num2str(100*(1 - mean(err)))]);