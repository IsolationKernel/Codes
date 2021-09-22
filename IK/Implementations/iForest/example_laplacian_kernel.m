clear all;
addpath('kernels/laplacian_kernel')
addpath('svm/matlab');
name = 'austra';

foldnum = 5;

load(['data/', name, '.mat']);

%%%Normalization
mx = max(data(:,1:end-1),[], 1);
mn = min(data(:,1:end-1), [], 1);
delta = mx - mn;
delta(delta == 0) = 1;
data(:,1:end-1) = (data(:,1:end-1) - repmat(mn, size(data,1), 1))./repmat(delta, size(data,1),1);

%%% Isolation Kernel Parameters
lambda = 3;   % gamma


i = 1;

tstidx = folds(i,:);
trnidx = tstidx(1:floor(length(tstidx)*fr));
tstidx(1:length(trnidx)) = [];
train_data = data(trnidx,1:end-1);
train_label = data(trnidx,end);
test_data = data(tstidx,1:end-1);
test_label = data(tstidx,end);
tic;
%%% Calculate Kernel
Ktrn = lap_kernel(train_data, train_data, lambda);

%%% Train SVM
K_train = [(1:size(train_data,1))', Ktrn];
[model] = svmtrain(train_label, K_train, ['-t 4']);
trn_time=toc; tic;
   
%%% Calculate Kernel for test data
Ktst = lap_kernel(test_data, train_data, lambda);

%%% SVM Predict
K_test = [(1:size(test_data,1))', Ktst];
[predlb, acc,~] = svmpredict(test_label, K_test, model);
test_time = toc;
disp(acc(1));
