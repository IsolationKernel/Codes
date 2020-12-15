clear all;
addpath('kernels/Random_Forests')
addpath('kernels/Random_Forests/cartree');
addpath('kernels/Random_Forests/cartree/mx_files');
addpath('kernels/Random_Forests/ancillary');

%%%This is RF_DG, if want to use RF_B, please simply replace 
%%% "Random_Forests" above with "Random_Forests_b"

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

%%% RF Kernel Parameters
psi = 256;   % sample size

i = 1;

tstidx = folds(i,:);
trnidx = tstidx(1:floor(length(tstidx)*fr));
tstidx(1:length(trnidx)) = [];
train_data = data(trnidx,1:end-1);
train_label = data(trnidx,end);
test_data = data(tstidx,1:end-1);
test_label = data(tstidx,end);
tic;
%%% Build Frandom Forest
forest = build_forest(train_label, train_data, psi);

%%% Calculate Kernel            
Ktrn = get_sim(train_data, train_data, forest);

%%% Train SVM
K_train = [(1:size(train_data,1))', Ktrn];
[model] = svmtrain(train_label, K_train, '-t 4');
trn_time = toc;

tic;
%%% Calculate Kernel for test data
Ktst = get_sim(test_data, train_data, forest);
K_test = [(1:size(test_data,1))', Ktst];

%%% SVM Predict
[predlb, acc,~] = svmpredict(test_label, K_test, model);
test_time = toc;
disp(acc(1));