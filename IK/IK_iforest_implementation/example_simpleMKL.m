clear all;
addpath('svm/matlab');
addpath('simplemkl');
name = 'austra';

load(['data/', name, '.mat']);

%%%Normalization
mx = max(data(:,1:end-1),[], 1);
mn = min(data(:,1:end-1), [], 1);
delta = mx - mn;
delta(delta == 0) = 1;
data(:,1:end-1) = (data(:,1:end-1) - repmat(mn, size(data,1), 1))./repmat(delta, size(data,1),1);


i = 1;

tstidx = folds(i,:);
trnidx = tstidx(1:floor(length(tstidx)*fr));
tstidx(1:length(trnidx)) = [];
train_data = data(trnidx,1:end-1);
train_label = data(trnidx,end);
test_data = data(tstidx,1:end-1);
test_label = data(tstidx,end);

tic;
[model,Ktrn] = mkltrain(train_label, train_data);
trn_time = toc;

tic;
[~,acc,~] = mklpredict(test_label, test_data, model);
test_time = toc;
disp(acc(1));
 