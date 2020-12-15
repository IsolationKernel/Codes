addpath('svm/matlab');
name = 'austra';

load(['data/', name, '.mat']);

%%%Normalization
mx = max(data(:,1:end-1),[], 1);
mn = min(data(:,1:end-1), [], 1);
delta = mx - mn;
delta(delta == 0) = 1;
data(:,1:end-1) = (data(:,1:end-1) - repmat(mn, size(data,1), 1))./repmat(delta, size(data,1),1);

gamma = 2.^[0];  %2.^[-10:5]

i=1;

trnidx = folds(i,:);
tstidx = trnidx(floor(fr*length(trnidx))+1:end);
trnidx(floor(fr*length(trnidx))+1:end) = [];

trn = data(trnidx,:);
tst = data(tstidx,:);

tic;
%%% SVM training
model = svmtrain(trn(:,end), trn(:,1:end-1),['-g ',num2str(gamma)]);
trn_time=toc;

tic;
%%% SVM testing
[pred,acc2,~] = svmpredict(tst(:,end), tst(:,1:end-1), model);
tst_time=toc;
disp(acc2(1));
    
