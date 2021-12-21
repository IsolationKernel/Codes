addpath('svm/matlab');
addpath('GMML-master');
name = 'austra';

load(['data/', name, '.mat']);

%%%Normalization
mx = max(data(:,1:end-1),[], 1);
mn = min(data(:,1:end-1), [], 1);
delta = mx - mn;
delta(delta == 0) = 1;
data(:,1:end-1) = (data(:,1:end-1) - repmat(mn, size(data,1), 1))./repmat(delta, size(data,1),1);

%%% GMML parameters
t = 0.1;  %[0.1:0.1:0.9]
gamma = 2.^[-3];  %2.^[-10:5]

i=1;

trnidx = folds(i,:);
tstidx = trnidx(floor(fr*length(trnidx))+1:end);
trnidx(floor(fr*length(trnidx))+1:end) = [];

trn = data(trnidx,:);
tst = data(tstidx,:);

tic;
%%% GMML learning metric
A = MetricLearning(trn(:,end), trn(:,1:end-1),t);
%%% SVM training based on trasformed data
model = svmtrain(trn(:,end), trn(:,1:end-1)*A,['-g ',num2str(gamma)]);
trn_time=toc;

tic;
%%% SVM testing based on trasformed data
[pred,acc2,~] = svmpredict(tst(:,end), tst(:,1:end-1)*A, model);
tst_time=toc;
disp(acc2(1));
    