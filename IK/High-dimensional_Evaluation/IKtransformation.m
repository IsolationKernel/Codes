
%% Generating svm file for Brute force vs Ball tree index  
% the best results are shown in Table 2

clear
load('wGaussians.mat')
libsvmwrite('Odata.svm', class(1:size(data,1)), sparse(data)); % Original data

tic
psi=32; % best psi should be tuned for different datasets
t=200;
[ndata] = IKspace (data,data, psi, t);
toc
libsvmwrite('IKdata.svm', class(1:size(ndata,1)), sparse(ndata)); % IK features

% once get the transformed data, please run testIndexing.ipynb to conduct the
% run time comparision 

%% Generating svm file for SVM classifiation



cd classification_datasets % need to enter the folder to process libsvmread

clear  
[data, class] = libsvmread('a9a');


tic
psi=8; % best psi should be tuned for different datasets
t=200;
[ndata] = IKspace (data,data, psi, t);
toc

libsvmwrite('IKa9a8', class, sparse(ndata)); % IK features

