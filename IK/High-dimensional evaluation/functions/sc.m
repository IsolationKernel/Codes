function [cluster_labels evd_time kmeans_time total_time] = sc(S, num_clusters)
%SC Spectral clustering using a sparse similarity matrix (t-nearest-neighbor).
%
%   Input  : A              : N-by-N sparse similarity matrix, where
%                             N is the number of data 
%            num_clusters   : number of clusters
%
%   Output : cluster_labels : N-by-1 vector containing cluster labels
%            evd_time       : running time for eigendecomposition
%            kmeans_time    : running time for k-means
%            total_time     : total running time
%
%   Author : Wen-Yen Chen (wychen@alumni.cs.ucsb.edu)
%			 Chih-Jen Lin (cjlin@csie.ntu.edu.tw)

%
% Convert the sparse distance matrix to a sparse similarity matrix,
% where S = exp^(-(A^2 / 2/sigma^2)).
% Note: This step can be ignored if A is sparse similarity matrix.
%
%disp('Converting distance matrix to similarity matrix...');
%tic;
n = size(S, 1);
 

%
% Do laplacian, L = D^(-1/2) * S * D^(-1/2)
%
%disp('Doing Laplacian...');
D = sum(S, 2) + (1e-10);
D = sqrt(1./D); % D^(-1/2)
D = spdiags(D, 0, n, n);
L = D * S * D;
clear D S;
%time1 = toc;

%
% Do eigendecomposition, if L =
%   D^(-1/2) * S * D(-1/2)    : set 'LM' (Largest Magnitude), or
%   I - D^(-1/2) * S * D(-1/2): set 'SM' (Smallest Magnitude).
%
%disp('Performing eigendecomposition...');
OPTS.disp = 0;
OPTS.tol = eps*10^12;%%%
[V, val] = eigs(L, num_clusters, 'LM', OPTS);
%time2 = toc;

%
% Do k-means
%
%disp('Performing kmeans...');
% Normalize each row to be of unit length
sq_sum = sqrt(sum(V.*V, 2)) + 1e-20;
U = V ./ repmat(sq_sum, 1, num_clusters);
clear sq_sum V;
%cluster_labels = k_means(U, 'random', num_clusters);
cluster_labels = kmeans(U,num_clusters);
%total_time = toc;

%
% Calculate and show time statistics
%
%evd_time = time2 - time1
%kmeans_time = total_time - time2
%total_time
%disp('Finished!');
