function [acc, pred] = CrossValidateKNN(y, X, tCL, k, knn_size)
% Cross-validation for evaluating the k-nearest neighbor classifier with
% a learned metric.  Performs k-fold cross validation, training on the
% training fold and evaluating on the test fold.

% This function is taken from the source code of:
% Jason V. Davis, Brian Kulis, Prateek Jain, Suvrit Sra, and Inderjit
% S. Dhillon.  "Information-theoretic Metric Learning."  Proc. 24th
% International Conference on Machine Learning (ICML), 2007.

[n,m] = size(X);
if (n ~= length(y)),
   disp('ERROR: num rows of X must equal length of y');
   return;
end

%permute the rows of X and y
rp = randperm(n);
y = y(rp);
X = X(rp, :);

pred = zeros(n,1);
for (i=1:1),
   test_start = ceil(n/k * (i-1)) + 1;
   test_end = ceil(n/k * i);

   yt = [];
   Xt = zeros(0, m);
   if (i > 1);
       yt = y(1:test_start-1);
       Xt = X(1:test_start-1,:);
   end
   if (i < k),
       yt = [yt; y(test_end+1:length(y))];
       Xt = [Xt; X(test_end+1:length(y), :)];
   end
   
   nt = length(yt);
   yt = yt(1:nt);
   Xt = Xt(1:nt, :);

   %train model
   M = feval(tCL, yt, Xt);

   %evaluate model 
   XT = X(test_start:test_end, :);

   pred(test_start:test_end) = inKNN(yt, Xt, sqrtm(M), knn_size, XT);
end
acc = sum(pred(test_start:test_end)==y(test_start:test_end))/length(test_start:test_end);
end