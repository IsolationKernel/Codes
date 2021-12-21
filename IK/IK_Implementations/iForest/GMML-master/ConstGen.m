function [S, D] = ConstGen(X, Y, num_const)
% This function generates some pairs for the similarity and dissimilarity 
% sets and computes the matrices S and D.

n = size(X,1);

k1 = randi(n,[num_const,1]);
k2 = randi(n,[num_const,1]);

ss = (Y(k1) == Y(k2));
dd = not(ss);  % dd = (Y(k1) ~= Y(k2));

S = lowmemsub(X',k1(ss),k2(ss), 1);
D = lowmemsub(X',k1(dd),k2(dd), 10);

% An another way for computing the matrices S and D:
% SD = X(k1(ss),:) - X(k2(ss),:);
% DD = X(k1(dd),:) - X(k2(dd),:);
% S = SD'*SD;
% D = DD'*DD;
end