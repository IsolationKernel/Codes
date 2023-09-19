function [LC] = getLC(Dis,Den,K)
%input:
%Dis: distance matrix (N*N) of a dataset
%Den: density vector (N*1) of the same dataset
%K: K parameter for KNN

%output:
%LC: Local Contrast

N = size(Den,1);
LC = zeros(N,1);
for i = 1:N
    [~,inx] = sort(Dis(i,:));
    knn = inx(2:K+1); % K-nearest-neighbors of instance i
    LC(i) = sum(Den(i) > Den(knn));    
end
end

