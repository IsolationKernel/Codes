function [ndata] = GKfeature(data,t,sigma)
%GKFEATURE Summary of this function goes here
%   Detailed explanation goes here
n=size(data,1);
% define function for computing kernel dot product 

kFunc = @(data,rowInd,colInd) gaussianKernel(data,rowInd,colInd,sigma);

%K = gaussianKernel(data,1:n,1:n,sigma); % kernel similarity matrix

[C,W] = recursiveNystrom(data,min(n,t),kFunc);

%nK=C*W*C'; % approximates data's full kernel matrix, K.%sum(sum(nK-K))

ndata = C*chol(W)'; % GK feature map

end

