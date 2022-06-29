function [ndata,U,D] = sNstr(data,k,sig)
% Large-Scale Nyström Approximation: rank-k nystrom approximation of the PSD matrix G such
% that G ~= U*D*U'. 

%  k is the rank 
%  sig is the bandwidth of Gaussian kernel

%  Mu Li, James Kwok, and Bao-liang Lu. Making Large-Scale Nyström Approximation Possible.
%  Proceedings of the Twenty-Seventh International Conference on Machine Learning (ICML),
%  Haifa, Israel, June 2010

m=k;
n = size(data,1);
p=5;
q=2;

idx = randperm(n);   

cols = idx(1:m);    
  
m_dis=pdist2(data,data(cols,:));
C = exp(-0.5*(m_dis.^2)./sig^2);

W = C(cols,:);    

%%%% perform truncated SVD on m-by-m matrix W

[V,D] = rsvd(W,k,p,q);

%%%% form the approximation


U = C * ( sqrt(m/n) * V );
D = (n/m) * diag(diag(D).^-1);

ndata = U*chol(D)'; % feature map

end

%%

% m_dis=pdist2(data,data);
% G = exp(-0.5*(m_dis.^2)./sig^2);
% 
%  sum(sum(U*D*U'-G))
%  sum(sum(ndata*ndata'-G))