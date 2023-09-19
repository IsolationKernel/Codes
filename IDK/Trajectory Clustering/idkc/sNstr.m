function [U,D] = sNstr(data,k,sig)
%SNSTR 此处显示有关此函数的摘要
%   此处显示详细说明
 
m=k;
n = size(data,1);
p=5;
q=2;

idx = randperm(n);   

cols = idx(1:m);    
  
m_dis=pdist2(data,data(cols,:));
C = exp(-0.5*(m_dis.^2)./(2*sig^2));

W = C(cols,:);    

%%%% perform truncated SVD on m-by-m matrix W

[V,D] = rsvd(W,k,p,q);

%%%% form the approximation


U = C * ( sqrt(m/n) * V );
D = (n/m) * diag(diag(D).^-1);


end

