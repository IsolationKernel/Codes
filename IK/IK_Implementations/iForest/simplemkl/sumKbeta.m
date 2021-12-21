function Kaux=sumKbeta(K,beta)

% Usage
%  Kaux=sumKbeta(K,beta)
%
%  K is usually a 3D matrix where K(:,:,i) is 
%  the K_i gram matrix 
%
%   K can also be a (n*(n-1)/2) x nbkernel matrix build
%   by build_efficientkernel and is a struct

if ~isstruct(K)
    ind=find(beta);
    nbkernel=size(K,3);
    Kaux=zeros(size(K(:,:,1)));
    N=length(ind);
    for j=1:N
        Kaux=Kaux+ beta(ind(j))*K(:,:,ind(j));
    end
else
    if size(beta,1)>1;
        beta=beta';
    end;
if isa(K.data,'single');
    Kaux=devectorize_single(K.data*beta');
else
    Kaux=devectorize(K.data*beta');
end;
end;