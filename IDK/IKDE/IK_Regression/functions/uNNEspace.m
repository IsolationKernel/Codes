function [ndata] = uNNEspace (Sdata,data, psi, t)
% kernel space for data
% Sdata is used for partition
[sn,~]=size(Sdata);
[n,d]=size(data);
ndata=[];
c=0:psi:(n-1)*psi;
for i = 1:t    
    tdata=rand(psi,d);
    dis=pdist2(tdata,data);
    [~, centerIdx] = min(dis);
    z=zeros(psi,n);
    z(centerIdx+c)=1;
    ndata=[ndata z'];
end
end