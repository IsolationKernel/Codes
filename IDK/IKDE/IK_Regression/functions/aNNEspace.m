function [ndata] = aNNEspace (Sdata,data, psi, t)
% kernel space for data
% Sdata is used for partition
[sn,~]=size(Sdata);
[n,d]=size(data);
ndata=[];
c=0:psi:(n-1)*psi;
for i = 1:t
    subIndex = datasample(1:sn, psi, 'Replace', false);
    tdata=Sdata(subIndex,:);
    dis=pdist2(tdata,data);
    [~, centerIdx] = min(dis);
    z=zeros(psi,n);
    z(centerIdx+c)=1;
    ndata=[ndata z'];
end
end