function [ndata] = SIKspace (Sdata,data, psi, t)
% kernel space for data
% Sdata is used for partition
[sn,~]=size(Sdata);   
[n,d]=size(data);   
IDX=[]; 

for i = 1:t 
     subIndex = datasample(1:sn, psi, 'Replace', false);
      tdata=Sdata(subIndex,:);    
%     
%    tdata=rand(psi,d);
    
    dis=pdist2(tdata,data);    
    [~, centerIdx] = min(dis);
    IDX=[IDX; centerIdx+(i-1)*psi]; 
end 
 
IDR = repmat(1:n,t,1);
V=IDR-IDR+1; 
ndata = sparse(IDR(:)',IDX(:)',V(:)',n,t*psi);

end
 