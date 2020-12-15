function [ndata] = SIKspace (Sdata,data, psi, t)
% kernel space for data
% Sdata is used for partition
%核矩阵
[sn,~]=size(Sdata);   
[n,d]=size(data);   
IDX=[]; 

for i = 1:t 
     subIndex = datasample(1:sn, psi, 'Replace', false); %无放回随机采样 %psi表示采样个数
     tdata=Sdata(subIndex,:);    
%     
%    tdata=rand(psi,d);
    
    dis=pdist2(tdata,data);    %returns the distance between each pair of observations in X and Y using the metric specified by Distance.
                                %默认计算欧几里得距离
    [~, centerIdx] = min(dis);  %centeridx得到的是里data中每个sample最近的tdata中的点的索引
    IDX=[IDX; centerIdx+(i-1)*psi]; 
end 
 
IDR = repmat(1:n,t,1);
V=IDR-IDR+1; %全1数组 t行n列
ndata = sparse(IDR(:)',IDX(:)',V(:)',n,t*psi);

end
 