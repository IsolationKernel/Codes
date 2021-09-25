function [ mass, proximity ] = UaNNE (data, psi, t)
[n,d] = size(data);
proximity = zeros(n);

for i=1:t
    ndata=rand(psi,d);
    dis=pdist2(ndata,data);
    [~, centerIdx] = min(dis);
    proximity = proximity + (centerIdx' == centerIdx);
end

proximity = proximity/t;
mass = sum(proximity);
end
