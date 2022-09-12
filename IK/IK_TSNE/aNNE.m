function [ mass, proximity ] = aNNE (dis, psi, t)
n = size(dis, 1);
proximity = zeros(n);

for i = 1:t
    
    subIndex = datasample(1:n, ceil(psi), 'Replace', false);        
    [~, centerIdx] = min(dis(subIndex, :));
    %centerIdx = subIndex(centerIdx);    
    proximity = proximity + (centerIdx' == centerIdx);
end
proximity = proximity/t;
mass = sum(proximity);

end
 