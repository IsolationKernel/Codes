function dis = Linear1(data)
[n,m] = size(data);
Li = zeros(n,n);
for iIndex = 1:n
    for jIndex = 1:n
        for eIndex = 1:m
            Li(iIndex,jIndex) = Li(iIndex,jIndex) + data(iIndex,eIndex)*data(jIndex,eIndex);
        end    
    end
end
Li = Li./max(max(Li));
dis = 1-Li;

end