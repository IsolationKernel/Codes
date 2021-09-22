function [bf,ba] = testDP(data,class,SimMatrix)


SimMatrix=SimMatrix./(max(max(SimMatrix)));
SimMatrix=roundn(SimMatrix,-2);
range=unique(SimMatrix);
c=[];
for i=1:size(data,1)
    a=SimMatrix(i,:);
    b=sort(a);
    c=[c b(ceil(0.99*size(data,1)))];
end
thre=min(c);
P=sum(range<thre);
TT=zeros(20,P);
NmiScore=[];
k=length(unique(class));
AA=[];
TT=[];
parfor i=1:P
    dc=range(i);
    [ Z ] = OHierDP( SimMatrix,dc);
    Tclass = cluster(Z,'maxclust',k);
    TT(i)= fmeasure(class,Tclass);
    AA(i)= ami(class',Tclass');
    
end
bf =max(max(TT));
ba =max(max(AA));


end

