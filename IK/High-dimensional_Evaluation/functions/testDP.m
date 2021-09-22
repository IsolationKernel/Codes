function [bf,ba]=testDP(data,class,SimMatrix)

%% DP

% SimMatrix=pdist2(data,data,'minkowski',2);
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


for i=length(unique(class))
    parfor j=2:P
        [ Tclass, ~] = DensityPeak( SimMatrix,range(j),i);
        if max(Tclass)<200
             TT(i,j)= fmeasure(class,Tclass); 
             AA(i,j)= ami(class',Tclass');
        end       
    end
end
[a,b]=find(TT==max(max(TT)));
bk =a(end);
be =range(b(end));
bf =max(max(TT));
ba =max(max(AA));
% %%
% parfor i=1:P
%     dc=range(i);
%     %     A=Mdbscan(data,2,eps,SimMatrix);
%     %     if max(A)<200
%     [ Z ] = OHierDP( SimMatrix,dc);
%     for k=1:20
%         Tclass = cluster(Z,'maxclust',k);
%         [~,~,~,NmiScore(i,k),~,~,~,~,~,~] = evaluate(class,Tclass);
%         para{i,k}=[dc k];
%         %         end
%     end
% end
% 
% 
% [a,b]=find(NmiScore==max(max(NmiScore)));
% bk =b(end);
% be =range(a(end));
% bf =max(max(NmiScore));
% 
%  