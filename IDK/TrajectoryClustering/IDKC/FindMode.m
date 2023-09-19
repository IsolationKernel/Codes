function [ID] =  FindMode(ndata,k,Kn)
%FINDMODE 此处显示有关此函数的摘要
%   此处显示详细说明

Density=ndata*mean(ndata,1)';
IKDist=pdist2(ndata,ndata);

Density=gather(Density);
IKDist=gather(IKDist);


[Density] = getLC(IKDist,Density,Kn);
%
% figure
% scatter3(data(:,1),data(:,2),Density,10,Density,'filled')
% colorbar
%%
maxd=max(max(IKDist)); % max dis
NumIns=size(IKDist,2);
MinDist=Density-Density;
[~,SortDensity]=sort(Density,'descend'); % SortDensity is index
MinDist(SortDensity(1))=-1.;
nneigh(SortDensity(1))=0;

for ii=2:NumIns
    MinDist(SortDensity(ii))=maxd;
    for jj=1:ii-1
        if(IKDist(SortDensity(ii),SortDensity(jj))<MinDist(SortDensity(ii)))
            MinDist(SortDensity(ii))=IKDist(SortDensity(ii),SortDensity(jj));
            nneigh(SortDensity(ii))=SortDensity(jj); % nearest neigbour index
        end
    end
end
MinDist(SortDensity(1))=max(MinDist(:));

%
% figure
% scatter3(data(:,1),data(:,2),MinDist,10,MinDist,'filled')
% colorbar

%% normalise
%    Density=normalize(Density)+0.0000000001;
%     MinDist=normalize(MinDist)+0.0000000001;

Density=tiedrank(Density)+0.0000000001;
MinDist=tiedrank(MinDist)+0.0000000001;

Mult=(Density).*(MinDist);
[VSortMult,ISortMult]=sort(Mult,'descend');

ID=ISortMult(1:k); 
end

