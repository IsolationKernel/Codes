function [ Z ] = OHierDP( SimMatrix,dc)
% original DP clustering algorithm with hierarchical outputs
% Zhu, Y., Ting, K.M., Jin, Y. and Angelova, M., 2021, August. Hierarchical clustering that takes advantage of both density-peak and density-connectivity. Information Systems.

NumIns=size(SimMatrix,2);
Density=sum(SimMatrix'<=dc);

NeigDisList={};
NeigDisList{NumIns}=[];

NDL=zeros(1,NumIns);

NeigInsList={};
NeigInsList{NumIns}=[];

%% link to higher density points

[~,SortDensity]=sort(Density,'descend');

for j=2:NumIns
    i=SortDensity(j);
    index=SortDensity(1:j-1);
    [a, b]=min(SimMatrix(i,index));
    if ~isempty(a)
        NeigDisList{i}=a;
        NeigInsList{i}=index(b);
        NDL(i)=NeigDisList{i}(1);
    end
end

NDL(SortDensity(1))=max(NDL(:));
%% Normalise Density and NDL

Density=normalize(Density')+0.0000001;
NDL=normalize(NDL')+0.0000001;

Mult=(Density').*(NDL');


%% building tree

Z=[];
CanList=[];

Nindex=NumIns+1; % point label in tree
M=max(max(SimMatrix));
NLabel=1:NumIns;
scoreList=[Mult; NLabel]'; % combine Mult with point real index

while (~isempty(scoreList))
    I=scoreList(:,1);
    [a, ii]=min(I);
    b=scoreList(ii,2); % find point real index
    if isempty(NeigInsList{b})
        CanList=[CanList b];
        scoreList(ii,:)=[];
    else
        Nscore=NeigDisList{b}(1)/M*Density(b);
        nZ=[NLabel(b) NLabel(NeigInsList{b}(1)) Nscore];
        NLabel(NLabel==NLabel(b))=Nindex;
        NLabel(NLabel==NLabel(NeigInsList{b}(1)))=Nindex;
        Nindex=Nindex+1;
        scoreList(ii,:)=[];
        Z=[Z;nZ];
    end
end

M=max(Z(:,3));

for i=1:length(CanList)-1
    nZ=[NLabel(CanList(i)) NLabel(CanList(i+1)) M*1.1];
    NLabel(NLabel==NLabel(CanList(i)))=Nindex;
    NLabel(NLabel==NLabel(CanList(i+1)))=Nindex;
    Nindex=Nindex+1;
    Z=[Z;nZ];
end


end