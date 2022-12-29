function [ Pscore ] = point_score( Y,psi, window )
% calculate each point dissimilarity score
% input should be data, psi value and window size


Y = (Y - min(Y)).*((max(Y) - min(Y)).^-1);  % normalisation
Y(isnan(Y)) = 0.5;

type='NormalisedKernel';

Sdata=Y;
data=Y;

t=200;

[ndata] = aNNEspace(Sdata,data, psi, t);

%index each segmentation
index=1:window:length(Y);
if index(end)~=length(Y)
    index=[index length(Y)];
end

%kernel mean embedding
mdata=[];
for i=1:length(index)-1
    cdata=ndata(index(i):index(i+1),:);
    mdata(i,:)=mean(cdata,1);
end

k=1; % knn

score=[];
switch type
    case 'NormalisedKernel'
        for i=k+1:size(mdata,1)
            Cscore=[];
            for j=1:k
                Cscore(j)=mdata(i,:)*mdata(i-j,:)'/((mdata(i,:)*mdata(i,:)')^0.5*(mdata(i-j,:)*mdata(i-j,:)')^0.5); % normalised inner product
            end
            score(i)=1-mean(Cscore);
        end
    case 'MMD'
        for i=k+1:size(mdata,1)
            score(i)= mean(pdist2(mdata(i,:),mdata(i-k:i-1,:))); % MMD (euclidean distance)
        end
end


%% assign score to segmentation
Pscore=zeros(size(Y,1),1);
for i=1:length(index)-1
    Pscore(index(i):index(i+1))=score(i);
end
Pscore = (Pscore - min(Pscore)).*((max(Pscore) - min(Pscore)).^-1);  % normalisation
end