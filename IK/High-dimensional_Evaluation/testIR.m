% the retrieval precision of 5 nearestneighbour as shown in Table 2

clear
load('wGaussians.mat')
k=5;
  
% data=gpuArray(double(data));

% normalisation 
 data = (data - min(data)).*((max(data) - min(data)).^-1);
 data(isnan(data)) = 0.5;
 
%% retrieval using distance

DisMatrix=pdist2(data,data);  
 
P=[];
parfor i=1:size(data,1)
    cD=DisMatrix(i,:);
    [~,id]=sort(cD);
    cC=class(id(2:k+1));
    P(i)=sum(cC-class(i)==0)/k;
end
 
Dist_precision=mean(P)

%% retrieval using IK

 
t=200;
IKprecision=[];
parfor ii=1:10
    psi=2^ii;    
    [ndata] = SIKspace (data,data, psi, t);     
    IKDisMatrix=pdist2(ndata,ndata);  
    P=[];    
    for i=1:size(data,1)
        cD=IKDisMatrix(i,:);
        [~,id]=sort(cD);
        cC=class(id(2:k+1));
        P(i)=sum(cC-class(i)==0)/k;
    end
    IKprecision(ii)=mean(P);
end
 
BIKprecision=max(IKprecision)