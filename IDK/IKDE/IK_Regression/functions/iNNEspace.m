function [ndata] = iNNEspace(Sdata,data, psi, t)
% Sdata is used for subsampling 
% psi is the subsample size
% t is the sampling times

% ndata are the Isolation kernel features for instances in data 

[sn,~]=size(Sdata);
[n,~]=size(data);

ndata=[];
c=0:psi+1:(n-1)*(psi+1);

for i = 1:t
    % sampling    
    CurtIndex = datasample(1:sn, psi, 'Replace', false);
    Ndata = Sdata(CurtIndex,:);
    
    % filter out repeat
    
    [~,IA,~] = unique(Ndata,'rows'); 
    NCurtIndex=CurtIndex(IA);
    Ndata = Sdata(NCurtIndex,:);
    
    % radius
    [D,~] = pdist2(Ndata,Ndata,'minkowski',2,'Smallest',2);    
    R=D(2,:);  % radius
    
    % identify 1NN ball for each point    
    [D,I] = pdist2(Ndata,data,'minkowski',2,'Smallest',1);
    I(D>R(I))=psi+1; % outside ball
    
    z=zeros(psi+1,n);
    z(I+c)=1;
    z(psi+1,:)=[]; % get rid of values that outside ball
    ndata=[ndata sparse(z)'];    
end
end
