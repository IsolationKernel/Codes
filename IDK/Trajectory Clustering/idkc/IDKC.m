function [Tclass,Centre,GP,it,OTclass,tr] =IDKC(ndata,k,Kn,v,s,ID)
% Input
% ndata is the kernel feature
% k is the number of clusters
% Kn is the kNN size
% v is the learning rate
% s is the sample size for mode seletcion

% Output
% Tclass is the cluster labels
% it is the iteration times
t1=tic;
C={};
dID=1:size(ndata,1);
GP={};
D=[dID' ndata]; % add index in the first column


if nargin < 6
    %find modes based on sample
    sID = randperm(size(ndata,1),s);
    [ID] = FindMode(ndata(sID,:),k,Kn);
    ID=sID(ID);
end


%[ID] = FindMode(ndata,k,Kn);

Centre{1}=ID';
GP{1}=[ID' [1:k]'];
%% initialing clusters

L=size(ndata,2);

Csum=zeros(k,L);
Csize=zeros(k,1);
for i=1:k
    C{i}=D(ID(i),:);
    Csum(i,:)=sum(C{i}(:,2:end),1);
    Csize(i,1)=size(C{i}(:,2:end),1);
end
D(ID,:)=[];
it=1;

Cmean=Csum./repmat(Csize,1,L);

[S,T]=max(D(:,2:end)*Cmean',[],2);

r=max(S);

%% linking points

while size(D,1)>0
    
    Cmean=Csum./repmat(Csize,1,L);
    
    [S,T]=max(D(:,2:end)*Cmean',[],2);
    
    r=v*r;
    
    if  sum(S)==0  || r<0.000001
        %   disp('break')
        break
    end
    
    it=it+1;
    
    DI=T-T;
    for i=1:k
        I=(T==i & S>r);
        %   sum(I)
        if sum(I)>0
            C{i}=[ C{i}; D(I,:)];
            Csum(i,:)= Csum(i,:)+sum(D(I,2:end),1);
            Csize(i,1)= Csize(i,1)+sum(I);
            DI=DI+I;
        end
    end
    
    
    %   identify centres 
    
    Centre{it}=[];
    GP{it}=[];
    for jj=1:k
        CD=C{jj};
        [~,x]=max(CD(:,2:end)*sum(CD(:,2:end),1)');
        Centre{it}=[Centre{it}; CD(x(1),1)];        
        GP{it}=[GP{it};[CD(:,1) zeros(size(CD,1),1)+jj]];
    end
    
    
    D(DI>0,:)=[];
end


Tclass=zeros(size(ndata,1),1);


for i=1:size(C,2)
    Tclass(C{i}(:,1))=i;
end


%% postprocessing

t2=tic;

Th=ceil(size(ndata,1)*0.01);
OTclass=Tclass;

for iter=1:100
    Cmean=[];
    for i=1:k
        Cmean=[Cmean;mean(ndata(Tclass==i,:),1)];
    end
    [~,Tclass2]=max(ndata*Cmean',[],2);
    
    if sum(Tclass2~=Tclass)<Th || length(unique(Tclass2))<k
        break
    end
    Tclass=Tclass2;
end

% update centres
Centre{it+1}=[];

for i=1:k
    I=find(Tclass==i);
    CD=ndata(I,:);
    [~,x]=max(CD*sum(CD,1)');
    Centre{it+1}= [Centre{it+1};  I(x(1))];
end
tr=toc(t2)/(toc(t1));
GP{it+1}=[[1:size(ndata,1)]' Tclass];
%
end

