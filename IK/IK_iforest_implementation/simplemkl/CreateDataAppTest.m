function  [xapp,yapp,xtest,ytest,indice]=CreateDataAppTest(x,y,nbtrain, classcode)

% [xapp,yapp,xtest,ytest,indice]=CreateDataAppTest(x,y,nbtrain, classcode)
%
% if nbtrain =[nbapppos nbappneg ] % we have a specific number of positive
% and negative examples.

if nargin <4
    classcode(1)=1;
    classcode(2)=-1;
end;

if length(nbtrain)==1;
    xapp=[];
    yapp=[];
    xtest=[];
    ytest=[];
    indice=[];
    indice.app=[];
    indice.test=[];
    nbclass=length(classcode);
    nbdata=length(y);
    %keyboard
    for i=1:nbclass;
        ind=find(y==classcode(i));
        nbclasscode_i=length(ind);
        ratioclasscode_i=nbclasscode_i/nbdata;
        aux=randperm(nbclasscode_i);
        nbtrainclasscode_i=round(ratioclasscode_i*nbtrain);
        indapp=ind(aux(1:nbtrainclasscode_i));
        indtest=ind(aux(nbtrainclasscode_i+1:end));
        xapp=[xapp;x(indapp,:)];
        yapp=[yapp;y(indapp,:)];
        xtest=[xtest;x(indtest,:)];
        ytest=[ytest;y(indtest,:)];
        indice.app=[indice.app;indapp];
        indice.test=[indice.test;indtest];
    end;
end;

if length(nbtrain)==2;
    nbapppos=nbtrain(1);
    nbappneg=nbtrain(2);
    indpos=find(y==1);
    indneg=find(y==-1);
    nbpos=length(indpos);
    nbneg=length(indneg);
    auxpos=randperm(nbpos);
    auxneg=randperm(nbneg);
    indapp=[indpos(auxpos(1:nbapppos)) ;indneg(auxneg(1:nbappneg))];
    indtest=[ indpos(auxpos(nbapppos+1:end)) ; indneg(auxneg(nbappneg+1:end))];
    xapp=x(indapp,:);
    yapp=y(indapp);
    xtest=x(indtest,:);
    ytest=y(indtest,:);
end;