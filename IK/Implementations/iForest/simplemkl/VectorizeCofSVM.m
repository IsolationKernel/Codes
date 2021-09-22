function Cv=VectorizeCofSVM(yapp,Cplus,Cminus)
% Cv=VectorizeCofSVM(yapp,Cplus,Cminus) 
% n=length(yapp);
% if nargin <3
%     indpos=find(yapp==1);
%     indneg=find(yapp==-1);
%     Cv=zeros(size(yapp));
%     Cv(indpos)=Cplus*length(indpos)/n;
%     Cv(indneg)=Cplus*length(indneg)/n;
% else
%     Cv=zeros(size(yapp));
%     Cv(indpos)=Cplus;
%     Cv(indneg)=Cminus;
% end;

n=length(yapp);
if nargin <3
    indpos=find(yapp==1);
    indneg=find(yapp==-1);
    Cv=zeros(size(yapp));
    Cv(indpos)=Cplus*length(indneg)/n;
    Cv(indneg)=Cplus*length(indpos)/n;
else
    Cv=zeros(size(yapp));
    Cv(indpos)=Cplus;
    Cv(indneg)=Cminus;
end;
