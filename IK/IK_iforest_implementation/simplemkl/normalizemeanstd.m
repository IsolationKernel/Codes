function [xapp,xtest,meanxapp,stdxapp] = normalizemeanstd(xapp,xtest,meanx,stdx)

% USAGE
% 
%  [xapp,xtest,meanxapp,stdxapp] = normalizemeanstd(xapp,xtest)
%
% normalize inputs and output mean and standard deviation to 0 and 1
%
% 
tol=1e-5;



nbsuppress=0;
if nargin <3
    meanxapp=mean(xapp);
    stdxapp=std(xapp);
else
    meanxapp=meanx;
    stdxapp=stdx;
end;
nbxapp=size(xapp,1);
indzero=find(abs(stdxapp)<tol);
%keyboard
if ~isempty(indzero)

    stdxapp(indzero)=1;

end;
nbvar=size(xapp,2);

xapp= (xapp - ones(nbxapp,1)*meanxapp)./ (ones(nbxapp,1)*stdxapp) ;

if nargin >1 & ~isempty(xtest)
    nbxtest=size(xtest,1);
    xtest= (xtest - ones(nbxtest,1)*meanxapp)./ (ones(nbxtest,1)*stdxapp );
else
    xtest=[];
end;