function [K,aux]=mklbuildkernel(x,kernelcell,kerneloptioncell,xsup,beta,options);



aux=[]; % compatibility
if nargin < 4
    beta=1;
    xsup=x;
    xegalxsup=1;
    weightK=[];
    options=[];
else
    if isempty(xsup)
        xsup=x;
    end;
end;


j=1;
for k=1:length(kernelcell);
    kernel=kernelcell{k};
    kerneloptionvec=kerneloptioncell{k};
    nbkernel=length(kerneloptionvec);
    if exist('options') & isfield(options,'variablecell')
        variabletouse=options.variablecell{k};   
    else
        variabletouse=1:size(x,2);
    end;
    for i=1:nbkernel
        K(:,:,j)=svmkernel(x(:,variabletouse),kernel,kerneloptionvec(i),xsup(:,variabletouse));
        j=j+1;
    end
end;


% Process Test Kernel Summation and Normalization + Normalization

if length(beta)==size(K,3)   
    if exist('options') & isfield(options,'weightK')
        weightK=options.weightK;
    else
        weightK=ones(size(K,3),1);
    end
    
    Kt=zeros(size(x,1),size(xsup,1));
    nbkernel=size(K,3);
    for i=1:nbkernel
        Kt=Kt+beta(i)*K(:,:,i)*weightK(i);
    end
    K=Kt;
end;

