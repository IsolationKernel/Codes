function K=sumKbetals(Kinfo,sigma,ind1,ind2)


% Build the full Gram matrix from the stored
% matrix files
%
% LoadFromDisk
%            K.size=length(Weight);
%            K.tempdir=tempdir;
%            K.nbdata=size(xapp,1);
%
%  OntheFly
%            K.size=length(Weight);
%            K.x=xapp;
%            K.info=InfoKernel;


indsigma=find(sigma);
n=length(indsigma);

if ~isfield(Kinfo,'x') & ~isfield(Kinfo,'info');
    %---------------------------
    % load from disk
    %---------------------------
    N1=Kinfo.nbdata;
    if nargin < 3
        ind1=1:N1;
        ind2=1:N1;
        K=zeros(N1,N1);
    else
        K=zeros(length(ind1),length(ind2));
    end;

    for i=1:n

        file=['K' int2str(indsigma(i))];
        load([Kinfo.tempdir  file '.mat']);
        if isstruct(Kr) % you have used efficient kernel representation
            if isa(Kr.data,'single')
                Kr=devectorize_single(Kr.data);
            else
                Kr=devectorize(Kr.data);
            end;
        end;
        K=K+ sigma(indsigma(i))* Kr(ind1,ind2);

       


    end;

else
    %--------------------------
    %   Compute on the fly
    %--------------------------
    N1=size(Kinfo.x,1);

    if nargin < 3
        ind1=1:N1;
        ind2=1:N1;
        K=zeros(N1,N1);
    else
        K=zeros(length(ind1),length(ind2));
    end;
    for k=1:n
        indk=indsigma(k);
        variabletouse=Kinfo.info(indk).variable;
        poids=sigma(indsigma(k))*Kinfo.info(indk).Weigth;
        kernel=Kinfo.info(indk).kernel;
        kerneloption=Kinfo.info(indk).kerneloption;
        K=K+poids*svmkernel(Kinfo.x(ind1,variabletouse),kernel,kerneloption,Kinfo.x(ind2,variabletouse));


    end
end;