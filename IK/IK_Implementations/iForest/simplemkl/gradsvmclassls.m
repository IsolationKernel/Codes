function [grad] = gradsvmclassls(Kinfo,indsup,Alpsup,C,yapp,option);

% Usage
%
% [grad] = gradsvmclassls(Kinfo,indsup,Alpsup,C,yapp,option);
%
% compute the gradient of all the weight variables.
%
%
% if the structure Kinfo contains the field
% 'x' and the struc 'info' then the kernel is computed
%  on the fly
%
% if the structure Kinfo has a field 'tempdir'
% the kernel is laoded from files stored in './tempdir/'
%
%
% see sumKbetals for the structure of K

% A.R 26/09/2007
nsup  = length(indsup);
[n] = length(yapp);
nbkernel=Kinfo.size;

chunksize=3000;
for k=1:nbkernel;


    if ~isstruct(Kinfo)


        grad(k) = - 0.5*Alpsup'*K.matrix(indsup,indsup,k)*(Alpsup) ;
    elseif isfield(Kinfo,'x') & isfield(Kinfo,'info')
        %------------------------------------------
        % On the fly
        %------------------------------------------

        variabletouse=Kinfo.info(k).variable;
        kernel=Kinfo.info(k).kernel;
        kerneloption=Kinfo.info(k).kerneloption;

        if length(indsup)<=chunksize
            K=svmkernel(Kinfo.x(indsup,variabletouse),kernel,kerneloption)*Kinfo.info(k).Weigth;
            grad(k) = - 0.5*Alpsup'*K*(Alpsup) ;
        else
            %
            % if the nb of SV is too large, it may be useful to chunk
            Nbchunk=ceil(length(indsup)/chunksize);
            vectemp=zeros(length(indsup),1);
            for i=1:Nbchunk
                fprintf('.') ;
                ind1=(i-1)*chunksize+1:min( [i*chunksize length(indsup)]);
                for j=1:Nbchunk
                    ind2=(j-1)*chunksize+1:min( [j*chunksize length(indsup)]);
                    K=svmkernel(Kinfo.x(indsup(ind1),variabletouse),kernel,kerneloption,Kinfo.x(indsup(ind2),variabletouse))*Kinfo.info(k).Weigth;
                    vectemp(ind1)=vectemp(ind1)+K*Alpsup(ind2);
                end;
            end;

            grad(k) = - 0.5*Alpsup'*vectemp;
        end;

    elseif isfield(Kinfo,'tempdir');
        % LOAD from file


        file=['K' int2str(k)];
        load([Kinfo.tempdir '/' file '.mat']);
        if isstruct(Kr)
            if isa(Kr.data,'single')
                Kr=devectorize_single(Kr.data);
            else
                Kr=devectorize(Kr.data);
            end;
        end;
        grad(k) = - 0.5*Alpsup'*Kr(indsup,indsup)*Alpsup ;



    end;

end;
