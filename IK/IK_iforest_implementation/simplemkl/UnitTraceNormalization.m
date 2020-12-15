function [Weigth,InfoKernel]=UnitTraceNormalization(x,kernelvec,kerneloptionvec,variablevec)

chunksize=200;
N=size(x,1);
nbk=1;
for i=1:length(kernelvec);
    % i
    for k=1:length(kerneloptionvec{i})

        somme=0;

        chunks1=ceil(N/chunksize);

        for ch1=1:chunks1
            ind1=(1+(ch1-1)*chunksize) : min( N, ch1*chunksize);
            somme=somme+sum(diag(svmkernel(x(ind1,variablevec{i}),kernelvec{i},kerneloptionvec{i}(k))));
        end;
        %         for j=1:N
        %             somme=somme+svmkernel(x(j,variablevec{i}),kernelvec{i},kerneloptionvec{i}(k));
        %
        %         end
        if somme~=0
            Weigth(nbk)=1/somme;
            InfoKernel(nbk).kernel=kernelvec{i};
            InfoKernel(nbk).kerneloption=kerneloptionvec{i}(k);
            InfoKernel(nbk).variable=variablevec{i};
            InfoKernel(nbk).Weigth=1/somme;
            nbk=nbk+1;
%         else
%             A
        end;
    end;
end;