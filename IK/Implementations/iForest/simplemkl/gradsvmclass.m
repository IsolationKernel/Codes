function [grad] = gradsvmclass(K,indsup,Alpsup,C,yapp,option);

nsup  = length(indsup);
[n] = length(yapp);
if ~isstruct(K)


    d=size(K,3);
    for k=1:d;

        %  grad(k) = - 0.5*Alpsup'*Kaux(indsup,indsup)*(Alpsup)  ;
        grad(k) = - 0.5*Alpsup'*K(indsup,indsup,k)*(Alpsup)  ;
    end;
else
    d=K.nbkernel;
    for k=1:d;
        if isa(K.data,'single')
            Kaux=devectorize_single(K.data(:,k));
        else
            Kaux=devectorize(K.data(:,k));
        end;
        grad(k) = - 0.5*Alpsup'*Kaux(indsup,indsup)*(Alpsup)  ;

    end;

end;

