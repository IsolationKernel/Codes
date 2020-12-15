function [grad] = gradsvmoneagainstone(K,pos,Alpsup,yapp,nbsv,option);


[n] = length(yapp);
if ~isstruct(K)
    d=size(K,3);
else
    d=size(K.data,2); % efficient formulation of kernel
end;
nbclass=length(nbsv);

nbsv=[0 nbsv];
aux=cumsum(nbsv);
for k=1:d;
    S=0;
    for i=1:nbclass
        waux=Alpsup(aux(i)+1:aux(i)+nbsv(i+1));
        indsup=pos(aux(i)+1:aux(i)+nbsv(i+1));
        if ~isstruct(K)
            S=S +  (- 0.5* waux'* K(indsup,indsup,k)*waux) ;
        else
            Kaux=devectorize(K.data(:,k));
            S=S +  (- 0.5* waux'* Kaux(indsup,indsup)*waux) ;
        end;
    end;
    grad(k) = S;
end;

