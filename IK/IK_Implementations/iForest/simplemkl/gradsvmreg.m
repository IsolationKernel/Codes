function [grad] = gradsvmreg(K,Alpsup,yapp,option);

[n] = length(yapp);
d=size(K,3);

I = eye(n);
Idif = [I -I];


if ~isstruct(K)
for k=1:d;
    H = Idif'*K(:,:,k)*Idif;
   grad(k) =  -0.5*Alpsup'*H*Alpsup ;
end;


else
    d=K.nbkernel;
    for k=1:d;
        if isa(K.data,'single')
            Kaux=devectorize_single(K.data(:,k));
        else
            Kaux=devectorize(K.data(:,k));
        end;
        H = Idif'*Kaux*Idif;
        grad(k) = - 0.5*Alpsup'*H*(Alpsup)  ;

    end;

end;