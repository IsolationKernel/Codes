function [K,weight]=WeightK(K,option)


% USAGE
% 
% [K,weight]=WeightK(K,option)
%
%  Normalize kernel to unit trace and eventually reweight them according
%  to Bach's 2004 trick.  K=K/(nbvp)^option.power if option.power > 0
%
% option.weigth :   normalize the kernels with these weights
% option.pow :    nbvp^power
%
%
% Outputs
%
% K             : normalized kernels
% weights       : normalizing weights
%
%


if nargin ==1 | ~isfield(option,'weight');
    weight=[];
else
    weight=option.weight;
end;
if nargin ==1 | ~isfield(option,'power');
    pow=0;
else
    pow=option.power;
end,

[nbdata,ndata,nbkernel]=size(K);
seuilvp=1/2/nbdata;

if isempty(weight)
    for i=1:nbkernel
	if sum(diag(K(:,:,i)))>0
        	weight(i)=1/sum(diag(K(:,:,i)));
	else
		weight(i)=0;
	end;
        K(:,:,i)=K(:,:,i)*weight(i);
        if pow>0
            [V,D]=eig(K(:,:,i));clear V;
            D=diag(D);
            nbvp=sum(D>seuilvp);
            weight(i)=weight(i)/(nbvp^pow);
            K(:,:,i)=K(:,:,i)/(nbvp^pow);
        end    
    end;
else
    
    for i=1:nbkernel
        K(:,:,i)=K(:,:,i)*weight(i);
    end;
end;
