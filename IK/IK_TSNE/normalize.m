function [ ndata ] = normalize( data )
%  min-max normalisation to yield each attribute to be in [0,1] 
%   
ndata=[];
for i=1:size(data,2)
    d=data(:,i);
    if (max(d)-min(d))==0
        d=zeros(size(d,1),1);
        ndata=[ndata d];
    else
    d=(d-min(d))./(max(d)-min(d));
    ndata=[ndata d];
    end
end

