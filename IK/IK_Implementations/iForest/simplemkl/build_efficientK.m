function Kse = build_efficient_K(Ks);
% build efficient representation of the kernel matrices
% efficient_type = 0 -> not efficient
%                  1 -> efficient
%matlab7 =  str2num(version('-release')) >=14;


Kse.nbkernel = size(Ks,3);
Kse.n = size(Ks,1);

nbkernel = size(Ks,3);
n = size(Ks,1);
if isa(Ks,'single');
    Kse.data = zeros(n*(n+1)/2,nbkernel,'single');
else
    Kse.data = zeros(n*(n+1)/2,nbkernel);
end


for j=1:nbkernel
    if isa(Ks,'single');
        Kse.data(:,j) = vectorize_single(Ks(:,:,j));
    else
        Kse.data(:,j) = vectorize(Ks(:,:,j));
    end
end


