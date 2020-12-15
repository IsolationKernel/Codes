function [ k ] = gau_kernel( x, y, gamma )
   xx = sum(x.^2, 2);
   k1 = repmat(xx, 1, size(y,1));
   zz = sum(y.^2, 2);
   k2 = repmat(zz', size(x,1), 1);
   d = k1 + k2 - 2* x*y';
   k = exp(-d*gamma);
end

