function [G, mA, mB]=sharp(A, B, t)
% This function computes the point G=g(t) of the geodesic g joining 
% A and B such that g(0)=A and g(1)=B, i.e., A(A^{-1}B)^t=B(B^{-1}A)^{1-t}

% You can find the details of this algorithm in:
% B. Iannazzo, The geometric mean of two matrices from a computational
% viewpoint, arXiv preprint arXiv:1201.0101, 2011.

mA=cond(A);mB=cond(B);
if (mA > mB) % swap A and B if B is better conditioned
  C=A;A=B;B=C;t=1-t;
end
try
     RA=chol(A);
catch
     RA = chol(A+eye(size(A,1))*80);
end
RB=chol(B);
Z=RB/RA;
[U V]=eig(Z'*Z);
T=diag(diag(V).^(t/2))*U'*RA;
G=T'*T;
end