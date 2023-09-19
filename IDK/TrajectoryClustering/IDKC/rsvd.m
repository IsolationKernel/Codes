function [U,D,t,e] = rsvd(W,k,p,q)
% RSVD   randomized SVD on PSD matrix W
% 
%   randsvd(W,k) perform rank-k matrix decomposition by randomized SVD. 
%   
%   [U,D] = randsvd(W,k) outputs the approximation leading k singular values
%   D of W and the according singular vectors U. 
%
%   [U,D,t,e] = randsvd(W,k) returns the computation time t (not includes
%   computing the error) and approximation error e in Frobenius norm.
%
%   [U,D,t,e] = randsvd(W,k,p) uses the over-sampling scalar p; default = 5.
%   
%   [U,D,t,e] = randsvd(W,k,p,q) uses the # of power iterations q; default = 2.
%
%   See also NYS RNYS
%
%   Reference: Halko, N. et al. "Finding structure with randomness: Stochastic
%   algorithms for constructing approximate matrix decompositions", 2009.

%   Author: Mu Li (limu.cn@gmail.com) 
%   Date:   04/18/2010 


    
%%%%%% check argin
error(nargchk(2, 4, nargin))

m = size(W,1);
if m ~= size(W,2)
    error('W should be symmetric');
end

if k > m
    error('k (=%d) is larger than the size of W (%d-by-%d)', k, m, m);
end

if nargin < 3
    p = 5;                              % default = 5;
    if k + p > m
        p = m - p;
    end
end

if nargin < 4
    q = 2;
end

%%%%%% check argout
error(nargoutchk(0, 4, nargout));

%%%%%% main algorithm

tstart = tic;

G = randn(m, k+p);                      % random Gaussian matrix

Y = W * G; 
% Z = Y;

for iq = 1 : q - 1
    Y = W * Y;
end

[Q,~] = qr(Y,0);
    
% B = ( Q' * Z ) / ( Q' * G );
B = Q' * W * Q;

[V,D] = svd(B,'econ');
V = V(:,1:k);
D = D(1:k,1:k);

U = Q * V;

t = toc(tstart);

if nargout > 3 || nargout == 0
    e = norm( W - U * diag(D) * U', 'fro');
end

%%%% display the result
if nargout == 0
    fprintf('Computational time:   %.3f s\n', t);
    fprintf('Approximation error:  %.3f (F-norm)\n', e);
end

end