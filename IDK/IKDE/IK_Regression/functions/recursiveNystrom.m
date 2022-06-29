function [C,W] = recursiveNystrom(X,s,kernelFunc,accelerated_flag)
    %% Recursive Nystrom Sampling Based on Ridge Leverage Scores
    % This file implements Algorithm 3 as described in 
    %   https://arxiv.org/abs/1605.07583
    %
    % usage : 
    %
    % input :
    %
    %  * X : matrix with n rows (data points) and d columns (features)
    %
    %  * s : the number of samples used to construct the Nystrom
    %  approximation. default = sqrt(n). Generally should set s < n.
    %
    %  * kernelFunc : A function that can compute arbitrary submatrices of
    %  X's kernel matrix for some positive semidefinite kernel. For
    %  implementation specifics, see the provided example gaussianKernel.m
    %
    %  * accelerated_flag: either 0 or 1. default = 0. If the flag is set to 1,
    %  the code uses an accelerated version of the algorithm as described
    %  in Section 5.2.1 of https://arxiv.org/abs/1605.07583. This version 
    %  will output a lower quality Nystrom approximation, but will run more
    %  quickly. We recommend setting accelerated_flag = 0 (the default) 
    %  unless the standard version of the algorithm runs too slowly for 
    %  your purposes.
    %
    % output :
    %
    %  * C : A subset of s columns from A's n x n kernel matrix.
    %
    %  * W : An s x s positive semidefinite matrix such that 
    %  C*W*C' approximates K.
    %
    % In learning applications, it is natural to compute F = C*chol(W)'.
    % F has n rows and each row can be supplied as a data point to a linear
    % algorithm (regression, SVM, etc.) to approximate the kernel version 
    % of the algorithm. Caveat: the accelerated version of our algorithm 
    % runs in O(ns) time. Computing F = C*chol(W)' takes O(ns^2) time, so 
    % it may be more prudent to access the matrix implicitly.
    %
    % example call:
    %  
    %  Compute a Nystrom approximation for a Gaussian kernel matrix with
    %  variance parameter gamma = 40,. I.e. the kernel function for points
    %  x,y is e^-(40*||x - y||^2).
    %
    %  gamma = 40;
    %  kFunc = @(X,rowInd,colInd) gaussianKernel(X,rowInd,colInd,gamma);
    %  [C,W] = recursiveNystrom(X,s,kFunc);
    
    %% Parameter processing and defaults    
    if nargin == 0
         error('recursiveNystrom:TooFewInputs','requires at least 1 input argument');
    end
    if nargin < 4
        accelerated_flag = 0;
    end
    % kernelFunc and s parameters should really be set by the user but we 
    % provide some defaults
    if nargin < 3
        % default to Gaussian kernel with gamma=1
        kernelFunc = @(X,rowInd,colInd) gaussianKernel(X,rowInd,colInd,1);
    end
    if nargin < 2
        s = ceil(sqrt(size(X,1)));
    end
    
    [n,d] = size(X);

    if(~accelerated_flag)
        % in the standard algorithm s samples are used in the final Nystrom
        % approximation as well as at each recursive level
        sLevel = s;
    else
        % in the accelerated version  < s samples are used at recursive 
        % levels to keep the total runtime at O(n*s)
        sLevel = ceil(sqrt((n*s + s^3)/(4*n)));
    end
    
    %% Start of algorithm  
    oversamp = log(sLevel);
    k = ceil(sLevel/(4*oversamp));
    nLevels = ceil(log(n/sLevel)/log(2));
    % random permutation for successful uniform samples
    perm = randperm(n);
    
    % set up sizes for recursive levels
    lSize = zeros(1,nLevels+1);
    lSize(1) = n;
    for i = 2:nLevels+1
        lSize(i) = ceil(lSize(i-1)/2);
    end
    
    % rInd: indices of points selected at previous level of recursion 
    % at the base level it's just a uniform sample of ~sLevel points
    samp = 1:lSize(end);
    rInd = perm(samp);
    weights = ones(length(rInd),1);

    % we need the diagonal of the whole kernel matrix, so compute upfront
    kDiag = kernelFunc(X,1:n,[]);
    
    %% Main recursion, unrolled for efficiency 
    for l = nLevels:-1:1  
        % indices of current uniform sample
        rIndCurr = perm(1:lSize(l));
        % build sampled kernel
        KS = kernelFunc(X,rIndCurr,rInd);
        SKS = KS(samp,:);
        SKSn = size(SKS,1);
        
        % optimal lambda for taking O(klogk) samples
        if(k >= SKSn)
            % for the rare chance we take less than k samples in a round
            lambda = 10e-6;
            % don't set to exactly 0 to avoid stability issues
        else
            lambda = (sum(diag(SKS).*weights.^2) - sum(abs(real(eigs(@(x) (SKS*(x.*weights)).*weights, SKSn, k)))))/k;
        end
        
        % compute and sample by lambda ridge leverage scores
        if(l ~= 1)
            % on intermediate levels, we independently sample each column
            % by its leverage score. the sample size is sLevel in expectation
            R = inv(SKS + diag(lambda*weights.^(-2)));
            % max(0,.) helps avoid numerical issues, unnecessary in theory
            levs = min(1,oversamp*(1/lambda)*max(0,(kDiag(rIndCurr) - sum((KS*R).*KS,2))));
            samp = find(rand(1,lSize(l)) < levs');
            % with very low probability, we could accidentally sample no
            % columns. In this case, just take a fixed size uniform sample.
            if(isempty(samp))
                levs(:) = sLevel/lSize(l);
                samp = randperm(lSize(l),sLevel);
            end
            weights = sqrt(1./(levs(samp)));

        else
            % on the top level, we sample exactly s landmark points without replacement
            R = inv(SKS + diag(lambda*weights.^(-2)));
            levs = min(1,(1/lambda)*max(0,(kDiag(rIndCurr) - sum((KS*R).*KS,2))));
            levs(levs<0)=0+eps;                      
            levs(isnan(levs))=0+eps;
            samp = datasample(1:n,s,'Replace',false,'Weights',levs);
        end
        rInd = perm(samp);
    end

    % build final Nystrom approximation
    % pinv or inversion with slight regularization helps stability
    C = kernelFunc(X,1:n,rInd);
    SKS = C(rInd,:);
    W = inv(SKS+(10e-6)*eye(s,s));
end

%-------------------------------------------------------------------------------------
% Copyright (c) 2017 Christopher Musco, Cameron Musco
% 
% Permission is hereby granted, free of charge, to any person obtaining a copy
% of this software and associated documentation files (the "Software"), to deal
% in the Software without restriction, including without limitation the rights
% to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
% copies of the Software, and to permit persons to whom the Software is
% furnished to do so, subject to the following conditions:
% 
% The above copyright notice and this permission notice shall be included in
% all copies or substantial portions of the Software.
% 
% THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
% IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
% FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
% AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
% LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
% OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
% THE SOFTWARE.