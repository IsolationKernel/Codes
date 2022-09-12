function [P, beta] = d2p(D, u, tol)
%D2P Identifies appropriate sigma's to get kk NNs up to some tolerance 
%
%   [P, beta] = d2p(D, kk, tol)
% 
% Identifies the required precision (= 1 / variance^2) to obtain a Gaussian
% kernel with a certain uncertainty for every datapoint. The desired
% uncertainty can be specified through the perplexity u (default = 15). The
% desired perplexity is obtained up to some tolerance that can be specified
% by tol (default = 1e-4).
% The function returns the final Gaussian kernel in P, as well as the 
% employed precisions per instance in beta.
%
%

% This file is part of the Matlab Toolbox for Dimensionality Reduction.
% The toolbox can be obtained from http://homepage.tudelft.nl/19j49
% You are free to use, change, or redistribute this code in any way you
% want for non-commercial purposes. However, it is appreciated if you 
% maintain the name of the original author.
%
% (C) Laurens van der Maaten, Delft University of Technology

    
    if ~exist('u', 'var') || isempty(u)
        u = 15;
    end
    if ~exist('tol', 'var') || isempty(tol)
        tol = 1e-4; 
    end
    
    % Initialize some variables
    n = size(D, 1);                     % number of instances
    P = zeros(n, n);                    % empty probability matrix
    beta = ones(n, 1);                  % empty precision vector
    logU = log(u);                      % log of perplexity (= entropy)

    % Run over all datapoints
    for i=1:n
        
        if ~rem(i, 500)
            disp(['Computed P-values ' num2str(i) ' of ' num2str(n) ' datapoints...']);
        end
        
        % Set minimum and maximum values for precision
        betamin = -Inf; 
        betamax = Inf;

        % Compute the Gaussian kernel and entropy for the current precision
        [H, thisP] = Hbeta(D(i, [1:i - 1, i + 1:end]), beta(i));
        
        % Evaluate whether the perplexity is within tolerance
        Hdiff = H - logU;
        tries = 0;
        while abs(Hdiff) > tol && tries < 50
            
            % If not, increase or decrease precision
            if Hdiff > 0
                betamin = beta(i);
                if isinf(betamax)
                    beta(i) = beta(i) * 2;
                else
                    beta(i) = (beta(i) + betamax) / 2;
                end
            else
                betamax = beta(i);
                if isinf(betamin) 
                    beta(i) = beta(i) / 2;
                else
                    beta(i) = (beta(i) + betamin) / 2;
                end
            end
            
            % Recompute the values
            [H, thisP] = Hbeta(D(i, [1:i - 1, i + 1:end]), beta(i));
            Hdiff = H - logU;
            tries = tries + 1;
        end
        
        % Set the final row of P
        P(i, [1:i - 1, i + 1:end]) = thisP;
    end    
%    disp(['Mean value of sigma: ' num2str(mean(sqrt(1 ./ beta)))]);
%    disp(['Minimum value of sigma: ' num2str(min(sqrt(1 ./ beta)))]);
%    disp(['Maximum value of sigma: ' num2str(max(sqrt(1 ./ beta)))]);
end
    


% Function that computes the Gaussian kernel values given a vector of
% squared Euclidean distances, and the precision of the Gaussian kernel.
% The function also computes the perplexity of the distribution.
function [H, P] = Hbeta(D, beta)
    P = exp(-D * beta);
    sumP = sum(P)+realmin;
    H = log(sumP) + beta * sum(D .* P) / sumP;
    % why not: H = exp(-sum(P(P > 1e-5) .* log(P(P > 1e-5)))); ???
    P = P / sumP;
end

