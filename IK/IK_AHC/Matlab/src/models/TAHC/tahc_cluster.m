function Z = tahc_cluster(dist, method)

    % ---INPUT---
    % dist:  The distance between the rows of X, form of pdist function
    % method: linkage function of traditional AHC.

    % ---OUTPUT---
    % Z: hierarchical cluster tree  which is represented as a matrix with size (numPts-1 X 3)

    [numPts, numPts2] = size(dist); % numPts is the number of points
    if (numPts == numPts2)
        error('ErrorFormOfDistanceMatrix: distance should be a vector form like output of pdist function! ');
    end
    
    Z = linkage(dist, method);

end