function Ksub = gaussianKernel(X,rowInd,colInd,sigma)
    %% Guassian kernel generator
    % Outputs a submatrix of the Gaussian kernel with variance paramater 
    % gamma for the data rows of X. 
    %
    % usage : 
    %
    % input:
    %
    %  * X : A matrix with n rows (data points) and d columns (features)
    %
    %  * rowInd, colInd : Lists of indices between 1 and n. 
    %
    %  NOTE: colInd can be an empty list, in which case the **diagonal** 
    %  entries of the kernel will be output for the indices in rowInd.
    %  
    %  * gamma : kernel variance parameter
    %
    % output:
    %
    %  * Ksub : Let K(i,j) = e^-(gamma*||X(i,:)-X(j,:)||^2). Then Ksub = 
    %  K(rowInd,colInd). Or if colInd = [] then Ksub = diag(K)(rowInd).
    
    if(isempty(colInd))
        Ksub = ones(length(rowInd),1);
    else
        nsqRows = sum(X(rowInd,:).^2,2);
        nsqCols = sum(X(colInd,:).^2,2);
        Ksub = bsxfun(@minus,nsqRows,X(rowInd,:)*(2*X(colInd,:))');
        Ksub = bsxfun(@plus,nsqCols',Ksub);
    %    Ksub = exp(-gamma*Ksub);     
        
        Ksub = exp(-0.5.*Ksub./sigma^2);   
    end
end 

% 
% 
%   m_dis=pdist2(X,X);
%   G = exp(-0.5*(m_dis.^2)./sigma^2);