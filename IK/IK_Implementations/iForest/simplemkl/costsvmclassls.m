function [cost,Alpsupaux,w0aux,posaux] = costsvmclassls(K,StepSigma,DirSigma,Sigma,indsup,Alpsup,C,yapp,option);

% Usage 
% 
% [cost,Alpsupaux,w0aux,posaux] = costsvmclassls(K,StepSigma,DirSigma,Sigma,indsup,Alpsup,C,yapp,option);
%
% compute svm solution and cost for given value of Sigma and StepSigma
%
% option.sumbeta must be set to either 'storefullsum' or 'onthefly'
% depending if you want to store the full gram matrix or compute in
% on the fly
%


% AR 26/09/2007

nsup    = length(indsup);
[n]=length(yapp);

Sigma = Sigma+ StepSigma * DirSigma;

verbosesvm=option.verbosesvm;
lambdareg=option.lambdareg;
span=1;
alphainit=zeros(size(yapp));
alphainit(indsup)=yapp(indsup).*Alpsup;



kernel='numerical';
switch option.sumbeta
    case 'storefullsum'
        kerneloption.matrix=sumKbetals(K,Sigma);
        [xsup,Alpsupaux,w0aux,posaux,timeps,alpha,cost] = svmclass([],yapp,C,lambdareg,kernel,kerneloption,verbosesvm,span,alphainit);

    case 'onthefly';
        K.sigma=Sigma;
        kerneloption=K;        
        qpsize=3000;
        chunksize=3000;
        
        [xsup,Alpsupaux,w0aux,posaux,alpha,status,cost] = svmclasslsformkl([],yapp,C,lambdareg,kernel,kerneloption,option.verbosesvm,span,qpsize,chunksize,alphainit);

    otherwise
    error('No kernels defined ...');
end;

