function [cost,Alpsupaux,w0aux,posaux] = costsvmclass(K,StepSigma,DirSigma,Sigma,indsup,Alpsup,C,yapp,option);


global nbcall
nbcall=nbcall+1;

nsup    = length(indsup);
[n]=length(yapp);

Sigma = Sigma+ StepSigma * DirSigma;
kerneloption.matrix=sumKbeta(K,Sigma);
kernel='numerical';
span=1;
lambdareg=option.lambdareg;
verbosesvm=option.verbosesvm;
alphainit=zeros(size(yapp));
alphainit(indsup)=yapp(indsup).*Alpsup;
[xsup,Alpsupaux,w0aux,posaux,timeps,alpha,cost] = svmclass([],yapp,C,lambdareg,kernel,kerneloption,verbosesvm,span,alphainit);