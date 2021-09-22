function [cost,Alpsupaux,b,posaux] = costsvmreg(K,StepSigma,DirSigma,Sigma,indsup,Alpsup,C,yapp,options);




nsup    = length(indsup);
[n]=length(yapp);

Sigma = Sigma+ StepSigma * DirSigma;
kerneloption.matrix=sumKbeta(K,Sigma);
kernel='numerical';
span=[];
lambdareg=options.lambdareg;
verbose=options.verbosesvm;

[xsup,ysup,aux,b,posaux,Alpsupaux,cost] = svmreg([],yapp,C,options.svmreg_epsilon,kernel,kerneloption,lambdareg,verbose,span,[],[],[],Alpsup);
