function [cost,Alpsupaux,w0aux,posaux,nbsv] = costsvmoneagainstone(K,StepSigma,DirSigma,Sigma,Alpsup,C,yapp,pos,nbsv,nbclass,option);




%nsup    = length(indsup);
[n]=length(yapp);

Sigma = Sigma+ StepSigma * DirSigma;
kerneloption.matrix=sumKbeta(K,Sigma);
kernel='numerical';
span=1;
lambdareg=option.lambdareg;

warmstart.nbsv=nbsv;
warmstart.alpsup=Alpsup;
warmstart.pos=pos;
verbose=option.verbosesvm;
[xsup,Alpsupaux,w0aux,nbsv,aux,posaux,cost]=svmmulticlassoneagainstone([],yapp,nbclass,C,lambdareg,kernel,kerneloption,verbose);
