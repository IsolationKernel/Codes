%
% Example MKL MultiClass SVM Classifiction
%



close all
clear all
%--------------------------------------------------
% Creatong

data='dna';
ratio=0.1;
classcode= [ 1 2 3];
nbclass=3;
%----------------------------------------------------------
%   Learning and Learning Parameters
C = 1000;
lambda = 1e-7;
verbose = 1;
options.algo='oneagainstall';
options.seuildiffsigma=1e-4;
options.seuildiffconstraint=0.1;
options.seuildualitygap=1e-2;
options.goldensearch_deltmax=1e-1;
options.numericalprecision=1e-9;
options.stopvariation=1;
options.stopKKT=1;
options.stopdualitygap=0;
options.firstbasevariable='first';
options.nbitermax=500;
options.seuil=0.000;
options.seuilitermax=10;
options.lambdareg = 1e-8;
options.miniter=0;
options.verbosesvm=0;
options.efficientkernel=1;
%------------------------------------------------------------

kernelt={'gaussian' 'gaussian' 'poly' 'poly' };
kerneloptionvect={[0.5 1 2 5 7 10 12 15 17 20]   [1 2 3]};
variablevec={'all'  'all' };

load(['../data/' data '/' data '.mat']);
nbtrain=round(ratio*size(y,1));

randn('seed',0);
rand('seed',0);

[nbdata,dim]=size(x);
[xapp,yapp,xtest,ytest,indice]=CreateDataAppTest(x, y, nbtrain,classcode);
[xapp,xtest,meanxapp,stdxapp] = normalizemeanstd(xapp,xtest);



%-------------------- Creating kernels ------------------------------
[kernel,kerneloptionvec,variableveccell]=CreateKernelListWithVariable(variablevec,dim,kernelt,kerneloptionvect);
[Weight,InfoKernel]=UnitTraceNormalization(xapp,kernel,kerneloptionvec,variableveccell);
K=mklkernel(xapp,InfoKernel,Weight,options);

%------------------------------------------------------------------
%
%  K is a 3-D matrix, where K(:,:,i)= i-th Gram matrix
%
%------------------------------------------------------------------
% or K can be a structure with uses a more efficient way of storing
% the gram matrices
%
K = build_efficientK(K);

%---------------------One Against All algorithms---------------------

[beta,w,w0,pos,nbsv,SigmaH,obj] = mklmulticlass(K,yapp,C,nbclass,options,verbose);
xsup=xapp(pos,:);
Kt=mklkernel(xtest,InfoKernel,Weight,options,xsup,beta);
kernel='numerical';
kerneloption.matrix=Kt;
[ypred,maxi] = svmmultival([],[],w,w0,nbsv,kernel,kerneloption);
[Conf,metric]=ConfusionMatrix(ypred,ytest,classcode)





