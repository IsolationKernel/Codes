% Example of how to use the mklreg
%

clear all
close all
clc
%-------------------creating and plotting data----------------
n=100;
bruit=0.3;
freq=0.8;
x=linspace(0,8,n)';
xtest=linspace(0,8,n)';
xapp=x;
yapp=cos(exp(freq*x)) +randn(n,1)*bruit;
ytest=cos(exp(freq*xtest));


%----------------------Learning Parameters -------------------
C = 100; 
verbose=1;
options.algo='svmreg';
options.seuildiffsigma=1e-3;
options.seuildiffconstraint=0.01;
options.seuildualitygap=0.01;
options.goldensearch_deltmax=1e-2;
options.numericalprecision=1e-8;
options.stopvariation=0;
options.stopKKT=0;
options.stopdualitygap=1;
options.firstbasevariable='first';
options.nbitermax=500;
options.seuil=0;
options.seuilitermax=10;
options.lambdareg = 1e-8;
options.miniter=0;
options.verbosesvm=0;
options.svmreg_epsilon=0.01;
options.efficientkernel=0;
optionK.pow=0;

kernelt={'gaussian' 'poly'};
kerneloptionvect={[0.01:0.05:0.2 0.5 1 2 5 7 10 12 15 17 20] [1 2 3]};
variablevec={'all' 'all'};
dim=size(xapp,2);

[kernel,kerneloptionvec,optionK.variablecell]=CreateKernelListWithVariable(variablevec,dim,kernelt,kerneloptionvect);
[K]=mklbuildkernel(xapp,kernel,kerneloptionvec,[],[],optionK);
[K,optionK.weightK]=WeightK(K);   



[beta,w,b,posw,story,obj] = mklsvm(K,yapp,C,options,verbose);
kerneloption.matrix=mklbuildkernel(xtest,kernel,kerneloptionvec,xapp(posw,:),beta,optionK);
ypred=svmval([],[],w,b,'numerical',kerneloption);
plot(xtest,ytest,'b',xapp,yapp,'r',xapp,yapp,'r+',xtest,ypred,'g')