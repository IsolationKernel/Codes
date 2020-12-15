%
% Example MKL MultiClass SVM Classifiction
%



close all
clear all
%------------------------------------------------------
% Creating data
%------------------------------------------------------
n=20;
sigma=1.2;
nbclass=3;

x1= sigma*randn(n,2)+ ones(n,1)*[-1.5 -1.5];
x2= sigma*randn(n,2)+ ones(n,1)*[0 2];
x3= sigma*randn(n,2)+ ones(n,1)*[2 -1.5];
xapp=[x1;x2;x3];
yapp=[1*ones(1,n) 2*ones(1,n) 3*ones(1,n)]';

[n1, n2]=size(xapp);
[xtesta1,xtesta2]=meshgrid([-4:0.1:4],[-4:0.1:4]);
[na,nb]=size(xtesta1);
xtest1=reshape(xtesta1,1,na*nb);
xtest2=reshape(xtesta2,1,na*nb);
xtest=[xtest1;xtest2]';


%----------------------------------------------------------
%   Learning and Learning Parameters
%   Parameters are similar to those used for mklsvm
%-----------------------------------------------------------

C = 100;
lambda = 1e-7;
verbose = 1;
options.algo='oneagainstall';
options.seuildiffsigma=1e-4;
options.seuildiffconstraint=0.1;
options.seuildualitygap=1e-2;
options.goldensearch_deltmax=1e-1;
options.numericalprecision=1e-8;
options.stopvariation=1;
options.stopKKT=1;
options.stopdualitygap=0;
options.firstbasevariable='first';
options.nbitermax=500;
options.seuil=0.;
options.seuilitermax=10;
options.lambdareg = 1e-6;
options.miniter=0;
options.verbosesvm=0;
options.efficientkernel=1;
%------------------------------------------------------------

kernelt={'gaussian' 'gaussian' 'poly' 'poly' };
kerneloptionvect={[0.5 1 2 5 7 10 12 15 17 20] [0.5 1 2 5 7 10 12 15 17 20] [1 2 3] [1 2 3]};
variablevec={'all' 'single' 'all' 'single'};



[nbdata,dim]=size(xapp);
[kernel,kerneloptionvec,variableveccell]=CreateKernelListWithVariable(variablevec,dim,kernelt,kerneloptionvect);
[Weight,InfoKernel]=UnitTraceNormalization(xapp,kernel,kerneloptionvec,variableveccell);
K=mklkernel(xapp,InfoKernel,Weight,options);

%---------------------Learning & Testing ----------------


[beta,w,w0,pos,nbsv,SigmaH,obj] = mklmulticlass(K,yapp,C,nbclass,options,verbose);
xsup=xapp(pos,:);
Kt=mklkernel(xtest,InfoKernel,Weight,options,xsup,beta);
kernel='numerical';
kerneloption.matrix=Kt;
switch options.algo
    case 'oneagainstall'
        [ypred,maxi] = svmmultival([],[],w,w0,nbsv,kernel,kerneloption);
    case 'oneagainstone'
        [ypred,vote]=svmmultivaloneagainstone([],[],w,w0,nbsv,kernel,kerneloption);
end;


%------------------------------------------------------------------
%           Plotting the decision function
%-------------------------------------------------------------------
ypredmat=reshape(ypred,na,nb);
contour(xtesta1,xtesta2,ypredmat,[1 2 3]);hold on
style=['x+*'];
color=['bgr'];
hold on
for i=0:nbclass-1
    h=plot(xapp(i*n+1:(i+1)*n,1),xapp(i*n+1:(i+1)*n,2),[style(i+1) color(i+1)]);
    set(h,'LineWidth',2);
    hold on
end;

if ~isempty(xsup)
    h=plot(xsup(:,1),xsup(:,2),'ok');
    set(h,'LineWidth',2);
    axis( [ -4 4 -4 4]);
    legend('classe 1','classe 2','classe 3', 'Support Vector');
    hold off
end




