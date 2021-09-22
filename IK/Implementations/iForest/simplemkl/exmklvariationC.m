
clear all
%close all


data='pima'
nbiter=1;
ratio=0.7;
N=100;
Cvec = logspace(-2,3,N);



options.algo='svmclass';
options.seuildiffsigma=1e-3;
options.seuildiffconstraint=0.1;
options.seuildualitygap=0.01;
options.goldensearch_deltmax=1e-1;
options.numericalprecision=1e-14;
options.stopvariation=0;
options.stopKKT=0;
options.stopdualitygap=1;
options.firstbasevariable='first';
options.nbitermax=500;
options.seuil=0.0;
options.seuilitermax=10;
options.lambdareg = 1e-8;
options.miniter=0;

puiss=0;

lambda = 1e-8; 
span=1;
verbose=1;
load(['../data/' data '/' data ]);
classcode=[1 -1];

kernelt={'gaussian' 'gaussian' 'poly' 'poly' };
kerneloptionvect={[0.5 1 2 5 7 10 12 15 17 20] [0.5 1 2 5 7 10 12 15 17 20] [1 2 3] [1 2 3]};
variablevec={'all' 'single' 'all' 'single'};
filename=['resultatvariationC-' data '.mat'];
[nbdata,dim]=size(x);
nbtrain=floor(nbdata*ratio)
rand('state',12);

    
    
    

    indice=randperm(nbdata);
    indapp=indice(1:nbtrain);
    indtest=indice(nbtrain+1:nbdata);
    xapp=x(indapp,:);
    xtest=x(indtest,:);
    yapp=y(indapp,:);
    ytest=y(indtest,:);

    [xapp,xtest]=normalizemeanstd(xapp,xtest);
    [nbdata,dim]=size(xapp);
    [kernel,kerneloptionvec,optionK.variablecell]=CreateKernelListWithVariable(variablevec,dim,kernelt,kerneloptionvect);
    [K]=mklbuildkernel(xapp,kernel,kerneloptionvec,[],[],optionK);
    option.power=0;
    [K,optionK.weightK]=WeightK(K,option);
    [Kt]=mklbuildkernel(xtest,kernel,kerneloptionvec,xapp,[],optionK);

 
    options2=options;
    optionssor=options;
%     first=1;
%     for j=1:length(Cvec); 
%         C=Cvec(j) 
%         if ~first
%             aux  =optionssor.alphainit;
%             aux(find(aux)==Cvec(j-1))=Cvec(j);
%             optionssor.alphainit=aux;
%             optionssor.thetainit=theta;
%         end;
%         tic
%         [w,b,beta,posw,fval(j),storysoren]=mklsvmclassSILP(K,yapp,C,verbose,optionssor);
%         theta=fval(i,j);
%         time(i,j)=toc
%         verbose=1;
%         betavecsoren(j,:)=beta';
%         optionssor.sigmainit=beta';
%         alphainit=zeros(size(yapp));
%         alphainit(posw)=w.*yapp(posw);
%         optionssor.alphainit=alphainit;
%                sumKt=sumKbeta(Kt,beta'.*optionK.weightK); % See mklbuildkernel 
%         ypred=sumKt(:,posw)*w +b ; 
%           bcsoren(i,j)=mean(sign(ypred)==ytest);
%         first=0;
%     end    
      first=1;  
    for j=length(Cvec):-1:1
        C=Cvec(j) 
        if ~first
            aux  =options.alphainit;
            aux(find(aux)==Cvec(j+1))=Cvec(j);
            options.alphainit=aux;
        end;
        
        tic
           [beta2,w2,b2,posw,story(j),obj(j)] = mklsvm(K,yapp,C,options,verbose);
           timelasso2(j)=toc
        alphainit=zeros(size(yapp));
        alphainit(posw)=w2.*yapp(posw);
        
        betavec2(j,:)=beta2;
       %sumKt=sumKbeta(Kt,beta2.*optionK.weightK); % See mklbuildkernel 
       % ypred=sumKt(:,posw)*w2 +b2 ; 
       %   bc(i,j)=mean(sign(ypred)==ytest);
        
        options.sigmainit=beta2;
        options.alphainit=alphainit;
        first=0;
    end;

plot(Cvec,betavec2,'LineWidth',2);
set(gca,'Xscale','log');
set(gcf,'color','white');
xlabel('C','Fonts',16)
ylabel('d_k','Fonts',16)
set(gca,'Fonts',16)


