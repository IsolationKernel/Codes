%
%

clear all
close all


nbiter=1;

ratio=0.1;
data='credit'

C = [1000];

options.seuildiffsigma=1e-4;
options.seuildiffconstraint=0.1;
options.seuildualitygap=0.01;
options.goldensearch_deltmax=1e-1;
options.numericalprecision=1e-8;
options.stopvariation=0;
options.stopKKT=0;
options.stopdualitygap=1;
options.firstbasevariable='first';
options.nbitermax=500;
options.seuil=0.00;
options.seuilitermax=10;
options.lambdareg = 1e-8;
options.miniter=0;
options.verbosesvm=0;
options.sumbeta='storefullsum'; % 'storefullsum' or 'onthefly'
options.storefly='store';
options.efficientkernel=0;

verbose=1;

load(['../data/' data '/' data ]);
%x=single(x);
classcode=[1 -1];

kernelt={'gaussian' 'gaussian' 'poly' 'poly' };
kerneloptionvect={[0.5 1 2 5 7 10 12 15 17 20] [0.5 1 2 5 7 10 12 15 17 20] [1 2 3] [1 2 3]};
variablevec={'all' 'single' 'all' 'single'};


% % spamdata
%  kerneloptionvect={[0.5 1 2 5 7 10 12 15 17 20]  [1 2 3 4]  [1 2 3 4]};
%  variablevec={'all' 'all' 'single'}
%
% % coverbin
%  kerneloptionvect={[0.5 1 2 5 7 10 12 15 17 20]  [1 2 3 4]};
%  variablevec={'all' 'all'}

[nbdata,dim]=size(x);
nbtrain=floor(nbdata*ratio);
rand('state',0);;

for i=1: nbiter




    indice=randperm(nbdata);
    indapp=indice(1:nbtrain);
    indtest=indice(nbtrain+1:nbdata);
    xapp=x(indapp,:);
    xtest=x(indtest,:);
    yapp=y(indapp,:);
    ytest=y(indtest,:);

    [xapp,xtest]=normalizemeanstd(xapp,xtest);

    fprintf('Creating & Processing Kernels...');
    %------------------------------------------------------
    % create the list of kernels and their weights
    %------------------------------------------------------

    [kernelvec,kerneloptionvec,optionK.variablecell]=CreateKernelListWithVariable(variablevec,dim,kernelt,kerneloptionvect);
    [Weight,InfoKernel]=UnitTraceNormalization(xapp,kernelvec,kerneloptionvec,optionK.variablecell);

    %%---------------------------------------------
    %%  Comments
    %%---------------------------------------------
    %% for large scale mkl, 2 bottlenecks
    %%  - two many examples : one has to use SVM with decomposition methods
    %%  - two many kernels and examples: one has to store precomputed kernels
    %%    or build them on the fly during the decomposition method
    %%
    %%  if there are two many kernels but their sums can be stored in
    %%  memory (say nb of examples is less than 3000), one can set
    %% option.sumbeta to 'storefullsum' so that the kernels are summed from
    %% file 'store' or computed on the fly 'fly' and stored in memory... no decomposition methods is used.
    %%
    %%
    %% In the other cases, decomposition methods has to be used. Kernels
    %% can be computed on the fly or loaded on the fly. For doing so
    %% option.sumbeta must be set to 'onthefly'.



    switch options.storefly

        case 'store'
            %------------------------------------
            %   save each row of a kernel
            %------------------------------------
            tempdir='./temp/';


            for k=1:length(Weight)

                Kr=svmkernel(xapp(:,InfoKernel(k).variable),InfoKernel(k).kernel,InfoKernel(k).kerneloption);

                Kr=Kr*Weight(k);
                if options.efficientkernel
                    Kr=build_efficientK(Kr);
                end;
                save([tempdir 'K' int2str(k) '.mat'],'Kr');

            end;


            K.size=length(Weight);
            K.tempdir=tempdir;
            K.nbdata=size(xapp,1);




        case 'fly'
            %------------------------------------------
            %      For doing on the fly kernel processing
            %----------------------------------------
            K.size=length(Weight);
            K.x=xapp;
            K.info=InfoKernel;
            %%Example : Kin=sumKbetaread(K,sigma,1:size(xapp,1), 1:size(xapp,1));

    end;

    fprintf('done \n');
    Nr=10;
    sigmainit=[rand(1,Nr) zeros(1,K.size-Nr)];
    options.sigmainit=sigmainit/sum(sigmainit);



    % Cv=VectorizeCofSVM(yapp,C);
    tic
    [beta,w,b,posw,story,obj]=mklsvmls(K,yapp,C,options,verbose);
    time(i)=toc;

    save (['resultat-' data '.mat'],'time')

end;
