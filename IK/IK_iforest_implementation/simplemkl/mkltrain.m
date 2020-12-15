function [model,K ] = mkltrain( train_label, train_data)
   addpath('../toollp');
   C = [500];
   verbose=1;
   [nbdata,dim]=size(train_data);

   options.algo='svmclass';
   %------------------------------------------------------
% choosing the stopping criterion
%------------------------------------------------------
options.stopvariation=0; % use variation of weights for stopping criterion 
options.stopKKT=0;       % set to 1 if you use KKTcondition for stopping criterion    
options.stopdualitygap=1; % set to 1 for using duality gap for stopping criterion

%------------------------------------------------------
% choosing the stopping criterion value
%------------------------------------------------------
options.seuildiffsigma=1e-3;        % stopping criterion for weight variation 
options.seuildiffconstraint=0.1;    % stopping criterion for KKT
options.seuildualitygap=0.0001;       % stopping criterion for duality gap

%------------------------------------------------------
% Setting some numerical parameters 
%------------------------------------------------------
options.goldensearch_deltmax=1e-1; % initial precision of golden section search
options.numericalprecision=1e-8;   % numerical precision weights below this value
                                   % are set to zero 
options.lambdareg = 1e-8;          % ridge added to kernel matrix 

%------------------------------------------------------
% some algorithms paramaters
%------------------------------------------------------
options.firstbasevariable='first'; % tie breaking method for choosing the base 
                                   % variable in the reduced gradient method 
options.nbitermax=500;             % maximal number of iteration  
options.seuil=0;                   % forcing to zero weights lower than this 
options.seuilitermax=15;           % value, for iterations lower than this one 

options.miniter=0;                 % minimal number of iterations 
options.verbosesvm=0;              % verbosity of inner svm algorithm 
options.efficientkernel=1;         % use efficient storage of kernels 


%------------------------------------------------------------------------
%                   Building the kernels parameters
%------------------------------------------------------------------------
kernelt={'gaussian' };
kerneloptionvect={2.^[-7:7]  };
variablevec={'all'  };
classcode=[1 -1];
train_label(train_label~=1) = -1;
[kernel,kerneloptionvec,variableveccell]=CreateKernelListWithVariable(variablevec,dim,kernelt,kerneloptionvect);
 [Weight,InfoKernel]=UnitTraceNormalization(train_data,kernel,kerneloptionvec,variableveccell);
  K=mklkernel(train_data,InfoKernel,Weight,options);
  [beta,w,b,posw,story,obj] = mklsvm(K,train_label,C,options,verbose);
   model.beta = beta;
   model.sv_coef = w;
   model.b = b;
   model.Weight = Weight;
   model.options = options;
   model.sv = train_data(posw,:);
   model.sv_indices = posw;
   model.InfoKernel = InfoKernel;
   sv_label = train_label(posw);
   model.nSV = [length(posw(sv_label==1)), length(posw(sv_label~=1))];
end

