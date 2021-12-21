function [Sigma,Alpsup,w0,pos,history,obj] = mklsvmls(K,yapp,C,option,verbose)

% USAGE [Sigma,Alpsup,w0,pos,SigmaH,obj] = mklsvmls(K,yapp,C,option,verbose)
%
% Inputs
%
% K         : NxNxD matrix containing all the Gram Matrix
% C         : SVM hyperparameter
% option    : mkl algorithm hyperparameter
%
%       option.nbitermax : maximal number of iterations (default 1000)
%       option.algo      : selecting algorithm svmclass (default) or svmreg
%       option.seuil     : threshold for zeroing kernel coefficients
%                          (default 1e-12)
%       option.sigmainit  : initial kernel coefficient (default average)
%       option.alphainit : initial Lagrangian coefficient
%       option.lambdareg : ridge regularization added to kernel's diagonal
%                          for SVM (default 1e-10)
%       option.verbosesvm : verbosity of SVM (default 0) see svmclass or
%                           svmreg
%       option.svmreg_epsilon : epsilon for SVM regression (mandatory for
%                         svm regression)
%
% Outputs
%
% Sigma         : the weigths
% Alpsup        : the weigthed lagrangian of the support vectors
% w0            : the bias
% pos           : the indices of SV
% history        : history of the weigths
% obj           : objective value

[n] = length(yapp);
%-----------------------------------------
%   specific to Large Scale
%-----------------------------------------
if ~isempty(K)
    nbkernel=K.size;
else
    error('No kernels defined ...');
end;





if ~isfield(option,'nbitermax');
    nloopmax=1000;
else
    nloopmax=option.nbitermax;
end;
if ~isfield(option,'algo');
    option.algo='svmclass';
end;
if ~isfield(option,'seuil');
    seuil=1e-12;
else
    seuil=option.seuil;
end
if ~isfield(option,'seuilitermax')
    option.seuilitermax=20;
end;
if ~isfield(option,'seuildiffsigma');
    option.seuildiffsigma=1e-5;
end
if ~isfield(option,'seuildiffconstraint');
    option.seuildiffconstraint=0.05;
end

if ~isfield(option,'lambdareg');
    lambdareg=1e-10;
    option.lambdareg=1e-10;
else
    lambdareg=option.lambdareg;
end

if ~isfield(option,'numericalcleaning');
    option.numericalcleaning=eps;
end
if ~isfield(option,'numericalprecision');
    option.numericalprecision=0;
end;


if ~isfield(option,'verbosesvm');
    verbosesvm=0;
    option.verbosesvm=0;
else
    verbosesvm=option.verbosesvm;
end

if ~isfield(option,'sigmainit');
    Sigma=ones(1,nbkernel)/nbkernel;
else
    Sigma=option.sigmainit ;
    ind=find(Sigma==0);
end;


if isfield(option,'alphainit');
    alphainit=option.alphainit;
else
    alphainit=[];
end;


%--------------------------------------------------------------------------------
% Options used in subroutines
%--------------------------------------------------------------------------------
if ~isfield(option,'goldensearch_deltmax');
    option.goldensearch_deltmax=1e-1;
end
if ~isfield(option,'goldensearchmax');
    optiongoldensearchmax=1e-8;
end;
if ~isfield(option,'firstbasevariable');
    option.firstbasevariable='first';
end;

%------------------------------------------------------------------------------%
% Initialize
%------------------------------------------------------------------------------%
kernel       = 'numerical';
span         = 1;
nloop = 0;
loop = 1;
numericalaccuracy=1e-9;
goldensearch_deltmaxinit= option.goldensearch_deltmax;
qpsize=3000;
chunksize=3000;
% Monitoring



%---------------------------------------------
% Initializing SVM
%---------------------------------------------
SumSigma=sum(Sigma);

switch option.sumbeta
    case 'storefullsum'
        kerneloption.matrix=sumKbetals(K,Sigma);
        qpsize=length(kerneloption.matrix);
        chunksize=qpsize;
        [xsup,Alpsup,w0,pos,aux,aux,obj] = svmclass([],yapp,C,lambdareg,kernel,kerneloption,verbosesvm,span,alphainit);

    case 'onthefly';
        K.sigma=Sigma;
        kerneloption=K;
         [xsup,Alpsup,w0,pos,aux,aux,obj] = svmclasslsformkl([],yapp,C,lambdareg,kernel,kerneloption,verbosesvm,span,qpsize,chunksize,alphainit);

    otherwise
    error('No kernels defined ...');
end;

 [grad] = gradsvmclassls(K,pos,Alpsup,C,yapp,option);
%--------------------------------------------
%
%---------------------------------------------

Sigmaold  = Sigma ;
Alpsupold = Alpsup ;
w0old     = w0;
posold    = pos ;
history.obj=[];
history.sigma=[];
history.KKTconstraint=[1];
history.dualitygap=[];
%------------------------------------------------------------------------------%
% Update Main loop
%------------------------------------------------------------------------------%

while loop ; nloop = nloop+1;

    history.sigma= [history.sigma;Sigma];
    history.obj=[history.obj obj];

    %-----------------------------------------
    % Update weigths Sigma
    %-----------------------------------------
    t = cputime ;
    [Sigma,Alpsup,w0,pos,obj] = mklsvmlsupdate(K,Sigma,pos,Alpsup,w0,C,yapp,grad,obj,option) ;
    if seuil ~=0 & max(Sigma)>seuil & nloop < option.seuilitermax
        Sigma=(Sigma.*(Sigma>seuil))*SumSigma/sum(Sigma.*(Sigma>seuil));
    end;

    %-------------------------------
    % Numerical cleaning
    %-------------------------------
    Sigma(find(abs(Sigma<option.numericalprecision)))=0;
     Sigma=Sigma/sum(Sigma);
    %-----------------------------------------------------------
    % Enhance accuracy of line search if necessary
    %-----------------------------------------------------------
    if max(abs(Sigma-Sigmaold))<option.numericalprecision & option.goldensearch_deltmax > optiongoldensearchmax
        option.goldensearch_deltmax=option.goldensearch_deltmax/10;
    elseif option.goldensearch_deltmax~=goldensearch_deltmaxinit
        option.goldensearch_deltmax*10;
    end;
    
    
    %----------------------------------------------------
    % process approximate KKT conditions
    %----------------------------------------------------
    [grad] = gradsvmclassls(K,pos,Alpsup,C,yapp,option);
    indpos=find(Sigma>0);
    indzero=find(Sigma==0);

    KKTconstraint=abs ((min(grad(indpos))-max(grad(indpos)))/min(grad(indpos))) ;
    KKTconstraintZero=  ( min(grad(indzero))>  max(grad(indpos)) );

    history.KKTconstraint=[history.KKTconstraint KKTconstraint];

    %----------------------------------------------------
    % process duality gap
    %----------------------------------------------------
    %    for i=1:nbkernel
    %    normek(i)=Alpsup'*K(pos,pos,i)*Alpsup;
    %    end
    normek=-2*grad ; % Alpsup'*K(pos,pos,i)*Alpsup;
    dualitygap=(obj +  0.5* max(normek) - sum(abs(Alpsup)))/obj;
    history.dualitygap=[history.dualitygap dualitygap];
    
   %------------------------------------------
    %  verbosity
    %------------------------------------------
    if verbose
        if nloop == 1 || rem(nloop,10)==0
            fprintf('------------------------------\n');
            fprintf('Iter | Obj.    | DiffBetas |\n');
            fprintf('------------------------------\n');
        end;
        fprintf('%d   | %8.4f | %6.4f   | %6.4f |\n',[nloop obj   max(abs(Sigma-Sigmaold)) dualitygap]);
    end
    

    %----------------------------------------------------
    % check variation of Sigma conditions
    %----------------------------------------------------
    if  option.stopvariation==1 & option.stopKKT== 0 & max(abs(Sigma - Sigmaold))<option.seuildiffsigma;
        loop = 0;
        fprintf(1,'variation convergence criteria reached \n');
        history.sigma= [history.sigma;Sigma];
        history.obj=[history.obj obj];
    end;
    %----------------------------------------------------
    % check approximate KKT conditions
    %----------------------------------------------------
    if  option.stopKKT==1 &  option.stopvariation== 0 & ( KKTconstraint < option.seuildiffconstraint & KKTconstraintZero  )
        loop = 0;
        fprintf(1,'KKT convergence criteria reached \n');
        history.sigma= [history.sigma;Sigma];
        history.obj=[history.obj obj];
    end;
    
    %----------------------------------------------------
    % check KKT and variation of Sigma conditions
    %----------------------------------------------------
    if  option.stopKKT== 1 & option.stopvariation== 1 & max(abs(Sigma - Sigmaold))<option.seuildiffsigma &  ( KKTconstraint < option.seuildiffconstraint & KKTconstraintZero  )
        loop = 0;
        fprintf(1,'variation and KKT convergence criteria reached \n');

        history.sigma= [history.sigma;Sigma];
        history.obj=[history.obj obj];
    end;
    
    %----------------------------------------------------
    % check duality gap
    %----------------------------------------------------
        if  option.stopdualitygap== 1 & dualitygap < option.seuildualitygap
        loop = 0;
        fprintf(1,'Duality gap criteria reached \n');

        history.sigma= [history.sigma;Sigma];
        history.obj=[history.obj obj];
    end;


    %----------------------------------------------------
    % check for premature convergence
    %----------------------------------------------------

    if  max(abs(Sigma-Sigmaold))<option.numericalprecision & (length(indpos)==1 | abs(max(Sigma(indpos))-1) < option.numericalprecision) ...
           & option.goldensearch_deltmax <=optiongoldensearchmax
        loop=0;
        history.sigma= [history.sigma;Sigma];
        history.obj=[history.obj obj];
        status=1;
        fprintf(1,'Premature Convergence KKT- Duality Gap Not Satisfied\n');
    end;


    %-----------------------------------------------------
    % check nbiteration conditions
    %----------------------------------------------------
    if nloop>=nloopmax ,
        loop = 0;
        history.sigma= [history.sigma;Sigma];
        history.obj=[history.obj obj];
        fprintf(1,'maximum number of iterations reached\n')
    end;
    if nloop < option.miniter & loop==0
        loop=1;
    end;

    %-----------------------------------------------------
    % Updating Variables
    %----------------------------------------------------

    Sigmaold  = Sigma ;
    Alpsupold = Alpsup ;
    w0old     = w0;
    posold    = pos ;

end;

