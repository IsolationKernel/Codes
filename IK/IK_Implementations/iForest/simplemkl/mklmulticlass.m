function [Sigma,Alpsup,w0,pos,nbsv,SigmaH,obj] = mklmulticlass(K,yapp,C,nbclass,option,verbose)

% USAGE [Sigma,Alpsup,w0,pos,Time,SigmaH,obj] = mkladapt(K,yapp,C,option,verbose)
%
% Input
% K         : NxNxD matrix containing all the Gram Matrix
% yapp      : training labels
% C         : SVM hyperparameter
% nbclass   : nb of classes in the problem
% verbose   : verbosity of algorithm
% option    : mkl algorithm hyperparameter
%
%       option.nbitermax : maximal number of iterations (default 1000)
%       option.algo      : selecting algorithm svmclass (default) or svmreg
%       option.seuil     : threshold for zeroing kernel coefficients
%                          (default 1e-12)
%       option.sigmainit  : initial kernel coefficient (default average)
%       option.alphainit : initial Lagrangian coefficient
%


[n] = length(yapp);
if ~isempty(K)
    if size(K,3)>1
        nbkernel=size(K,3);
        if option.efficientkernel==1
            K = build_efficientK(K);
        end;
    elseif option.efficientkernel==1 & isstruct(K);
        nbkernel=K.nbkernel;     
    end;
else
    error('No kernels defined ...');
end;

if ~isfield(option,'nbitermax');
    nloopmax=1000;
else
    nloopmax=option.nbitermax;
end;
if ~isfield(option,'algo');
    option.algo='oneagainstall';
end;
if ~isfield(option,'seuil');
    seuil=1e-12;
else
    seuil=option.seuil;
end
if ~isfield(option,'lambdareg');
    lambdareg=1e-10;
    option.lambdareg=1e-10;
else
    lambdareg=option.lambdareg;
end

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
goldensearch_deltmaxinit= option.goldensearch_deltmax;



% if option.efficientkernel==1
%     K = build_efficientK(K);
% end;

if nargout>=8,
    SigmaH = zeros(nloopmax,d);
end;

% Initializing SVM
t = cputime ;
SumSigma=sum(Sigma);
if ~isempty(K)
    kerneloption.matrix=sumKbeta(K,Sigma);
end;

switch option.algo
    case 'oneagainstall'

        [xsup,Alpsup,w0,nbsv,pos,obj]=svmmulticlassoneagainstall([],yapp,nbclass,C,lambdareg,kernel,kerneloption,verbosesvm);
     [grad] = gradsvmoneagainstall(K,pos,Alpsup,yapp,nbsv,option);
   
    case 'oneagainstone'
        [xsup,Alpsup,w0,nbsv,aux,pos,obj]=svmmulticlassoneagainstone([],yapp,nbclass,C,lambdareg,kernel,kerneloption,verbosesvm);
[grad] = gradsvmoneagainstone(K,pos,Alpsup,yapp,nbsv,option);

end;

Sigmaold  = Sigma ;
Alpsupold = Alpsup ;
w0old     = w0;
posold    = pos ;
history.obj=[];
history.sigma=[];
history.KKTconstraint=[1];
history.dualitygap=[];

%------------------------------------------------------------------------------%
% Update loop
%------------------------------------------------------------------------------%
if nloopmax==0
    SigmaH=[];
    return
end;
while loop ; nloop = nloop+1;

    SigmaH(nloop,:) = Sigma;
    history.sigma= [history.sigma;Sigma];
    history.obj=[history.obj obj];

    %---------------------------------------------
    % Update Sigma
    %---------------------------------------------
    t = cputime ;
    [Sigma,Alpsup,w0,pos,nbsv,obj] = mklmulticlassupdate(K,Sigma,pos,Alpsup,w0,C,yapp,nbclass,nbsv,grad,obj,option) ;
    %-----------------------------------------
    % Thresholding
    %-----------------------------------------

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
    %-----------------------------------------------------------
    % Enhance accuracy of line search if necessary
    %-----------------------------------------------------------
    if max(abs(Sigma-Sigmaold))==0 & option.goldensearch_deltmax > 1e-12
        option.goldensearch_deltmax=option.goldensearch_deltmax/10;
    elseif option.goldensearch_deltmax~=goldensearch_deltmaxinit
        option.goldensearch_deltmax*10;
    end;


    %----------------------------------------------------
    % process approximate KKT conditions
    %----------------------------------------------------
    switch option.algo
        case 'oneagainstall'
    [grad] = gradsvmoneagainstall(K,pos,Alpsup,yapp,nbsv,option);
        case 'oneagainstone';
                [grad] = gradsvmoneagainstone(K,pos,Alpsup,yapp,nbsv,option);

    end;
            

    indpos=find(Sigma>option.numericalprecision);
    indzero=find(abs(Sigma<=option.numericalprecision));


    KKTconstraint=abs ((min(grad(indpos))-max(grad(indpos)))/min(grad(indpos))) ;
    KKTconstraintZero=  ( min(grad(indzero))>  max(grad(indpos)) );

    history.KKTconstraint=[history.KKTconstraint KKTconstraint];

    %----------------------------------------------------
    % process duality gap
    %----------------------------------------------------
    normek=-2*grad ; % Alpsup'*K(pos,pos,i)*Alpsup;
    dualitygap=1; %(obj +  0.5* max(normek) - sum(abs(Alpsup)))/obj;
    history.dualitygap=[history.dualitygap dualitygap];

    %------------------------------------------
    %  verbosity
    %------------------------------------------
    if verbose
        if nloop == 1 || rem(nloop,10)==0
            fprintf('--------------------------------------\n');
            fprintf('Iter | Obj.    | DiffBetas | KKT C    |\n');
            fprintf('---------------------------------------\n');
        end;
        fprintf('%d   | %8.4f | %6.4f   | %6.4f |\n',[nloop obj   max(abs(Sigma-Sigmaold)) KKTconstraint]);
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
        status=2;
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


