function [w,bsvm,Sigma,posw,fval,history]=mlksvmclass(K,yapp,C,verbose,option);

nbkernel=size(K,3);
n=size(yapp,1);
if ~isfield(option,'nbitermax');
    nbitermax=1000;
else
    nbitermax=option.nbitermax;
end;
if ~isfield(option,'seuildiffsigma');
    option.seuildiffsigma=1e-5;
end
if ~isfield(option,'seuildiffconstraint');
    option.seuildiffconstraint=0.05;
end
if ~isfield(option,'seuildualitygap');
    option.seuildiffconstraint=0.01;
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
    Sigma=ones(nbkernel,1)/nbkernel;
else
    Sigma=option.sigmainit ;
    if size(Sigma,1)==1
        Sigma=Sigma';
    end;
end;


if isfield(option,'alphainit');
    alphainit=option.alphainit;
else
    alphainit=[];
end;
if option.efficientkernel==1
    K = build_efficientK(K);
end;


kernel='numerical';
span=1;
verbosesvm=0;
sumSigma=sum(Sigma);
theta=-inf;




%  matrix and parameters
%  SVMClass Cost function evaluation
%


%---------------------------------------------------------------------
% Setting the linear prog  parameters
% nbvar = nbkernel+1;
%
% var = [theta Sigma1, Sigma_2,   ..., SigmaK];
%---------------------------------------------------------------------
f=[-1;zeros(nbkernel,1)];
Aeq=[0 ones(1,nbkernel)]; % 1 seule egalit�;
beq=sumSigma;
LB=[-inf;zeros(nbkernel,1)];
UB=[inf*ones(nbkernel,1)];
A=[];

optimopt=optimset('MaxIter',10000,'Display','off', 'TolCon',1e-3,'TolFun',1e-5);





nbverbose=1;

iter=0;
b=[];
Sigmaold=Sigma;
Sigmaold(1)=Sigmaold(1)-1;
loop=1;

history.theta=[];
history.sigma=[];
history.KKTconstraint=[1];
history.dualitygap=[1];
history.sigma= [history.sigma;Sigma'];
kerneloption.matrix=sumKbeta(K,Sigma);

x=[];
exitflag=0;
while loop


    kerneloption.matrix=sumKbeta(K,Sigma);
    if ~isempty(alphainit) & iter >0;
        alphainit=zeros(size(yapp));
        alphainit(posw)=alphaw;
    end;
    [xsup,w,bsvm,posw,timeps,alphaw,obj]=svmclass([],yapp,C,lambdareg,kernel,kerneloption,verbosesvm,span,alphainit);
    for i=1:nbkernel

        if ~isstruct(K)
            Saux(i)=0.5*w'*K(posw,posw,i)*w;% - sum(alphaw);
        else

            Kaux=devectorize(K.data(:,i));
            Saux(i)=0.5*w'*Kaux(posw,posw)*w;% - sum(alphaw);


        end;



    end;
    S=Saux-sum(alphaw);

    constraintviol=S*Sigma;


    sumfk2divdk= Saux*Sigma;
    primalobj=sumfk2divdk  +C*sum(max( 1-yapp.*(kerneloption.matrix(:,posw)*w + bsvm),0));
    dualobj= -max(Saux) + sum(alphaw);
    dualitygap=(primalobj-dualobj)/primalobj;

    %------------------------------------------------------
    % verbosity
    %----------------------------------------------------

    iter=iter+1;
    if verbose ~= 0

        if nbverbose == 1
            disp('------------------------------------------------');
            disp('iter     Theta      ConstViol     DeltaSigma');
            disp('------------------------------------------------');
        end
        if nbverbose == 20
            nbverbose=1;
        end

        if exitflag==0
            fprintf('%d   | %8.4f | %8.4f | %6.4f |%6.4f \n',[iter theta constraintviol  max(abs(Sigma-Sigmaold))], dualitygap);
        else
            fprintf('%d   | %8.4f | %8.4f | %6.4f | lp cvg pb \n',[iter theta constraintviol  max(abs(Sigma-Sigmaold))]);
        end;
        nbverbose = nbverbose+1;
    end
    %-----------------------------------------------------
    %       Maximum constraint violation check
    %------------------------------------------------------
    KKTconstraint=abs(S*Sigma/theta-1);
    history.KKTconstraint=[history.KKTconstraint KKTconstraint];
    history.dualitygap=[history.dualitygap dualitygap];


    %----------------------------------------------------
    % check variation of Sigma conditions
    %----------------------------------------------------
    if  option.stopvariation==1 & option.stopKKT==0 & max(abs(Sigma - Sigmaold))<option.seuildiffsigma;
        loop=0;
        fprintf(1,'variation convergence criteria reached \n');
    end;

    %----------------------------------------------------
    % check KKT conditions
    %----------------------------------------------------
    if   option.stopKKT==1 &  option.stopvariation==0 & ( KKTconstraint < option.seuildiffconstraint )
        loop = 0;
        fprintf(1,'KKT (maximum violation constraint) convergence criteria reached \n');
    end;

    %----------------------------------------------------
    % check KKT and variation of Sigma conditions
    %----------------------------------------------------
    if  option.stopvariation== 1 & option.stopKKT==1 & max(abs(Sigma - Sigmaold))<option.seuildiffsigma &  ( KKTconstraint < option.seuildiffconstraint )
        loop = 0;
        fprintf(1,'variation and KKT convergence criteria reached \n');
    end;
    %----------------------------------------------------
    % check for duality gap
    %----------------------------------------------------
    if option.stopdualitygap==1 & dualitygap<option.seuildualitygap
        loop=0;
        fprintf(1,'Duality gap of primal-dual \n');
    end;


    %----------------------------------------------------
    % check nbiteration conditions
    %----------------------------------------------------
    if iter>=nbitermax ,
        loop = 0;
        fprintf(1,'Maximal number of iterations reached \n');
    end;

    %----------------------------------------------------
    %  Optimize the weigths Sigma using a LP
    %----------------------------------------------------
    Sigmaold=Sigma;
    A=[A;1 -S];
    aux=0;
    b=[b;aux];
   %         [x,fval,exitflag] =linprog(f,A,b,Aeq,beq,LB,UB,[theta;Sigma],optimopt);

    if exist('lp_solve')==2
      
        sens=[-ones(size(A,1),1); zeros(size(Aeq,1),1)];
        [fval,x,lagrangia,exitflag]=lp_solve(-f,[A;Aeq],[b;beq],sens,LB,UB);
        
    elseif exist('linprog')==2
        [x,fval,exitflag] =linprog(f,A,b,Aeq,beq,LB,UB,[theta;Sigma],optimopt);
        exitflag=~(exitflag>0);
    else
        error('No available linear programming function...');
    end;
    if ~isempty(x)
        theta=x(1);
        Sigma=x(2:end);
        xold=x;
        fvalold=fval;
    else
        theta=xold(1);
        Sigma=xold(2:end);
        fval=fvalold;
        loop=0;
        fprintf(1,'Premature convergence of the algorithm \n');
    end;


    history.sigma= [history.sigma;Sigma'];
    history.theta=[history.theta theta];






end;



