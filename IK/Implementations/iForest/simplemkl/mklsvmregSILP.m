function [w,bsvm,beta,posw]=mlksvmreg(K,yapp,C,epsilon,lambda,verbose,options);




if nargin < 7
    options.seuildiffconstraint=0.00001; % variation 1- S*beta/Theta 
    options.seuildiffbeta=0.01; % diff entre theta et constrainte et variation beta 
    options.nbitermax=5000;
    options.sumbeta=1;
end;
if ~isfield(options,'seuildiffconstraint');
    options.seuil=0.01;
end;
if ~isfield(options,'seuildiffbeta');
    options.seuildiff=0.01;
end;
if ~isfield(options,'nbitermax');
    options.nbitermax=1000;
end;

nbkernel=size(K,3);
n=size(yapp,1);

lambda=1e-5;
kernel='numerical';
span=1;
verbosesvm=0;




%  matrix and parameters
%  SVM REG Cost function evaluation 
%
I = eye(n);
Idif = [I -I];
c = [-epsilon+yapp ; -epsilon-yapp];

%---------------------------------------------------------------------
% Setting the linear prog  parameters
% nbvar = nbkernel+1; 
%
% var = [theta beta1, beta_2,   ..., betaK];
%---------------------------------------------------------------------
f=[-1;zeros(nbkernel,1)];
Aeq=[0 ones(1,nbkernel)]; % 1 seule egalité;
beq=options.sumbeta;
LB=[-inf;zeros(nbkernel,1)];
UB=[inf*ones(nbkernel,1)];
A=[];
theta=-inf;
optimopt=optimset('MaxIter',10000,'Display','off', 'TolCon',1e-3,'TolFun',1e-5,'LargeScale','on');


% initialisation
if ~isfield(options,'betainit');
    beta=ones(nbkernel,1)/nbkernel;
else
    beta=options.betainit;
end;



nbverbose=0;
alpha=[];
iter=0;
b=[];
betaold=beta;
betaold(1)=betaold(1)-1;

while max(abs(beta-betaold)) > options.seuildiffbeta & iter < options.nbitermax
    
    Kaux=zeros(n,n);
    
    kerneloption.matrix=sumKbeta(K,beta);
    [xsup,ysup,w,bsvm,posw,alpha] = svmreg([],yapp,C,epsilon,kernel,kerneloption,lambda,verbosesvm,[],[],[],[],alpha);
    for i=1:nbkernel
        H=Idif'*K(:,:,i)*Idif;
        S(i)=0.5*alpha'*H*alpha-c'*alpha;
    end;
    
    constraintviol=S*beta;
    
    
    if abs(S*beta-theta)<options.seuildiffconstraint
        break    
    end,
    
    
    A=[A;1 -S]; 
    aux=0;   
    b=[b;aux]; 
    
    betaold=beta;
    if exist('lp_solve')==2
        sens=[-ones(size(A,1),1); zeros(size(Aeq,1),1)];
        [fval,x,lagrangia,exitflag]=lp_solve(-f,[A;Aeq],[b;beq],sens,LB,UB);
    elseif exist('linprog')==2  
            [x,fval,exitflag] =linprog(f,A,b,Aeq,beq,LB,UB,[theta;beta],optimopt);
            exitflag=~(exitflag>0);         
    else
        error('No available linear programming function...');
    end;
    
    
    
    
    theta=x(1);
    beta=x(2:end);
    
    
    
    if verbose ~= 0 
        
        if nbverbose == 20 | nbverbose==0 
            disp('------------------------------------------------'); 
            disp('iter     Theta      ConstViol     DeltaBeta'); 
            disp('------------------------------------------------'); 
            nbverbose = 0; 
        end 
        
        if exitflag==0
            fprintf('%d   | %8.4f | %8.4f | %6.4f |\n',[iter theta constraintviol  max(abs(beta-betaold))]); 
        else
            fprintf('%d   | %8.4f | %8.4f | %6.4f | lp cvg pb \n',[iter theta constraintviol  max(abs(beta-betaold))]); 
        end;
        nbverbose = nbverbose+1; 
    end 
    
    iter=iter+1;
end;




function Kaux=sumKbeta(K,beta)  


nbkernel=size(K,3);
Kaux=zeros(size(K(:,:,1)));
for j=1:nbkernel
    Kaux=Kaux+ beta(j)*K(:,:,j);    
end
