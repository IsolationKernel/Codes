function [Sigma,Alpsup,w0,pos,CostNew] = mklsvmpdate(K,Sigma,pos,Alpsup,w0,C,yapp,GradNew,CostNew,option)


%------------------------------------------------------------------------------%
% Initialize
%------------------------------------------------------------------------------%

d = length(Sigma);
gold = (sqrt(5)+1)/2 ;


SigmaInit = Sigma ;
SigmaNew  = SigmaInit ; 
descold = zeros(1,d);

%---------------------------------------------------------------
% Compute Current Cost and Gradient
%%--------------------------------------------------------------
% switch option.algo
%     case 'svmclass'
%         CostNew = costsvmclass(K,0,descold,SigmaNew,pos,Alpsup,C,yapp,option) ;
%         GradNew = gradsvmclass(K,pos,Alpsup,C,yapp,option) ;
%     case 'svmreg'
%         CostNew = costsvmreg(K,0,descold,SigmaNew,pos,Alpsup,C,yapp,option) ;
%         GradNew = gradsvmreg(K,Alpsup,yapp) ;
% end;

%[CostNew] = costsvmclassls(K,0,descold,SigmaNew,pos,Alpsup,C,yapp,option);
%[GradNew] = gradsvmclassls(K,pos,Alpsup,C,yapp,option);

NormGrad = GradNew*GradNew';
GradNew=GradNew/sqrt(NormGrad);
CostOld=CostNew;
%---------------------------------------------------------------
% Compute reduced Gradient and descent direction
%%--------------------------------------------------------------

switch option.firstbasevariable
    case 'first'
        [val,coord] = max(SigmaNew) ;

    case 'random'
        [val,coord] = max(SigmaNew) ;
        coord=find(SigmaNew==val);
        indperm=randperm(length(coord));
        coord=coord(indperm(1));
    case 'fullrandom'
        indzero=find(SigmaNew~=0);
        if ~isempty(indzero)
        [mini,coord]=min(GradNew(indzero));
        coord=indzero(coord);
    else
        [val,coord] = max(SigmaNew) ;
    end;
        
end;
GradNew = GradNew - GradNew(coord) ;
desc = - GradNew.* ( (SigmaNew>0) | (GradNew<0) ) ;
desc(coord) = - sum(desc);  % NB:  GradNew(coord) = 0




%----------------------------------------------------
% Compute optimal stepsize
%-----------------------------------------------------
stepmin  = 0;
costmin  = CostOld ;
costmax  = 0 ;
%-----------------------------------------------------
% maximum stepsize
%-----------------------------------------------------
ind = find(desc<0);
stepmax = min(-(SigmaNew(ind))./desc(ind));
deltmax = stepmax;
if isempty(stepmax) | stepmax==0
    Sigma = SigmaNew ;
    return
end,
if stepmax > 0.1
    stepmax=0.1;
end;


%-----------------------------------------------------
%  Projected gradient
%-----------------------------------------------------

while costmax<costmin;

            [costmax,Alpsupaux,w0aux,posaux] = costsvmclassls(K,stepmax,desc,SigmaNew,pos,Alpsup,C,yapp,option) ;


    if costmax<costmin
        costmin = costmax;
        SigmaNew  = SigmaNew + stepmax * desc;
        % SigmaNew  =SigmaP;
        % project descent direction in the new admissible cone
        % keep the same direction of descent while cost decrease
        desc = desc .* ( (SigmaNew>option.numericalprecision) | (desc>0) ) ;
        desc(coord) = - sum(desc([[1:coord-1] [coord+1:end]]));  
        ind = find(desc<0);
        Alpsup=Alpsupaux;
        w0=w0aux;
        pos=posaux;
        if ~isempty(ind)
            stepmax = min(-(SigmaNew(ind))./desc(ind));
            deltmax = stepmax;
            costmax = 0;
        else
            stepmax = 0;
            deltmax = 0;
        end;
        
    end;
end;


%-----------------------------------------------------
%  Linesearch
%-----------------------------------------------------

Step = [stepmin stepmax];
Cost = [costmin costmax];
[val,coord] = min(Cost);
% optimization of stepsize by golden search
while (stepmax-stepmin)>option.goldensearch_deltmax*(abs(deltmax));
    stepmedr = stepmin+(stepmax-stepmin)/gold;
    stepmedl = stepmin+(stepmedr-stepmin)/gold;                     

            [costmedr,Alpsupr,w0r,posr] = costsvmclassls(K,stepmedr,desc,SigmaNew,pos,Alpsup,C,yapp,option) ;
            [costmedl,Alpsupl,w01,posl] = costsvmclassls(K,stepmedl,desc,SigmaNew,posr,Alpsupr,C,yapp,option) ;
 
    Step = [stepmin stepmedl stepmedr stepmax];
    Cost = [costmin costmedl costmedr costmax];
    [val,coord] = min(Cost);
    switch coord
        case 1
            stepmax = stepmedl;
            costmax = costmedl;
            pos=posl;
            Alpsup=Alpsupl;
            w0=w01;
        case 2
            stepmax = stepmedr;
            costmax = costmedr;
            pos=posr;
            Alpsup=Alpsupr;
            w0=w0r;
        case 3
            stepmin = stepmedl;
            costmin = costmedl;
            pos=posl;
            Alpsup=Alpsupl;
            w0=w01;
        case 4
            stepmin = stepmedr;
            costmin = costmedr;
            pos=posr;
            Alpsup=Alpsupr;
            w0=w0r;
    end;
end;


%---------------------------------
% Final Updates
%---------------------------------

CostNew = Cost(coord) ;
step = Step(coord) ;
% Sigma update
if CostNew < CostOld ;
    SigmaNew = SigmaNew + step * desc;  
    
end;       

Sigma = SigmaNew ;
