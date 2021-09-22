function [ cl, halo] = DensityPeak( SimMatrix,dc,top)
%DENSITYPEAK Summary of this function goes here
%   Detailed explanation goes here


NumIns=size(SimMatrix,2);
 

Density=sum(SimMatrix'<=dc);

%%%%%%%%%%%%%%%%%%
maxd=max(max(SimMatrix)); % max dis


[~,SortDensity]=sort(Density,'descend'); % SortDensity is index
MinDist(SortDensity(1))=-1.;
nneigh(SortDensity(1))=0;

for ii=2:NumIns
   MinDist(SortDensity(ii))=maxd;
   nneigh(SortDensity(ii))=SortDensity(1);
   for jj=1:ii-1
     if(SimMatrix(SortDensity(ii),SortDensity(jj))<MinDist(SortDensity(ii)))
        MinDist(SortDensity(ii))=SimMatrix(SortDensity(ii),SortDensity(jj));
        nneigh(SortDensity(ii))=SortDensity(jj); % nearest neigbour index
     end
   end
end
MinDist(SortDensity(1))=max(MinDist(:));

for i=1:NumIns
  ind(i)=i;
  gamma(i)=Density(i)*MinDist(i);
end
  %% normalise
    Density=normalize(Density')+0.000000001;
    MinDist=normalize(MinDist')+0.000000001;
%% Select original

% tt=gscatter(Density,MinDist,class,'brcrgywk','xo*',5,'off');
% title ('Decision Graph','FontSize',15.0)
% xlabel ('Density')
% ylabel ('MinDist')
% 
% 
% subplot(1,1,1)
% rect = getrect(1);
% rhomin=rect(1);
% deltamin=rect(2);
% NCLUST=0;
% for i=1:NumIns
%   cl(i)=-1;
% end
% for i=1:NumIns
%   if ( (Density(i)>rhomin) && (MinDist(i)>deltamin))
%      NCLUST=NCLUST+1;
%      cl(i)=NCLUST;
%      icl(NCLUST)=i;
%   end
% end
% fprintf('NUMBER OF CLUSTERS: %i \n', NCLUST);
% disp('Performing assignation')
%  

%% select dense*dist

% Mult=(Density').*(MinDist');
% [VSortMult,ISortMult]=sort(Mult,'descend');
%  
% tt=gscatter(1:NumIns,VSortMult,class(ISortMult),'brcrgywk','xo*',5,'off');
% title ('Decision Graph','FontSize',15.0)
% xlabel ('Density')
% ylabel ('MinDist')
%   
% rect = getrect(1);
% rhomin=rect(1);
% deltamin=rect(2);
% NCLUST=0;
% for i=1:NumIns
%   cl(i)=-1;
% end
% for i=1:NumIns
%   if ((Mult(i)>deltamin))
%      NCLUST=NCLUST+1;
%      cl(i)=NCLUST;
%      icl(NCLUST)=i;
%   end
% end
% fprintf('NUMBER OF CLUSTERS: %i \n', NCLUST);
% disp('Performing assignation')


%% select top k

Mult=(Density').*(MinDist');
[VSortMult,ISortMult]=sort(Mult,'descend');

deltamin=VSortMult(top);
NCLUST=0;
cl=zeros(size(SimMatrix,1),1);
for i=1:NumIns
  cl(i)=-1;
end
for i=1:NumIns
  if ((Mult(i)>=deltamin))
     NCLUST=NCLUST+1;
     cl(i)=NCLUST;
     icl(NCLUST)=i;
  end
end
% fprintf('NUMBER OF CLUSTERS: %i \n', NCLUST);
% disp('Performing assignation')

%% assignation
for i=1:NumIns
  if (cl(SortDensity(i))==-1)
    cl(SortDensity(i))=cl(nneigh(SortDensity(i)));
  end
end
%halo
for i=1:NumIns
  halo(i)=cl(i);
end
if (NCLUST>1)
  for i=1:NCLUST
    bord_rho(i)=0.;
  end
  for i=1:NumIns-1
    for j=i+1:NumIns
      if ((cl(i)~=cl(j))&& (SimMatrix(i,j)<=dc))
        Density_aver=(Density(i)+Density(j))/2.; %average density
        if (Density_aver>bord_rho(cl(i))) 
          bord_rho(cl(i))=Density_aver;
        end
        if (Density_aver>bord_rho(cl(j))) 
          bord_rho(cl(j))=Density_aver;
        end
      end
    end
  end
  for i=1:NumIns
    if (Density(i)<bord_rho(cl(i)))
      halo(i)=0;
    end
  end
end
for i=1:NCLUST
  nc=0;
  nh=0;
  for j=1:NumIns
    if (cl(j)==i) 
      nc=nc+1;
    end
    if (halo(j)==i) 
      nh=nh+1;
    end
  end
 % fprintf('CLUSTER: %i CENTER: %i ELEMENTS: %i CORE: %i HALO: %i \n', i,icl(i),nc,nh,nc-nh);
end


end

