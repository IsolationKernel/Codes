%% data 

clear
close all
iter=65;

rng(5)
data1=randn(500,2);

data2=randn(300,2)*5+5;

class1=zeros(size(data1,1),1)+1;
class2=zeros(size(data2,1),1)+2;


data=[data1;data2];

class=[class1;class2];

data = (data - min(data)).*((max(data) - min(data)).^-1);
data(isnan(data)) = 0.5; % data normalisation

data(1,:)=mean(data(class==1,:));
data(end,:)=mean(data(class==2,:));
figure('Renderer', 'painters', 'Position', [400 400 420 400])
gscatter(data(:,1),data(:,2),class,'rb') 
set(gca,'FontSize',10);
set(gcf,'color','w');
set(gca,'linewidth',1,'fontsize',20,'fontname','Times');
xlim([0 1])
ylim([0 1])
box on
 
 %% IKDC 
 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 
 t=500;
v=0.9;
psi=16;

s=min(size(data,1),10000);
k=size(unique(class),1); 
Kn=240;  
  
rng(33)
[ndata] = iNNEspace(data,data, psi, t); % inne feature space
[Tclass,Centre,GP,it] =IDKModes(ndata,k,Kn,v,s);
[AMI]=nmi(class,Tclass+1) 


gscatter(data(:,1),data(:,2),Tclass,'rb','...',6) 
hold on
scatter(data(Centre{end},1),data(Centre{end},2),300,'g.')
set(gca,'FontSize',10);
set(gcf,'color','w');
set(gca,'linewidth',1,'fontsize',20,'fontname','Times');
xlim([0 1])
ylim([0 1])
box on

legend('1', '2', 'Centre')

%% DP

dc= 0.3300;
IKDist = pdist2(data,data,'minkowski',2);
IKDist = IKDist./max(max(IKDist));

Density=sum(IKDist'<=dc)';
 
        [ Z ] = LCHierDP( IKDist,Density');
        Tclass = cluster(Z,'maxclust',k);
nmi(class',Tclass')

maxd=max(max(IKDist)); % max dis
NumIns=size(IKDist,2);
MinDist=Density-Density;
[~,SortDensity]=sort(Density,'descend'); % SortDensity is index
MinDist(SortDensity(1))=-1.;
nneigh(SortDensity(1))=0;

for ii=2:NumIns
    MinDist(SortDensity(ii))=maxd;
    for jj=1:ii-1
        if(IKDist(SortDensity(ii),SortDensity(jj))<MinDist(SortDensity(ii)))
            MinDist(SortDensity(ii))=IKDist(SortDensity(ii),SortDensity(jj));
            nneigh(SortDensity(ii))=SortDensity(jj); % nearest neigbour index
        end
    end
end
MinDist(SortDensity(1))=max(MinDist(:));

Density=tiedrank(Density)+0.0000000001;
MinDist=tiedrank(MinDist)+0.0000000001;

Mult=(Density).*(MinDist);
[VSortMult,ISortMult]=sort(Mult,'descend');
ID=ISortMult(1:k); 
 
figure
gscatter(data(:,1),data(:,2),Tclass,'br','...',6) 
hold on
scatter(data(ID,1),data(ID,2),300,'g.')
set(gca,'FontSize',10);
set(gcf,'color','w');
set(gca,'linewidth',1,'fontsize',20,'fontname','Times');
xlim([0 1])
ylim([0 1])
box on

legend('1', '2', 'Centre')


%% LC-DP
IKDist=pdist2(data,data,'minkowski',2);;
dc=0.0790;
kk=80
Density=sum(IKDist'<=dc)';
[LC] = getLC(IKDist,Density,kk);
[ Z ] = LCHierDP( IKDist,LC');
Tclass = cluster(Z,'maxclust',k);
nmi(class',Tclass')
 
IKDist = pdist2(data,data,'minkowski',2);
IKDist = IKDist./max(max(IKDist));

Density=sum(IKDist'<=dc)';
[Density] = getLC(IKDist,Density,Kn);


maxd=max(max(IKDist)); % max dis
NumIns=size(IKDist,2);
MinDist=Density-Density;
[~,SortDensity]=sort(Density,'descend'); % SortDensity is index
MinDist(SortDensity(1))=-1.;
nneigh(SortDensity(1))=0;

for ii=2:NumIns
    MinDist(SortDensity(ii))=maxd;
    for jj=1:ii-1
        if(IKDist(SortDensity(ii),SortDensity(jj))<MinDist(SortDensity(ii)))
            MinDist(SortDensity(ii))=IKDist(SortDensity(ii),SortDensity(jj));
            nneigh(SortDensity(ii))=SortDensity(jj); % nearest neigbour index
        end
    end
end
MinDist(SortDensity(1))=max(MinDist(:));

Density=tiedrank(Density)+0.0000000001;
MinDist=tiedrank(MinDist)+0.0000000001;

Mult=(Density).*(MinDist);
[VSortMult,ISortMult]=sort(Mult,'descend');
ID=ISortMult(1:k); 

figure
gscatter(data(:,1),data(:,2),Tclass,'br','...',6) 
hold on
scatter(data(ID,1),data(ID,2),300,'g.')
set(gca,'FontSize',10);
set(gcf,'color','w');
set(gca,'linewidth',1,'fontsize',20,'fontname','Times');
xlim([0 1])
ylim([0 1])
box on

legend('1', '2', 'Centre')