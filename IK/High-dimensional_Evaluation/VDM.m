% Generating results for Figure 1 (b)

%% Gaussian
clear
load('Gaussians.mat');  % please unzip Gaussians under "Clustering and Indexing datasets"
d=10000; % d=10 or 10000
data=data(:,1:d);

Gadis = Gaussian1(data, 5);
Gadis=Gadis./max(max(Gadis));
Gadis=1-Gadis;
den=Gadis(end-1,:);
spa=Gadis(end,:);
cen=Gadis(end-2,:);

std1=std(den(class==1));
std2=std(spa(class==2));
std3=std(cen(class==1));
std4=std(cen(class==2));
mean1=mean(den(class==1));
mean2=mean(spa(class==2));
mean3=mean(cen(class==1));
mean4=mean(cen(class==2));

vdm1=(std1.^2)/mean1
vdm2=(std2.^2)/mean2
vdm3=(std3.^2)/mean3
vdm4=(std4.^2)/mean4

%% AG
clear
load('Gaussians.mat');  % please unzip Gaussians under "Clustering and Indexing datasets"
d=10000; % d=10 or 10000
data=data(:,1:d);

k=ceil(0.1*size(data,1));
AG = KNN_AdaptiveGaussian1(data, k, 2);
AG=AG./max(max(AG));
den=AG(end-1,:);
spa=AG(end,:);
cen=AG(end-2,:);

std1=std(den(class==1));
std2=std(spa(class==2));
std3=std(cen(class==1));
std4=std(cen(class==2));
mean1=mean(den(class==1));
mean2=mean(spa(class==2));
mean3=mean(cen(class==1));
mean4=mean(cen(class==2));

vdm1=(std1.^2)/mean1
vdm2=(std2.^2)/mean2
vdm3=(std3.^2)/mean3
vdm4=(std4.^2)/mean4

%% IK
clear
load('Gaussians.mat');  % please unzip Gaussians under "Clustering and Indexing datasets"
d=10000; % d=10 or 10000
data=data(:,1:d);
dis=pdist2(data,data);
dis=dis./max(max(dis));

t=100;
psi=32;  
[ ~, proximity ] = aNNE (dis, psi, t);
dis_anne = 1 - proximity;
  
den=dis_anne(end-1,:);
spa=dis_anne(end,:);
cen=dis_anne(end-2,:);

std1=std(den(class==1));
std2=std(spa(class==2));
std3=std(cen(class==1));
std4=std(cen(class==2));
mean1=mean(den(class==1));
mean2=mean(spa(class==2));
mean3=mean(cen(class==1));
mean4=mean(cen(class==2));

vdm1=(std1.^2)/mean1
vdm2=(std2.^2)/mean2
vdm3=(std3.^2)/mean3
vdm4=(std4.^2)/mean4

%% SNN
clear
load('Gaussians.mat');  % please unzip Gaussians under "Clustering and Indexing datasets"
d=10000; % d=10 or 10000
data=data(:,1:d);
dis=pdist2(data,data);
dis=dis./max(max(dis));

k=ceil(size(data,1)*0.3);
[S,~] = JSNN(data,k,2);
S=1-S;
S=S./max(max(S));
  
den=S(end-1,:);
spa=S(end,:);
cen=S(end-2,:);

std1=std(den(class==1));
std2=std(spa(class==2));
std3=std(cen(class==1));
std4=std(cen(class==2));
mean1=mean(den(class==1));
mean2=mean(spa(class==2));
mean3=mean(cen(class==1));
mean4=mean(cen(class==2));

vdm1=(std1.^2)/mean1
vdm2=(std2.^2)/mean2
vdm3=(std3.^2)/mean3
vdm4=(std4.^2)/mean4