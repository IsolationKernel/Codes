% Calculate N_e numbers under different distance measures and kernels as shown in Figure 1.
clear
load('Gaussians.mat');  % please unzip Gaussians under "Clustering and Indexing datasets"
load('nGaussians.mat'); 

% adding query points
data(2000,:)=[];
class(2000,:)=[];
data=[data;nd]; 
class=[class;nc];
 
gscatter(data(:,1),data(:,2),class,'rgb');

d=10000; % d=10 or 10000
data=data(:,1:d);
e=0.005; % Calculate the ratio of the circle radius to the distance to the nearest point

%% IK
t=1000;
psi=32; 

dis=pdist2(data,data,'minkowski',0.5);
dis=dis./max(max(dis)); 

num=0;
n = 10; % repeat times

for i = 1:n
   [ ~, proximity ] = aNNE (dis, psi, t); % data dependent IK
%    [ ~, proximity ] = UaNNE (data, psi, t); % data independent IK
   dis_anne = 1 - proximity; 
 %  spa=dis_anne(end,1:1999); % Query point is in the center of sparse cluster
 %  den=dis_anne(end-1,1:1999); %  Query point is in the center of dense cluster
   cen=dis_anne(end-2,1:1999); % Query point is between 2 clusters
   ls = min(cen)*(1+e); % cen can be replaced with den or spa 
   num = num+sum(cen<ls);  % cen can be replaced with den or spa 
end
num = num/n;

%% Linear Kernel

% LK = Linear1(data);
% cen=LK(end-2,1:1999); % Query point is between 2 clusters
% %spa=LK(end,1:1999); Query point is in the center of sparse cluster
% %den=LK(end-1,1:1999); %  Query point is in the center of dense cluster
% ls = min(cen)*(1+e); % cen can be replaced with den or spa 
% num = sum(cen<ls);

%% Gaussian Kernel

% sigma=5;
% Gadis = Gaussian1(data, sigma);
% cen=Gadis(end-2,1:1999); 
% %spa=Gadis(end,1:1999);
% ls = min(cen)*(1+e);
% num = sum(cen<ls);

%% AG

% k=ceil(0.1*size(data,1));
% AG = KNN_AdaptiveGaussian1(data, k, 0.1);
% AG=AG./max(max(AG));
% cen=AG(end-2,1:1999);
% %spa=AG(end,1:1999);
% ls = min(cen)*(1+e);
% num = sum(cen<ls);

%% SNN

% % k=ceil(size(data,1)*0.7);
% k = 1200;
% [~,S] = JSNN(data,k, 0.5);
% S=1-S;
% S=normalize(S);
% cen=S(end-2,1:1999);
% %spa=S(end,1:1999);
% ls = min(cen)*(1+e);
% num = sum(cen<ls);
 
%% lp
 
% p=0.1; %p=0.1,0.5 & 2
% dis=pdist2(data,data,'minkowski',p);
% cen=dis(end-2 ,1:1999);
% %spa=S(end,1:1999);
% ls = min(cen)*(1+e);
% num = sum(cen<ls);


