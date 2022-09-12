% This Matlab code is used for demonstration of using t-SNE with Isolation kernel.
% The results are similar to Table 1 in the JAIR paper:
% Zhu, Y. and Ting, K.M., 2021, July. Improving the Effectiveness and
% Efficiency of Stochastic Neighbour Embedding with Isolation Kernel. 
% Journal of Artificial Intelligence Research.

clear
clc
close all

%% create 5 subspace clusters

n=250;
d=10;
b=zeros(5*n,5*d);
class=b(:,1);
for i=1:3
   b(n*(i-1)+1:n*i,(i-1)*d+1:d*i)=(randn(n,d)*i^4);    
   class(n*(i-1)+1:n*i)=i;
end
for i=4:5
   b(n*(i-1)+1:n*i,(i-1)*d+1:d*i)=(randn(n,d)*i^4)+100*i;    
   class(n*(i-1)+1:n*i)=i;
end
b(1,:)=b(1,:)-b(1,:); % add the origin point 

%% t-SNE with Gaussian kernel

per=250; 
tic
D=pdist2(b,b); 
[P B] = d2p(D .^ 2, per, 1e-5); 
ydata1 = tsne_p(P, class, 2);  
ydata1=normalize(ydata1);
time_c = toc;
disp(strcat("Time consumed by t-SNE: ", num2str(time_c)));
figure
gscatter(ydata1(:,1),ydata1(:,2),class)
hold on
scatter(ydata1(1,1),ydata1(1,2),400,'black','x','LineWidth',1.2)
legend('1','2','3','4','5','O')
title(['Gaussian kernel with perplexity=' num2str(per)])
set(gcf,'color','w');
set(gca,'linewidth',1,'fontsize',18,'fontname','Times'); 
 

 %% t-SNE with Isolation kernel
 
psi=per;
tic
D=pdist2(b,b); 
[ ~, sim ] = aNNE (D, psi, 200);
for i=1:size(b,1)
    sim(i,i)=0;
    sim(i,:)=sim(i,:)./sum(sim(i,:));
end
ydata = tsne_p(sim, class, 2);
ydata=normalize(ydata); 
time_c = toc;
disp(strcat("Time consumed by aNNE t-SNE: ", num2str(time_c)));
figure
gscatter(ydata(:,1),ydata(:,2),class)
hold on
scatter(ydata(1,1),ydata(1,2),400,'black','x','LineWidth',1.2)
legend('1','2','3','4','5','O')
title(['Isolation kernel with \psi=' num2str(psi)])
set(gcf,'color','w');
set(gca,'linewidth',1,'fontsize',18,'fontname','Times'); 
  
 