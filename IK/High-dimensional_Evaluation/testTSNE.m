% t-SNE visulaisation for Table 5 and Table 6

%% create 2 subspace clusters
clear
close all

n=500; % cluster size
w=5000; % cluster subspace dimensions
data=zeros(2*n,2*w);
class=data(:,1);
for i=1:2
  data(n*(i-1)+1:n*i,(i-1)*w+1:w*i)=(randn(n,w));  % for Table 5 in which both clusters are generated using N(0,1)
  %  data(n*(i-1)+1:n*i,(i-1)*w+1:w*i)=(randn(n,w)*i^5);    % for Table 6  in which clusters are generated using usingN(0,1)and N(0,32)
   class(n*(i-1)+1:n*i)=i;
end 
data(1,:)=data(1,:)-data(1,:); % add the origin point 
 
D=pdist2(data,data);

%% based on Gaussian kernel
per=30;
tic
P = d2p(D .^ 2, per, 1e-5);
toc

tic
ydata = tsne_p(P, class, 2);
toc
figure
gscatter(ydata(:,1),ydata(:,2),class)
hold on
scatter(ydata(1,1),ydata(1,2),400,'black','x','LineWidth',1.2)
title(['Gaussian kernel with ' 'perplexity=' num2str(per)])
legend('1','2','O')
%% based on Isolation kernel

psi=8;
t=200;
P = aNNE_similarity(D, psi, t);
for i=1:size(data,1)
    P(i,i)=0;
    P(i,:)=P(i,:)./sum(P(i,:));
end
ydata = tsne_p(P, class, 2);
toc
figure
gscatter(ydata(:,1),ydata(:,2),class)
hold on
scatter(ydata(1,1),ydata(1,2),400,'black','x','LineWidth',1.2) 
title(['Isolation kernel with' ' \psi=' num2str(psi)])
legend('1','2','O')

% FIt-SNE is based on Python https://github.com/KlugerLab/FIt-SNE
% q-SNE is based on Python https://github.com/i13abe/qSNE

