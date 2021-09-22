%  Estimated intrinsic dimensions as shown in Figure 3

clear
load('wGaussians.mat')
 
k=50;
theta = 0.975;
rng(42, 'twister');
X=data;
n = size(X,1);
% [idxmax,distsmax] = knnsearch(X,X,'K',k+1);
% idxmax = idxmax(:,2:end); % 2:end skips first neighbor - the point itself
% distsmax = distsmax(:,2:end);
% idx = idxmax(:,1:k);
% dists = distsmax(:,1:k);
%

dis=pdist2(data,data);
[distsmax,idxmax]=sort(dis,2);

idxmax = idxmax(:,2:end); % 2:end skips first neighbor - the point itself
distsmax = distsmax(:,2:end);
idx = idxmax(:,1:k);
dists = distsmax(:,1:k);

warning('off');
X=full(data);
parfor i = 1:n
    if mod(i,1000)==0, fprintf('\n%d',i); end
    KNN = X(idx(i,:),:);
    id_tle(i) = idtle(KNN,dists(i,:));
end
boxplot(id_tle)
set(gcf,'color','w');
set(gca,'linewidth',1,'fontsize',14,'fontname','Times');
ylabel('ID')
title('Estimated intrinsic dimensions, k=50')