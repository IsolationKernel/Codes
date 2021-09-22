% Clustering evaluation results in Table 3

clear 
load('Gaussians.mat') 
disp('Gaussians')
% data normalisation
data = (data - min(data)).*((max(data) - min(data)).^-1);
data(isnan(data)) = 0.5; 
[DP_F1,AG_DP_F1,IK_DP_F1,SC_F1,AG_SC_F1,IK_SC_F1,DP_AMI,AG_DP_AMI,IK_DP_AMI,SC_AMI,AG_SC_AMI,IK_SC_AMI] = clustering(data,class); 
[DP_AMI,AG_DP_AMI,IK_DP_AMI,SC_AMI,AG_SC_AMI,IK_SC_AMI]
% F1 score and AMI score for two clustering algortihms (SC and DP). AG and
% IK indicate the similarity measure used in the clustering algorithm
 
   

clear 
load('wGaussians.mat') 
disp('wGaussians')
% data normalisation
data = (data - min(data)).*((max(data) - min(data)).^-1);
data(isnan(data)) = 0.5; 
[DP_F1,AG_DP_F1,IK_DP_F1,SC_F1,AG_SC_F1,IK_SC_F1,DP_AMI,AG_DP_AMI,IK_DP_AMI,SC_AMI,AG_SC_AMI,IK_SC_AMI] = clustering(data,class); 
[DP_AMI,AG_DP_AMI,IK_DP_AMI,SC_AMI,AG_SC_AMI,IK_SC_AMI]


clear 
load('ijcnn.mat') 
disp('ijcnn')
% data normalisation
data = (data - min(data)).*((max(data) - min(data)).^-1);
data(isnan(data)) = 0.5; 
[DP_F1,AG_DP_F1,IK_DP_F1,SC_F1,AG_SC_F1,IK_SC_F1,DP_AMI,AG_DP_AMI,IK_DP_AMI,SC_AMI,AG_SC_AMI,IK_SC_AMI] = clustering(data,class); 
[DP_AMI,AG_DP_AMI,IK_DP_AMI,SC_AMI,AG_SC_AMI,IK_SC_AMI]


clear 
load('A9a.mat') 
disp('A9a')
% data normalisation
data = (data - min(data)).*((max(data) - min(data)).^-1);
data(isnan(data)) = 0.5; 
[DP_F1,AG_DP_F1,IK_DP_F1,SC_F1,AG_SC_F1,IK_SC_F1,DP_AMI,AG_DP_AMI,IK_DP_AMI,SC_AMI,AG_SC_AMI,IK_SC_AMI] = clustering(data,class); 
[DP_AMI,AG_DP_AMI,IK_DP_AMI,SC_AMI,AG_SC_AMI,IK_SC_AMI]


clear 
load('Mnist.mat') 
disp('Mnist')
% data normalisation
data = (data - min(data)).*((max(data) - min(data)).^-1);
data(isnan(data)) = 0.5; 
[DP_F1,AG_DP_F1,IK_DP_F1,SC_F1,AG_SC_F1,IK_SC_F1,DP_AMI,AG_DP_AMI,IK_DP_AMI,SC_AMI,AG_SC_AMI,IK_SC_AMI] = clustering(data,class); 
[DP_AMI,AG_DP_AMI,IK_DP_AMI,SC_AMI,AG_SC_AMI,IK_SC_AMI]


clear 
load('cifar10.mat') 
disp('cifar10')
% data normalisation
data = (data - min(data)).*((max(data) - min(data)).^-1);
data(isnan(data)) = 0.5; 
[DP_F1,AG_DP_F1,IK_DP_F1,SC_F1,AG_SC_F1,IK_SC_F1,DP_AMI,AG_DP_AMI,IK_DP_AMI,SC_AMI,AG_SC_AMI,IK_SC_AMI] = clustering(data,class); 
[DP_AMI,AG_DP_AMI,IK_DP_AMI,SC_AMI,AG_SC_AMI,IK_SC_AMI]



clear 
load('realsim.mat') 
disp('realsim')
% data normalisation
data = (data - min(data)).*((max(data) - min(data)).^-1);
data(isnan(data)) = 0.5; 
[DP_F1,AG_DP_F1,IK_DP_F1,SC_F1,AG_SC_F1,IK_SC_F1,DP_AMI,AG_DP_AMI,IK_DP_AMI,SC_AMI,AG_SC_AMI,IK_SC_AMI] = clustering(data,class); 
[DP_AMI,AG_DP_AMI,IK_DP_AMI,SC_AMI,AG_SC_AMI,IK_SC_AMI]