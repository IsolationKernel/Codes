clear
%% load data
data=load('http.csv');

ADLabels=data(:,end);
num_out = sum(ADLabels);
Data=data(:,1:end-1);

%% normalize
max_dim = max(Data);
min_dim = min(Data);
gap = max_dim-min_dim;
gap(gap==0) = 1;
Data = (Data-min_dim)./(gap);
 
% parameters of ISK
psi = 128; 
voro_num =100; %400
    

auc = zeros(rounds, 2);
mtime = zeros(rounds, 2);
rseed = zeros(rounds, 1);


for r = 1:rounds
    disp(['rounds ', num2str(r), ':']);
    tic
    
    
    % ISK
    tmp = toc;
    voros = build_voro_iNNE(Data,[],psi,voro_num,1);
    %voros = build_voros_zero_split(Data,[],psi,voro_num,1);
    %voros = build_voro_inter(Data,[],psi,voro_num,1);
    iv = convert_point_iNNE(Data,voros , voro_num, psi);
    tmp_mid = toc;
    train = tmp_mid-tmp
    %iv = convert_point(Data,voros , voro_num, psi);
    iv_mean = mean(iv);
    ISK_score = -iv* iv_mean';
    auc(r,2) = Measure_AUC(ISK_score, ADLabels);
    disp(['auc = ', num2str(auc(r,2)), '.']);  
    test = toc-tmp_mid
    
    % norm
    tmp = toc;
    Score = -sum(iv,2);
    auc(r,1) = Measure_AUC(Score, ADLabels);
    disp(['auc = ', num2str(auc(r,1)), '.']);
    test = toc-tmp
end

