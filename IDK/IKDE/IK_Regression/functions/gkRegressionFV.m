function [TClass] = gkRegressionFV(trainData,trainClass,data,id,index)
 

TestY={};
trainMSE=[];
for i=1:11
    sig= 2^(i-6); 
    t=ceil(size(trainData,1)^0.5);
 
   % [ndata] = GKfeature(data,t,sig); 
    [ndata] = sNstr(data,t,sig);  
    
    ntrainData=ndata(id~=index,:);
    
    ntestData=ndata(id==index,:);
    
    simTrain=full(ntrainData*ntrainData')./t; % get the similarity matrix for training data
    
    simTrain = simTrain - diag(diag(simTrain)); % replace diagonal elements with 0
    
    TrainY=simTrain*trainClass./(sum(simTrain,2)+eps); % kernel regression values on training data
    
    trainMSE(i)= immse(trainClass,TrainY); %MSE for training data
    
    
    
    simTest=full(ntestData*ntrainData')./t; % get the similarity matrix between testing and training data
    
    TestY{i}=simTest*trainClass./(sum(simTest,2)+eps); % kernel regression value for testing data
    
end

[~,bi]=min(trainMSE); % identifying the minimum MSE, i.e., the best psi 

TClass=TestY{bi}; % return the corresponding predicted values on testing data


end

