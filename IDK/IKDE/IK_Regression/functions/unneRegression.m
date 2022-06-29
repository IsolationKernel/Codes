function [TClass] = unneRegression(trainData,trainClass,data,id,index)
 
TestY={};
trainMSE=[];
for i=1:12
    psi= 2^i;
    t=100;
    
    if psi>size(trainData,1)
        break
    end    
   
   [ndata] = uNNEspace(trainData,data, psi, t); % Isolation features obtained with aNNE
        
    ndata=sparse(ndata);
        
    ntrainData=ndata(id~=index,:);
    
    ntestData=ndata(id==index,:);
    
    simTrain=(ntrainData*ntrainData')./t;
    
    simTrain = simTrain - diag(diag(simTrain)); % replace diagonal elements with 0
    
    TrainY=simTrain*trainClass./(sum(simTrain,2)+eps); % kernel regression value for training data
    
    trainMSE(i)= immse(trainClass,TrainY); %MSE for training data
    
    
    
    simTest=full(ntestData*ntrainData')./t;
    
    TestY{i}=simTest*trainClass./(sum(simTest,2)+eps); % kernel regression value for testing data
    
    % testMSE = immse(testClass,TestY) %MSE for testing data
    
end

[~,bi]=min(trainMSE);

TClass=TestY{bi};


end

