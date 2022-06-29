function [TClass] = gkRegression(trainData,trainClass,data,id,index)

TestY={};
trainMSE=[];
for i=1:11
    sig= 2^(i-6); 

    m_dis=pdist2(trainData,trainData);
    
    simTrain = exp(-0.5*(m_dis.^2)./(2*sig^2));
    
    simTrain = simTrain - diag(diag(simTrain)); % replace diagonal elements with 0
    
    TrainY=simTrain*trainClass./(sum(simTrain,2)+eps); % kernel regression value for training data
    
    trainMSE(i)= immse(trainClass,TrainY); %MSE for training data    
     
    testData=data(id==index,:); 
    
    m_dis=pdist2(testData,trainData);
    
    simTest = exp(-0.5*(m_dis.^2)./(2*sig^2));     
    
    TestY{i}=simTest*trainClass./(sum(simTest,2)+eps); % kernel regression value for testing data
    
    % testMSE = immse(testClass,TestY) %MSE for testing data
    
end

[~,bi]=min(trainMSE);

TClass=TestY{bi};


end

