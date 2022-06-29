function [TClass] = IKregression(trainData,trainClass,data,id,index,method)
 

TestY={};
trainMSE=[];
for i=1:11
    psi= 2^(i+1);
    t=100;
    
    if psi>size(trainData,1)
        break
    end
    
    
    switch method
        case 'IKDE_U'
            [ndata] = uNNEspace(trainData,data, psi, t); % Isolation features obtained with iNNE
        case 'IKDE_V'
           [ndata] = aNNEspace(trainData,data, psi, t); % Isolation features obtained with iNNE
        case 'IKDE_S'
           [ndata] = iNNEspace(trainData,data, psi, t); % Isolation features obtained with iNNE
        otherwise
            disp('using IKDE_V as default')
            [ndata] = aNNEspace(trainData,data, psi, t); % Isolation features obtained with iNNE
    end
  
        
    ndata=sparse(ndata);
    
    ntrainData=ndata(id~=index,:);
    
    ntestData=ndata(id==index,:);
    
    simTrain=(ntrainData*ntrainData')./t; % get the similarity matrix for training data
    
    simTrain = simTrain - diag(diag(simTrain)); % replace diagonal elements with 0
    
    TrainY=simTrain*trainClass./(sum(simTrain,2)+eps); % kernel regression values on training data
    
    trainMSE(i)= immse(trainClass,TrainY); %MSE for training data
    
    
    
    simTest=full(ntestData*ntrainData')./t; % get the similarity matrix between testing and training data
    
    TestY{i}=simTest*trainClass./(sum(simTest,2)+eps); % kernel regression value for testing data
    
     
end

[~,bi]=min(trainMSE); % identifying the minimum MSE, i.e., the best psi 

TClass=TestY{bi}; % return the corresponding predicted values on testing data


end

