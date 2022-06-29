function [TClass] = anneRegression(trainData,trainClass,data,id,index)
 

TestY={};
trainMSE=[];

%% original version
 
for i=1:12
    psi= 2^i;
    t=100;
    
    if psi>size(trainData,1)
        break
    end
     
    [ndata] = aNNEspace (trainData,data, psi, t); % Isolation features obtained with aNNE
    
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
 
%% fast version
% 
% for i=1:12
%     psi= 2^i;
%     t=100;
%     
%     if psi>size(trainData,1)
%         break
%     end
%     
%     %   [ndata] = iNNEspace(trainData,data, psi, t); % Isolation features obtained with iNNE
%     [ndata] = aNNEspace (trainData,data, psi, t); % Isolation features obtained with aNNE
%     
%     
%     ntrainData=ndata(id~=index,:);
%     
%     ntestData=ndata(id==index,:);
%     
%     
%     Wr=sum(ntrainData.*repmat(trainClass,1,size(ntrainData,2)));
%     W=sum(ntrainData);
%     
%     % leave one out
%     SelfSim=trainClass-trainClass;
%     for ij=1:size(ntrainData,1)
%         SelfSim(ij)=ntrainData(ij,:)*ntrainData(ij,:)';
%     end
%     
%     TrainY=(ntrainData*Wr'-SelfSim.*trainClass)./(ntrainData*W'-SelfSim+eps);    
%     
%     
%     %     simTrain=full(ntrainData*ntrainData')./t;
%     %
%     %     simTrain = simTrain - diag(diag(simTrain)); % replace diagonal elements with 0
%     %
%     %     TrainY2=simTrain*trainClass./(sum(simTrain,2)+eps); % kernel regression value for training data
%     %     toc
%     %      sum(TrainY2-TrainY)
%     
%     trainMSE(i)= immse(trainClass,TrainY); %MSE for training data
%     
%     
%     
%     Wr=sum(ntrainData.*repmat(trainClass,1,size(ntrainData,2)));
%     W=sum(ntrainData);
%     TestY{i}=(ntestData*Wr')./(ntestData*W'+eps);
%     
%     
%     %     simTest=full(ntestData*ntrainData')./t;
%     %
%     %     TestY{i}=simTest*trainClass./(sum(simTest,2)+eps); % kernel regression value for testing data
%     
%     % testMSE = immse(testClass,TestY) %MSE for testing data
%     
% end
% 
% [~,bi]=min(trainMSE);
% 
% TClass=TestY{bi};
 
end

