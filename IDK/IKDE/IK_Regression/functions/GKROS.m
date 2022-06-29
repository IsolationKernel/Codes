function [TClass] = GKROS(trainData,trainClass,data,id,index)


%%%%%%%%%%%%%%%%%% train data

kk=[1, 3, 5, 7, 11, 21, 51, 101, 201, 501, 1001, 2001];
kk(kk>size(trainData,1))=[];
Idx = knnsearch(trainData,trainData,'K',size(trainData,1));
mdist=pdist2(trainData,trainData);


testData=data(id==index,:);

Idx2 = knnsearch(trainData,testData,'K',size(trainData,1));

Tmdist=pdist2(testData,trainData);


for i=1:length(kk)
    
    k=kk(i);
    
    Band=[];
    
    for ci=1:size(trainData,1)
        Band(ci)=max(mean(mdist(ci,Idx(ci,2:k+1))),eps);
    end
    
    simTrain=zeros(size(mdist));
    
    for ii=1:size(trainData,1)
        cmdist=mdist(ii,:);
        simTrain(ii,:)= exp(-0.5*(cmdist.^2)./(2*Band(ii)^2));
        simTrain(ii,Idx(ii,k+2:end))=0;
    end
    
    simTrain = simTrain - diag(diag(simTrain)); % replace diagonal elements with 0
    
    TrainY=simTrain*trainClass./(sum(simTrain,2)+eps); % kernel regression value for training data
    
    trainMSE(i)= immse(trainClass,TrainY); %MSE for training data
    
    
    
    
    TBand=[];
    for ci=1:size(testData,1)
        TBand(ci)=max(mean(Tmdist(ci,Idx2(ci,1:k))),eps);
    end
    
    simTest=zeros(size(Tmdist));
    
    for ij=1:size(testData,1)
        cmdist=Tmdist(ij,:);
        simTest(ij,:)= exp(-0.5*(cmdist.^2)./(2*TBand(ij)^2));
        simTest(ij,Idx2(ij,k+1:end))=0;
    end
    
    TestY{i}=simTest*trainClass./(sum(simTest,2)+eps); % kernel regression value for testing data
    
end

[~,bi]=min(trainMSE);

TClass=TestY{bi};


end