function[trainset, testset, trainlabel, testlabel] = datasetsplit(Y, L, ratio)

%split dataset, input are data, labels and split ratio


% ratio = 0.5;

train = round(length(Y)*ratio);

trainset = Y(1:train,:);
testset = Y(train+1:end, :);

trainlabel = L(1:train,:);
testlabel = L(train+1:end, :);




