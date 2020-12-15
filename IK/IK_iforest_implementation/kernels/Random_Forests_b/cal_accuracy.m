function [accuracy] = cal_accuracy(y,pred)
accuracy = numel(find(y-pred==0))/numel(y);