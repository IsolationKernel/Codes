function [f1,recall,precision,matrix] = fmeasure(ground_truth, class)
% Written by: Ye Zhu <ye.zhu@ieee.org>
% Altered by: Xiaoyu Qin <xiaoyu.qin@ieee.org>

matrix=zeros(max(ground_truth),max(class));
noise=zeros(max(ground_truth),1);
s=size(class,1);

for i = 1:s % test each example
    if class(i)>0
        matrix(ground_truth(i),class(i))=matrix(ground_truth(i),class(i))+1;
    else
        noise(ground_truth(i))=noise(ground_truth(i))+1;
    end
end

indF=zeros(max(ground_truth),1);

if size(matrix,2)~=0
    [f1,recall,precision, match]=fmean(matrix, noise);
    [posc,~]=find(match==1);
    for i=1:size(posc,1)
        indF(posc(i))=f1(i);
    end
    
    f1=sum(indF)/max(ground_truth);
    recall=sum(recall)/max(ground_truth);
    precision=sum(precision)/max(ground_truth);    
else
    f1=0;
    recall=0;
    precision=0;    
end

end

function [ f1,recall,precision, match ] = fmean( matrix, noise)

% first round
recall = calc_recall(matrix);
precision = calc_precision (matrix);
f1=2*precision.*recall./(precision+recall+0.0000001);

[match, ~] = hungarian(-f1);
matrix = [matrix noise];
[r,c]=size(matrix);
[rr,cc]=size(match);
match1=[match;zeros(r-rr,cc)];
match1=[match1 zeros(r,c-cc)];

% re-calculate
recall = calc_recall(matrix);
precision = calc_precision (matrix);
f1=2*precision.*recall./(precision+recall+0.0000001);

f1=f1(match1==1);
precision=precision(match1==1);
recall=recall(match1==1);

end

function [ m_recall ] = calc_recall (matrix)
m_recall=matrix;
sumrow=sum(matrix, 2);
if size(matrix,2)==1
    sumrow=matrix;
end
for j = 1:size(matrix,1)
	m_recall(j,:) = m_recall(j,:)/(sumrow(j)+0.0000001); % calculate the total positive examples 
end
end

function [ m_precision ] = calc_precision (matrix)
m_precision=matrix;
sumcol=sum(matrix);
if size(matrix,1)==1
	sumcol=matrix;
end
for j = 1:size(matrix,2)
	m_precision(:,j) = m_precision(:,j)/(sumcol(j)+0.0000001); % calculate the total positive examples 
end
end

