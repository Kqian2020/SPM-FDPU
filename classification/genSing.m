function [Single_target] = genSing( train_target)
% num_label*num_instance
% label value 1/0
train_target(train_target==-1)=0;
Single_target = zeros(size(train_target));
for i=1:size(train_target,2)
    if sum(train_target(:,i))==1
        Single_target(:,i) = train_target(:,i);
        continue
    end
    if sum(train_target(:,i))==0
        continue
    end
    pos = find(train_target(:,i)==1);
    pi = randperm(length(pos));
    pi = pi(1);
    Single_target(pos(pi),i) = 1;
end
assert(sum(Single_target(:))==size(train_target,2),'genrate single positive label error')
Single_target(Single_target==0)=-1;
end