function [ Result ] = evalt(Fpred, Ygnd, thr, flag)
%%
% Fpred: L*N predicted values
% Ypred: L*N predicted labels
% Ygnd: L*N groundtruth labels
% thr: threshold value
% flag: default value is true
%%
if flag
    % default
    Ypred = sign(Fpred);
else
    Ypred = sign(Fpred-thr);
end

%% ExampleBased
Result.AveragePrecision = Average_precision(Fpred,Ygnd);

Result.Coverage = coverage(Fpred,Ygnd);

Result.OneError = One_error(Fpred,Ygnd);

Result.RankingLoss = Ranking_loss(Fpred,Ygnd);

Result.HammingLoss = Hamming_loss(Ypred,Ygnd);

Result.myAccuracy = myAccuracy(Ypred,Ygnd);

%% LabelBased
Result.AvgAuc = avgauc(Fpred,Ygnd);